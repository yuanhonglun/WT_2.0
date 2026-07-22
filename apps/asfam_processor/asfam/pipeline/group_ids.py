"""Give each sample's dedup group ids their own range before alignment.

Stages 4 (isotope), 5 (adduct), 5b (spectral) and 6 (ISF) each number their
groups from a counter that restarts at zero for every sample, and the offsets
they carry (+100000 / +200000 / +300000) only hold the four stages apart
*within* one sample. Nothing holds one sample apart from the next, so sample A's
group 1006 and sample B's group 1006 are unrelated groups wearing the same id.

Alignment copies the id straight off the spot's representative, and the GUI
groups the aligned table by equality on ``duplicate_group_id``. The two unrelated
groups therefore fuse into one displayed group: that is how an m/z 609 feature
came to show an m/z 426 feature, 2 min away, as its M+1.
"""
from __future__ import annotations

from asfam.models import CandidateFeature


#: Width of one sample's id range. It has to exceed any single sample's group
#: ids: stage 5b numbers the highest of the four, from ``300000 + n_flagged``,
#: so a sample's ceiling is 300000 plus its feature count. The largest sample we
#: have run holds ~22k features, which leaves this stride a wide margin.
SAMPLE_STRIDE = 1_000_000

#: ``isotope_index`` is deliberately not in here. It is a position within a group
#: (0 = monoisotopic, n = M+n), not an id: it does not collide across samples,
#: and offsetting it would destroy it.
GROUP_ID_FIELDS = ("isotope_group_id", "adduct_group_id", "duplicate_group_id")


def namespace_group_ids(
    features_by_sample: dict[str, list[CandidateFeature]],
) -> None:
    """Offset every sample's group ids into a range of its own, in place.

    Done here, where the spill is read back for alignment, rather than at the
    four stages that write the ids: the checkpoint fingerprint covers the config,
    not the stage output (see :func:`asfam.io.spill.config_fingerprint`), so a
    ``_work/`` spill written before this fix still passes ``sample_is_complete``
    and is still reused — with its colliding ids intact. Remapping on the way in
    repairs those checkpoints too, without recomputing a single sample.

    The offset is keyed on the sample's position in the *sorted* id list, so it
    is a property of which samples are in the run and not of the order they were
    processed in — the same inputs always produce the same ids.
    """
    for index, sample_id in enumerate(sorted(features_by_sample)):
        offset = index * SAMPLE_STRIDE
        for feature in features_by_sample[sample_id]:
            for field in GROUP_ID_FIELDS:
                value = getattr(feature, field)
                if value is None:
                    continue
                if not 0 <= value < SAMPLE_STRIDE:
                    # Out of range means the sample's ids have grown into the
                    # next sample's slot, which is the very collision this
                    # function exists to prevent — and it would be invisible in
                    # the export. Fail loudly instead; ``_work/`` survives, so a
                    # re-run with a wider stride costs only stages 7-8.
                    raise ValueError(
                        f"sample {sample_id}: {field}={value} does not fit the "
                        f"{SAMPLE_STRIDE} id range reserved per sample; widen "
                        f"SAMPLE_STRIDE"
                    )
                setattr(feature, field, value + offset)
