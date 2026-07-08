"""Adduct-pair grouping for already-isotope-resolved feature lists.

The isotope estimator (``isotope_estimator.estimate_isotopes``) collapses
M+1/M+2 peaks of the same compound under one monoisotopic representative.
This module then walks the remaining representatives and tries to find
adduct pairs ([M+H]+ vs [M+Na]+ vs [M+NH4]+ etc.) by checking that the
neutral masses computed under each adduct's m/z formula agree within
tolerance.

Only the *representatives* (``is_duplicate=False``) are considered as
candidates: isotope satellites cannot be adducts.

Mutates features in place:

- The brightest member of an adduct group keeps ``is_duplicate=False``
  and gets ``adduct_type='[M+H]+'`` (or whatever the inferred base
  adduct is).
- Each non-representative gets ``is_duplicate=True``,
  ``duplicate_type='adduct'``, ``adduct_type='[M+Na]+'`` etc.,
  ``duplicate_group_id``, ``adduct_group_id``.

The intent matches MS-DIAL's "search representative adducts among
co-eluting peaks" step.

The ``duplicate_group_id`` written here is offset by
``DUP_GID_ADDUCT_OFFSET`` to keep its numeric range disjoint from
isotope group ids (the GUI groups features by ``duplicate_group_id``;
without an offset an isotope group and an adduct group can both reach
id 29 and visually merge into one cluster). Mirrors the same trick used
in ASFAM stage 5.
"""
from __future__ import annotations

from typing import List

from metabo_core.algorithms.mass_utils import check_adduct_pair
from metabo_core.models import CandidateFeature

# Adduct duplicate_group_id offset: keeps the adduct id namespace
# disjoint from the isotope estimator's namespace so the GUI's
# group-by-duplicate_group_id view doesn't accidentally fuse two
# unrelated groups. 100_000 is well above any realistic per-run isotope
# group count.
DUP_GID_ADDUCT_OFFSET = 100_000


def group_adducts(
    features: List[CandidateFeature],
    ionization_mode: str = "positive",
    mw_tolerance: float = 0.02,
    rt_tolerance: float = 0.06,
    group_id_start: int = 0,
) -> int:
    """Identify adduct partners among feature representatives.

    Returns the next available adduct_group_id.
    """
    # Representatives only: isotope satellites already labelled
    # is_duplicate=True and must be skipped.
    rep_indices = [
        i for i, f in enumerate(features)
        if not f.is_duplicate
    ]
    if len(rep_indices) < 2:
        return group_id_start

    rep_indices.sort(key=lambda i: features[i].rt_apex)

    # Adjacency on co-eluting representatives.
    adjacency: dict[int, set[int]] = {i: set() for i in rep_indices}
    pair_labels: dict[tuple[int, int], tuple[str, str]] = {}

    for pos, i in enumerate(rep_indices):
        fi = features[i]
        for pos2 in range(pos + 1, len(rep_indices)):
            j = rep_indices[pos2]
            fj = features[j]
            if fj.rt_apex - fi.rt_apex > rt_tolerance:
                break  # sorted by RT — further pairs are too distant
            if abs(fj.rt_apex - fi.rt_apex) > rt_tolerance:
                continue
            pair = check_adduct_pair(
                fi.precursor_mz, fj.precursor_mz,
                ionization_mode=ionization_mode,
                mw_tolerance=mw_tolerance,
            )
            if pair is None:
                continue
            adjacency[i].add(j)
            adjacency[j].add(i)
            pair_labels[(i, j)] = pair
            pair_labels[(j, i)] = (pair[1], pair[0])

    # Connected components.
    visited: set[int] = set()
    raw_components: list[list[int]] = []
    for start in rep_indices:
        if start in visited or not adjacency[start]:
            continue
        comp = []
        stack = [start]
        while stack:
            x = stack.pop()
            if x in visited:
                continue
            visited.add(x)
            comp.append(x)
            stack.extend(adjacency[x] - visited)
        raw_components.append(comp)

    # Pairwise RT checks above don't prevent transitive RT-chain
    # extension (A-B and B-C link even when A-C is far apart). Split
    # each component into RT-contiguous sub-components capped by
    # ``rt_tolerance``, mirroring the ASFAM stage 4 fix.
    components: list[list[int]] = []
    for comp in raw_components:
        if len(comp) <= 1:
            components.append(comp)
            continue
        sorted_comp = sorted(comp, key=lambda idx: features[idx].rt_apex)
        sub = [sorted_comp[0]]
        for idx in sorted_comp[1:]:
            if features[idx].rt_apex - features[sub[-1]].rt_apex <= rt_tolerance:
                sub.append(idx)
            else:
                components.append(sub)
                sub = [idx]
        components.append(sub)

    group_id = group_id_start
    for comp in components:
        if len(comp) < 2:
            continue
        # Brightest peak is the representative.
        rep_idx = max(comp, key=lambda i: features[i].ms1_height or 0.0)
        # Look up an adduct label for the rep from any pair it's in.
        rep_label = "[M+H]+" if ionization_mode == "positive" else "[M-H]-"
        for idx in comp:
            if idx == rep_idx:
                continue
            label_pair = pair_labels.get((rep_idx, idx))
            if label_pair is not None:
                rep_label = label_pair[0]
                break
        # Disjoint id namespace so the GUI's group-by-dup_gid view
        # doesn't merge unrelated isotope and adduct groups that happen
        # to share an integer id.
        dup_gid = group_id + DUP_GID_ADDUCT_OFFSET
        features[rep_idx].adduct_type = rep_label
        features[rep_idx].adduct_group_id = group_id
        features[rep_idx].duplicate_group_id = dup_gid
        # rep stays is_duplicate=False
        for idx in comp:
            if idx == rep_idx:
                continue
            label_pair = pair_labels.get((rep_idx, idx))
            partner_label = label_pair[1] if label_pair is not None else ""
            features[idx].adduct_type = partner_label
            features[idx].adduct_group_id = group_id
            features[idx].duplicate_group_id = dup_gid
            features[idx].duplicate_type = "adduct"
            features[idx].is_duplicate = True
        group_id += 1

    return group_id
