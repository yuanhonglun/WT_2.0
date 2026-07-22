# WT 2.0 — ASFAM Data Processing Pipeline

A dedicated pipeline for processing **ASFAM** (All-ion Stepwise Fragmentation
Acquisition Mode) LC-QTOF mass spectrometry data — from raw `mzML` files to
aligned, gap-filled, annotated feature tables and spectral libraries.

ASFAM acquires high-resolution MS2 spectra through **1 Da HR-MRM stepping
windows** across the full precursor range, so that virtually every precursor
ion is fragmented with substantially reduced chimericity. This repository is the
data-processing engine for the WT 2.0 widely-targeted metabolomics workflow. It
is the ASFAM mode of the **METRA** platform (*Metabolomics Engine for Trace
Reconstruction and Analysis*), packaged here as a standalone, self-contained
project.

> 📖 **Full documentation:** see [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) for
> installation, a GUI walkthrough, the complete CLI reference, the
> processing-stage breakdown, the output-column reference, and the rules a
> downstream script must follow when reading `features.csv`.

## Key features

- **MS1/MS2 dual-driven detection** — MS2-driven extraction (product-ion EIC
  peak detection + co-elution clustering) combined with complementary
  MS1-driven detection, so the run covers both conventional MS1 features and
  **MS2-only** features that carry no confidently detectable precursor signal.
- **MS-DIAL-style peak spotting** — the default chromatographic peak detector
  reimplements MS-DIAL's peak-spotting algorithm; MS2 deconvolution can run
  either the MSDec model or AMDIS-style component perception.
- **Four-layer duplicate marking** — isotopes, adducts, spectral duplicates and
  in-source fragments (ISF), each gated by strict co-elution (tight apex RT
  *and* high EIC correlation). Nothing is ever deleted: every feature is
  exported with an `is_duplicate` / `duplicate_type` flag and downstream code
  decides what to keep.
- **Spectral annotation** — composite-similarity (WDP/SDP/RDP) matching against
  MSP/MGF libraries with a two-tier confidence model, optional reranking, and
  optional library/neutral-loss inference of the precursor m/z for MS2-only
  features.
- **Cross-sample alignment with gap filling** — a union master list, greedy
  bucketed claiming, per-spot representative selection, spot reconciliation
  under an MS2 identity test, then re-integration of the quantitation ion in
  every sample where no peak was picked, so the exported matrix has a number in
  every cell and a `gap_fill_status` saying how it got there.
- **Memory bounded by one sample, not by the study** — stages 0–6.5 run one
  sample at a time and spill to disk before the next is loaded; alignment reads
  the spill back with raw data and library already freed. Peak RSS is
  essentially flat as samples are added.
- **Checkpoint and resume** — each spilled sample is a checkpoint keyed by a
  parameter fingerprint; a re-run with the same settings skips it, and
  annotation/alignment/export can be re-run alone.
- **GUI and CLI** — a bilingual (English/中文) Qt interface with feature table,
  EIC and mirrored MS2 plots, an m/z–RT overview, per-feature notes, and
  project save/load; plus a scriptable command line.
- **Open formats** — reads `mzML`; exports `CSV`, `MGF`, and `MSP`.

## Installation

Requires Python ≥ 3.9. Install the three packages in editable mode:

```bash
python -m pip install -e packages/metabo_core \
                      -e packages/metabo_gui \
                      -e apps/asfam_processor
```

A prebuilt Windows executable is also published on the
[Releases](https://github.com/yuanhonglun/WT_2.0/releases) page and needs no
Python installation; check its tag against the version of the source you are
reading, as the build is cut at release time rather than on every commit.

## Quick start

**GUI:**

```bash
python -m asfam.gui.app
```

**CLI:**

```bash
asfam sample_seg1.mzML sample_seg2.mzML -o results/ --library lib/pos.msp
```

Run `asfam --help` for all options. Outputs (`features.csv`,
`ms2_spectra.mgf`, `ms2_spectra.msp`, the chromatogram store `alignment.eic`,
and a report under `_debug/`) are written to the output directory.

> ⚠️ **Reading `features.csv`:** duplicates are marked, not removed. Any
> downstream script must filter on `is_duplicate == False` — `len(df)` is *not*
> the feature count. See [§11 of the user guide](docs/USER_GUIDE.md#11-output-files).

## Repository layout

```
packages/
  metabo_core/   shared algorithms (peak spotting, deconvolution, similarity,
                 library matching, alignment, gap filling)
  metabo_gui/    shared Qt building blocks (theme, canvas, plots, notes)
apps/
  asfam_processor/   the ASFAM pipeline, GUI, and CLI
docs/
  USER_GUIDE.md      full documentation
```

## Testing

```bash
pytest
```

## Citation

If you use this software, please cite the WT 2.0 publication. *(Citation details
will be added upon publication.)*

## License

BSD 3-Clause with a Non-Commercial Clause. Free for academic and non-commercial
use; commercial use requires written permission from the author. See
[`LICENSE`](LICENSE).

## Author

Honglun Yuan — Hainan University — yuanhonglun@hotmail.com
