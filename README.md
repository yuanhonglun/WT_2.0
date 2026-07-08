# WT 2.0 — ASFAM Data Processing Pipeline

A dedicated pipeline for processing **ASFAM** (All-ion Stepwise Fragmentation
Acquisition Mode) LC-QTOF mass spectrometry data — from raw `mzML` files to
annotated feature tables and spectral libraries.

ASFAM acquires high-resolution MS2 spectra through **1 Da HR-MRM stepping
windows** across the full precursor range, so that virtually every precursor
ion is fragmented with substantially reduced chimericity. This repository is the
data-processing engine for the WT 2.0 widely-targeted metabolomics workflow. It
is the ASFAM mode of the **METRA** platform (*Metabolomics Engine for Trace
Reconstruction and Analysis*), packaged here as a standalone, self-contained
project.

> 📖 **Full documentation:** see [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) for
> installation details, a GUI walkthrough, the complete CLI reference, the
> processing-stage breakdown, and output formats.

## Key features

- **MS1/MS2 dual-driven detection** — MS2-driven extraction (EIC peak detection
  + RT clustering) combined with complementary MS1-driven detection for
  comprehensive metabolome coverage, including **MS2-only** features that carry
  no detectable MS1 signal.
- **MS2 ion recall & apex re-centroiding** — a second detection pass recovers
  weak co-eluting fragments, and product ions are re-centroided at the
  chromatographic apex for higher m/z accuracy.
- **Four-layer deduplication** — isotope removal, adduct consolidation, spectral
  duplicate detection, and in-source fragmentation (ISF) detection, all gated by
  strict co-elution (tight RT + high EIC correlation).
- **Spectral annotation & m/z inference** — composite-similarity matching
  against MSP/MGF libraries, plus library and neutral-loss inference of
  precursor m/z for low-response features.
- **Cross-replicate alignment** — Gaussian-similarity matching with gap filling.
- **GUI and CLI** — an interactive Qt interface (feature table, EIC/MS2 plots,
  duplicate visualization) and a scriptable command line.
- **Open formats** — reads `mzML`; exports `CSV`, `MGF`, and `MSP`.

## Installation

Requires Python ≥ 3.9. Install the three packages in editable mode:

```bash
python -m pip install -e packages/metabo_core \
                      -e packages/metabo_gui \
                      -e apps/asfam_processor
```

## Quick start

**GUI:**

```bash
python -m asfam.gui.app
```

**CLI:**

```bash
asfam sample_seg1.mzML sample_seg2.mzML -o results/ --library lib/pos.msp
```

Run `asfam --help` for all options. Outputs (`features.csv`, `ms2_spectra.mgf`,
`ms2_spectra.msp`, and a processing report) are written to the output directory.

## Repository layout

```
packages/
  metabo_core/   shared algorithms (peak picking, similarity, deconvolution,
                 library matching, alignment)
  metabo_gui/    shared Qt building blocks (theme, canvas, dialogs, plots)
apps/
  asfam_processor/   the ASFAM pipeline, GUI, and CLI
scripts/
  compare_asfam_msdial.py   ASFAM vs. MS-DIAL comparison utility
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
