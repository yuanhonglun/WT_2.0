# WT 2.0 — ASFAM Data Processing Pipeline · User Guide

This guide covers installation, data preparation, running the pipeline through
both the GUI and the command line, configuration, the processing stages, and the
output formats.

- [1. Overview](#1-overview)
- [2. Installation](#2-installation)
- [3. Preparing input data](#3-preparing-input-data)
- [4. Quick start](#4-quick-start)
- [5. Using the GUI](#5-using-the-gui)
- [6. Command-line interface](#6-command-line-interface)
- [7. Configuration](#7-configuration)
- [8. The processing pipeline](#8-the-processing-pipeline)
- [9. Deduplication and co-elution rules](#9-deduplication-and-co-elution-rules)
- [10. Spectral libraries](#10-spectral-libraries)
- [11. Output files](#11-output-files)
- [12. Comparing against MS-DIAL](#12-comparing-against-ms-dial)
- [13. Repository structure](#13-repository-structure)
- [14. Development and testing](#14-development-and-testing)
- [15. Troubleshooting](#15-troubleshooting)
- [16. Citation, license, and contact](#16-citation-license-and-contact)

---

## 1. Overview

**ASFAM** (All-ion Stepwise Fragmentation Acquisition Mode) is an LC-QTOF
acquisition strategy that walks a **1 Da HR-MRM stepping window** across the full
precursor range (typically m/z 75–1075). Because each window is only 1 Da wide,
the MS2 spectrum recorded at each step is dominated by fragments of a single
precursor, dramatically reducing the spectral chimericity that limits wide-window
DIA. In effect, nearly every precursor in the range is fragmented with
near-MRM selectivity but at high resolution.

This pipeline turns the resulting raw `mzML` files into a clean, annotated
feature table. Its distinguishing capability is that it detects features from
**both** the MS1 and MS2 channels, so it recovers **MS2-only** features —
low-abundance compounds that produce usable fragment spectra but no confidently
detectable precursor signal — in addition to the conventional MS1-driven
features.

This project is the ASFAM mode of the **METRA** platform (*Metabolomics Engine
for Trace Reconstruction and Analysis*). It is distributed here as a
self-contained repository containing only the ASFAM pipeline and the shared core
and GUI libraries it depends on.

---

## 2. Installation

### Prerequisites

- **Python ≥ 3.9** (64-bit recommended).
- A working C/Fortran runtime for NumPy/SciPy (bundled with the standard wheels).
- On Linux, Qt may require system libraries (e.g. `libgl1`, `libxcb`) for the
  GUI. The CLI has no display requirement.

### Install from source

Clone the repository and install the three packages in editable mode:

```bash
git clone https://github.com/yuanhonglun/WT_2.0.git
cd WT_2.0
python -m pip install -e packages/metabo_core \
                      -e packages/metabo_gui \
                      -e apps/asfam_processor
```

This installs, with their dependencies:

| Package | Role |
|---|---|
| `metabo-core` | Shared algorithms: peak detection, similarity, deconvolution, library matching, alignment |
| `metabo-gui` | Shared Qt building blocks: theme, canvas, dialogs, plot widgets |
| `asfam-processor` | The ASFAM pipeline, GUI, and the `asfam` CLI entry point |

Key third-party dependencies (installed automatically): `numpy`, `scipy`,
`pandas`, `pymzml`, `matchms`, `msbuddy`, `matplotlib`, `PyQt5`, `lxml`.

### Verify

```bash
asfam --version
pytest            # optional: runs the test suite
```

---

## 3. Preparing input data

### File format

The pipeline reads **`mzML`** files. Convert vendor raw data with
[ProteoWizard `msconvert`](https://proteowizard.sourceforge.io/) (or your
vendor's exporter) before processing. Centroided data is expected.

### Segments and replicates

ASFAM acquisitions are commonly split into **segments** (contiguous blocks of
1 Da windows covering part of the precursor range) and acquired for multiple
**replicates** of a sample. A single run therefore yields several `mzML` files:

```
sample_A_segment1.mzML
sample_A_segment2.mzML
...
```

- Pass **all segment files for one sample** together — the pipeline merges
  segments internally (Stage 3).
- To align across replicates, process each replicate and use the cross-replicate
  alignment stage (Stage 7).

### Ionization mode

Choose `positive` or `negative` to match how the data was acquired. This governs
the adduct rules used during deduplication and inference.

---

## 4. Quick start

**GUI** — launch, then load your files through the setup panel:

```bash
python -m asfam.gui.app
```

**CLI** — process two segment files and annotate with a library:

```bash
asfam sample_seg1.mzML sample_seg2.mzML \
      -o results/ \
      --library path/to/library.msp \
      --mode positive \
      --workers 4
```

When it finishes you will find `features.csv`, `ms2_spectra.mgf`,
`ms2_spectra.msp`, and a processing report inside `results/`.

---

## 5. Using the GUI

Start the GUI with `python -m asfam.gui.app`. The window is organized into the
following functional areas:

- **Setup panel** — select input `mzML` files, the output directory, an optional
  spectral library, ionization mode, and worker count. Worker count can be set to
  match your CPU.
- **Progress panel** — shows the current processing stage and a running log while
  the pipeline executes. Processing runs on a background worker, so the interface
  stays responsive and the run can be cancelled.
- **Feature table** — the resulting features, one row each, with precursor m/z,
  retention time, intensity, annotation, and relationship flags. Sortable and
  filterable.
- **EIC plot** — the extracted-ion chromatogram for the selected feature, with
  pan/zoom.
- **MS2 plot** — the reconstructed MS2 spectrum for the selected feature. When an
  annotation exists, the reference spectrum is overlaid (mirror plot) for visual
  confirmation. Spectra can be exported.
- **Scatter plot** — an m/z-vs-RT overview of all features, useful for spotting
  duplicate/isotope/adduct groups, which are drawn with connecting lines.

### Typical workflow

1. Add the segment `mzML` files for your sample in the setup panel.
2. Set the output directory and (optionally) a spectral library.
3. Choose the ionization mode and worker count.
4. Start processing and watch the progress panel.
5. Review features in the table; click a row to inspect its EIC and MS2 spectrum.
6. Use the scatter view to sanity-check deduplication groupings.
7. Results are written to the output directory automatically; you can also save a
   project file to revisit the session later.

---

## 6. Command-line interface

```
asfam [MZML_FILES...] -o OUTPUT [options]
```

| Argument | Description |
|---|---|
| `mzml_files` | One or more input `mzML` files (segments × replicate). **Required.** |
| `-o`, `--output` | Output directory. **Required.** |
| `--library FILE` | Spectral library (`MSP`/`MGF`) used for annotation and for inferring the precursor m/z of low-response features. Optional. |
| `--config FILE` | Processing config `JSON` file overriding the defaults (see §7). Optional. |
| `--mode {positive,negative}` | Ionization mode. Default `positive`. |
| `--workers N` | Number of parallel workers. Default `4`. |
| `-v`, `--verbose` | Verbose (DEBUG-level) logging. |
| `--version` | Print the version and exit. |
| `-h`, `--help` | Show help and exit. |

**Examples**

```bash
# Minimal run
asfam sample.mzML -o out/

# Multiple segments + library annotation, negative mode
asfam seg1.mzML seg2.mzML seg3.mzML -o out/ --library lib.msp --mode negative

# Custom configuration and more workers, verbose
asfam *.mzML -o out/ --config my_config.json --workers 8 -v
```

Progress is printed per stage; on completion the feature count and output path
are reported.

---

## 7. Configuration

Without `--config`, sensible defaults are used for every stage. To override them,
pass a `JSON` file via `--config` (CLI) or through the setup panel (GUI). The
config controls the algorithm-level parameters grouped by concern:

- **Peak detection** — smoothing, baseline, minimum peak width/height, signal
  thresholds.
- **Similarity / spectral matching** — the composite-similarity weighting used
  for deduplication, annotation, and inference.
- **Deduplication** — RT and correlation gates, mass tolerances for isotope and
  adduct grouping, ISF criteria.
- **Alignment** — RT tolerance and Gaussian-similarity thresholds for
  cross-replicate matching, and gap-filling behavior.
- **Annotation** — library score thresholds and the number of candidates kept.

The two run-level options exposed directly on the CLI — **ionization mode**
(`--mode`) and **worker count** (`--workers`) — override whatever is in the
config file. The authoritative list of parameters and their defaults lives in
[`apps/asfam_processor/asfam/config.py`](../apps/asfam_processor/asfam/config.py);
inspect it to see the exact field names before writing a custom config. A
practical approach is to start from the defaults and override only the few
parameters you need.

---

## 8. The processing pipeline

The end-to-end pipeline is orchestrated by
`asfam/pipeline/orchestrator.py` and runs the following stages:

| Stage | Name | What it does |
|---|---|---|
| 0 | **Load** | Parse each `mzML`, separate MS1 and the 1 Da MS2 stepping windows, index scans by RT. |
| 1 | **MS2 detection** | Build EICs for MS2 product ions, detect chromatographic peaks, and cluster co-eluting fragments (AMDIS-style) into candidate MS2 spectra. |
| 1b | **MS1-driven complementary detection** | Independently mine the MS1 channel for features, recovering precursors the MS2-driven pass may have missed. |
| 2 | **MS1 assignment** | Two-pass, exclusive assignment of MS1 peaks to features with peak-shape scoring, preventing one MS1 peak from being claimed by several features. |
| 2.5 | **MS2-only inference** | For features with fragment spectra but no MS1 signal, infer the precursor m/z via spectral-library matching and neutral-loss consensus. |
| 3 | **Segment merge** | Merge features across the acquisition segments into one feature set per sample. |
| 4 | **Isotope dedup** | Collapse isotopologues with a graph-based method: an isotope edge requires strict apex-RT proximity, correlated MS1 EICs, and a Δm/z on a known isotope step (¹³C·n), reinforced by an ASFAM-specific MS2 isotope-step echo — the lighter isotopologue's top fragments reappearing mass-shifted in the heavier one's MS2. |
| 5 | **Adduct dedup** | Consolidate adducts of the same neutral mass, validated by EIC Pearson correlation. |
| 5b | **Duplicate detection** | Merge near-identical co-eluting features by cosine similarity. |
| 6 | **ISF detection** | Flag in-source fragments using the precursor–fragment m/z relationship together with EIC correlation. |
| 6.5 | **Library annotation** | Score features against the reference library with composite similarity and attach the best candidates. |
| 7 | **Cross-replicate alignment** | Match features across replicates by Gaussian similarity in the m/z–RT plane, with gap filling. |
| 8 | **Export** | Write the feature table, MS2 spectral libraries, and the processing report. |

Intensity-weighted centroids are used throughout for feature m/z and intensity
(never a plain arithmetic mean), so that strong ions dominate the reported
values as expected.

---

## 9. Deduplication and co-elution rules

Isotope removal, adduct consolidation, and ISF assignment share one hard
prerequisite: the two features being related must **co-elute** — their apex
retention times must be very close **and** their MS1 EICs must be highly
correlated. Only on top of that co-elution gate does each relation add its own
test:

- **Isotope dedup** — features clearly separated in RT are never merged as
  isotopes; connected components spanning a wide RT range are split.
- **Adduct dedup** — additionally requires the expected neutral-mass difference
  for the adduct pair.
- **ISF** — requires a genuine precursor→fragment relationship between two
  co-eluting features.

These gates are deliberate: they prevent the over-merging that inflates or
distorts feature tables when co-elution is not actually satisfied.

---

## 10. Spectral libraries

A reference library (passed with `--library`) is used in two places:

1. **Annotation** (Stage 6.5) — features are scored against the library with a
   composite similarity that combines fragment matching with precursor/RT
   evidence, and the best-scoring candidates are attached to each feature.
2. **m/z inference** (Stage 2.5) — for MS2-only features, library matching helps
   infer the precursor m/z that no MS1 peak could provide.

**Supported formats:** `MSP` and `MGF`. Each library entry should provide the
precursor m/z, the fragment peak list, and — where available — the compound
name and molecular formula, so those fields can propagate into the annotated
output. Larger, higher-quality libraries yield more and more-confident
annotations.

---

## 11. Output files

All outputs are written to the directory given by `-o/--output`.

### `features.csv`

The primary feature table, one row per feature. The file begins with a short
metadata header:

```
# mode=asfam
# version=<software version>
# chromatographic_mode=lc
```

followed by the table. Columns include the feature identifier, precursor m/z,
retention time, intensity/area, the number of fragment ions, annotation fields
(name, formula, score) where an annotation was made, and relationship flags
produced by the deduplication stages (isotope/adduct/duplicate/ISF).

### `ms2_spectra.mgf`

The reconstructed MS2 spectra in MGF form. Each block looks like:

```
BEGIN IONS
FEATURE_ID=<id>
PEPMASS=<precursor m/z>
RTINSECONDS=<retention time in seconds>
CHARGE=1+
NAME=<annotation name, if any>
FORMULA=<formula, if any>
<mz>	<intensity>
...
END IONS
```

### `ms2_spectra.msp`

The same spectra in NIST-style MSP form:

```
NAME: <name or feature id>
PRECURSORMZ: <precursor m/z>
RETENTIONTIME: <retention time in minutes>
FORMULA: <formula, if any>
Num Peaks: <n>
<mz>	<intensity>
...
```

Both libraries are suitable for downstream identification tools or for building a
project-specific spectral library.

### Processing report

A human-readable summary of what each stage did (feature counts, merges,
annotations) is written alongside the results (in a `_debug/` subdirectory),
which is useful for auditing a run.

---

## 12. Comparing against MS-DIAL

`scripts/compare_asfam_msdial.py` is a utility for benchmarking an ASFAM run
against an [MS-DIAL](http://prime.psc.riken.jp/compms/msdial/main.html) export —
matching features by m/z and RT and reporting recall and annotation overlap. It
underpins the ASFAM-vs-MS-DIAL comparison reported for WT 2.0.

```bash
python scripts/compare_asfam_msdial.py --help
```

Note that it expects local ASFAM and MS-DIAL result directories, which are not
shipped with this repository; point it at your own outputs.

---

## 13. Repository structure

```
WT_2.0/
├── packages/
│   ├── metabo_core/        # shared, reusable algorithms and models
│   │   └── metabo_core/
│   │       ├── algorithms/     peak detection, similarity, dedup, smoothing, ...
│   │       ├── annotation/     library matching + reranking
│   │       ├── alignment/      cross-sample alignment
│   │       ├── gcms/           deconvolution core (reused by ASFAM MS2 clustering)
│   │       ├── config/         algorithm-level config objects
│   │       ├── io/             mzML and spectral-library I/O
│   │       ├── models/         data models (scans, features, chromatography)
│   │       └── constants/      mass constants
│   └── metabo_gui/         # shared Qt widgets (theme, canvas, dialogs, plots)
├── apps/
│   └── asfam_processor/
│       └── asfam/
│           ├── pipeline/       the staged pipeline (stage0 … stage8)
│           ├── core/           ASFAM EIC, clustering, peak detection glue
│           ├── io/             mzML reader, exporters, project files
│           ├── gui/            Qt application (panels, plots, worker)
│           ├── config.py       app-facing parameter surface
│           └── cli.py          the `asfam` command
├── scripts/
│   └── compare_asfam_msdial.py
├── pyproject.toml          # pytest configuration
├── LICENSE
└── README.md
```

`metabo_core` never imports app or GUI code — a boundary enforced by
`packages/metabo_core/tests/test_boundaries.py`. The ASFAM MS2-clustering step
reuses the deconvolution routines in `metabo_core/gcms/`, which is why that
shared module is included here even though this repository ships no GC-MS app.

---

## 14. Development and testing

Run the full test suite from the repository root:

```bash
pytest
```

Run a single file or test:

```bash
pytest packages/metabo_core/tests/test_similarity.py -q
pytest apps/asfam_processor/tests/test_app_config.py -q
```

Pytest discovery and the import paths are configured in the root
`pyproject.toml`. The GUI tests run headless; on a machine without a display set
`QT_QPA_PLATFORM=offscreen`.

The single authoritative version string lives in
`packages/metabo_core/metabo_core/__version__.py`; every package, the GUI About
dialog, and `asfam --version` derive their version from it.

---

## 15. Troubleshooting

- **`asfam: command not found`** — the console script is registered by installing
  `apps/asfam_processor`. Re-run the editable install, or invoke the module
  directly: `python -m asfam.cli`.
- **Qt / GUI fails to start on a headless machine** — install the required system
  libraries, or use the CLI. For headless *tests*, set
  `QT_QPA_PLATFORM=offscreen`.
- **No MS2-only features detected** — confirm your data is genuine ASFAM (1 Da
  stepping windows) and that both MS1 and MS2 scans survived the `mzML`
  conversion (do not filter out MS2).
- **Few or no annotations** — check that `--library` points to a valid `MSP`/`MGF`
  file for the correct ionization mode, and that library entries carry precursor
  m/z and peak lists.
- **Over- or under-merged features** — tune the RT and correlation gates in the
  configuration (see §7 and §9) rather than disabling deduplication entirely.

---

## 16. Citation, license, and contact

**Citation.** If you use this software, please cite the WT 2.0 publication.
*(Full citation details will be added upon publication.)*

**License.** BSD 3-Clause with a Non-Commercial Clause: free for academic and
non-commercial use; commercial use requires written permission from the author.
See [`LICENSE`](../LICENSE).

**Contact.** Honglun Yuan — Hainan University — yuanhonglun@hotmail.com
