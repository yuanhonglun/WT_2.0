# WT 2.0 — ASFAM Data Processing Pipeline · User Guide

This guide covers installation, data preparation, running the pipeline through
both the GUI and the command line, configuration, the processing stages, the
output files, and the rules a downstream script must follow when reading them.

- [1. Overview](#1-overview)
- [2. Installation](#2-installation)
- [3. Preparing input data](#3-preparing-input-data)
- [4. Quick start](#4-quick-start)
- [5. Using the GUI](#5-using-the-gui)
- [6. Command-line interface](#6-command-line-interface)
- [7. Configuration](#7-configuration)
- [8. The processing pipeline](#8-the-processing-pipeline)
- [9. Duplicate marking and co-elution rules](#9-duplicate-marking-and-co-elution-rules)
- [10. Alignment, quantitation ions, and gap filling](#10-alignment-quantitation-ions-and-gap-filling)
- [11. Output files](#11-output-files)
- [12. Intermediate results, checkpoints, and resuming](#12-intermediate-results-checkpoints-and-resuming)
- [13. Spectral libraries](#13-spectral-libraries)
- [14. Repository structure](#14-repository-structure)
- [15. Development and testing](#15-development-and-testing)
- [16. Troubleshooting](#16-troubleshooting)
- [17. Citation, license, and contact](#17-citation-license-and-contact)

---

## 1. Overview

**ASFAM** (All-ion Stepwise Fragmentation Acquisition Mode) is an LC-QTOF
acquisition strategy that walks a **1 Da HR-MRM stepping window** across the full
precursor range (typically m/z 75–1075). Because each window is only 1 Da wide,
the MS2 spectrum recorded at each step is dominated by fragments of a single
precursor, dramatically reducing the spectral chimericity that limits wide-window
DIA. In effect, nearly every precursor in the range is fragmented with
near-MRM selectivity but at high resolution.

This pipeline turns the resulting raw `mzML` files into a clean, aligned,
gap-filled and annotated feature table. Its distinguishing capability is that it
detects features from **both** the MS1 and MS2 channels, so it recovers
**MS2-only** features — low-abundance compounds that produce usable fragment
spectra but no confidently detectable precursor signal — in addition to the
conventional MS1-driven features.

It is distributed as a self-contained repository containing the ASFAM pipeline
and the shared core and GUI libraries it depends on — nothing else is required.

### Two design decisions worth knowing before you start

1. **Nothing is deleted — everything is marked.** The deduplication stages never
   drop a feature; they set `is_duplicate` and `duplicate_type` and let you
   decide. The exported table therefore has many more rows than there are
   readable compounds. See [§9](#9-duplicate-marking-and-co-elution-rules) and
   [§11](#11-output-files).
2. **Peak memory is set by one sample, not by the study.** Stages 0–6.5 run one
   sample at a time and spill their result to disk before the next sample is
   loaded; alignment then reads the spill back with the raw scans and the
   spectral library both already released. Adding samples costs disk and time,
   not RAM. See [§12](#12-intermediate-results-checkpoints-and-resuming).

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
| `metabo-core` | Shared algorithms: peak spotting, deconvolution, similarity, library matching, alignment, gap filling |
| `metabo-gui` | Shared Qt building blocks: theme, canvas, plot toolbar, spectrum display, notes/feedback dock |
| `asfam-processor` | The ASFAM pipeline, GUI, and the `asfam` CLI entry point |

Key third-party dependencies (installed automatically): `numpy`, `scipy`,
`pandas`, `pymzml`, `matchms`, `msbuddy`, `matplotlib`, `PyQt5`, `lxml`.

### Prebuilt Windows executable

The [Releases](https://github.com/yuanhonglun/WT_2.0/releases) page carries a
PyInstaller build of the GUI (`ASFAMProcessor.exe`) that needs no Python
installation: download the zip, unzip it, and run the executable inside. Builds
are cut at release time, so check the release tag against the source version if
you need an exact match. `ASFAMProcessor.spec` in `apps/asfam_processor/` is the
build recipe if you want to produce your own.

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
vendor's exporter) before processing. Centroided data is expected, and **both
MS1 and MS2 scans must survive the conversion** — filtering MS2 out removes the
channel the pipeline is built around.

### Segments, samples, and file naming

An ASFAM acquisition is split into **segments** — contiguous blocks of 1 Da
stepping windows, each covering part of the precursor range — so one sample
yields several `mzML` files. The pipeline merges the segments internally
(Stage 3), and it needs to know which files belong together *before* it opens
any of them, so the grouping is read from the file names.

Recognized naming patterns, most specific first:

```
<sample>_<mzLow>-<mzHigh>_<x>_<step>_<rep>     e.g.  MIX_100-129_1_30_1
<sample>_<mzLow>-<mzHigh>_<rep>                e.g.  CK1_075-110_1
<sample>_<mzLow>-<mzHigh>                      e.g.  Sample_100-129
```

If none match, any `N-M` range found anywhere in the name is used as the segment
range and the first number after it as the replicate index.

> ⚠️ **By default files are grouped by the trailing replicate index**, not by the
> sample name: every file whose name ends in `_1` becomes sample `1`, every `_2`
> becomes sample `2`, and so on. That is the right behavior for the common case
> (one sample acquired in *n* replicates × *m* segments), but it is wrong if you
> feed several differently-named samples that share replicate numbering. In that
> case use **Edit Samples** in the GUI to define the groups explicitly.

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

When it finishes, `results/` contains `features.csv`, `ms2_spectra.mgf`,
`ms2_spectra.msp`, the chromatogram store `alignment.eic` with its `spotmap.json`
index, the intermediate `_work/` directory, and a report under `_debug/`.

---

## 5. Using the GUI

Start the GUI with `python -m asfam.gui.app` (or run `ASFAMProcessor.exe` from a
release build). The window title shows the running version, e.g.
`ASFAM Processor v1.2.260721`. The interface is available in **English and
Chinese**; the choice is persisted between sessions.

### Toolbar

| Action | What it does |
|---|---|
| **Run** | Start the pipeline with the current setup and configuration. |
| **Stop** | Request cancellation; the run stops at the next stage boundary. |
| **Re-annotate** | Re-run annotation → alignment → export only, reusing the intermediate results on disk. Use it to try a different library or scoring threshold without repeating detection. It overwrites the existing annotations. |
| **Save Project** / **Open Project** | Save or restore a session as a `.asfam` project file (configuration, features, file paths, statistics, and a pointer to the intermediate directory). |
| **Clear Intermediates** | Delete the `_work/` directory. This is the *only* thing that removes it — see [§12](#12-intermediate-results-checkpoints-and-resuming). |
| **Export** | Write the current results out again (same files as Stage 8). |
| **Notes Panel** | Show/hide the per-feature notes dock — see below. |
| **About** | Version and license information. |

### Panels

- **Setup panel** — input `mzML` files (**Add Files** / **Remove** / **Edit
  Samples**), the spectral library, the output directory, and the parameters,
  grouped into **General**, **Detection**, **Dedup** and **Alignment** tabs
  (ion mode, worker count, peak height and S/N minima, EIC m/z tolerance, RT
  clustering tolerance, minimum fragments, library match threshold, MS-Buddy
  formula prediction, …). Configurations can be saved to and loaded from JSON.
- **Progress panel** — the current stage and a running log. Processing runs on a
  background worker, so the interface stays responsive and the run can be
  cancelled.
- **Feature table** — one row per feature, with ID, review status, m/z, RT,
  signal type, MS1 m/z source, detection mode, height, fragment count, CV, S/N,
  formula, adduct, name, and the composite/WDP/RDP annotation scores. Searchable
  and sortable. A view selector switches between the aligned result across all
  samples and a single sample's own features.
- **EIC plot** — the extracted-ion chromatogram of the selected feature, raw and
  smoothed, with pan/zoom. It is served from `alignment.eic`, so plotting never
  reopens the raw data.
- **MS2 plot** — the reconstructed MS2 spectrum. When an annotation exists, the
  reference spectrum is overlaid as a mirror plot for visual confirmation.
  Spectra can be exported.
- **Feature overview** — an m/z-vs-RT scatter of all features, filterable to all
  / MS1-only / MS2-only, useful for spotting duplicate, isotope and adduct
  groups.
- **Notes panel** — attach a review status, issue tags and free-text notes to
  individual features. Notes are stored alongside the project and are written
  into the export directory, so a manual review survives re-processing and can
  be handed to someone else.

### Typical workflow

1. Add the `mzML` files in the setup panel; check the grouping with **Edit
   Samples** if your file names do not follow the replicate convention.
2. Set the output directory and (optionally) a spectral library.
3. Choose the ionization mode, worker count, and any parameters you need.
4. **Run**, and watch the progress panel.
5. Review features in the table; click a row to inspect its EIC and MS2 mirror
   plot; use the overview scatter to sanity-check the duplicate groupings.
6. Flag anything doubtful in the notes panel.
7. Results are written to the output directory automatically; save a project
   file to revisit the session later.

---

## 6. Command-line interface

```
asfam MZML_FILES... -o OUTPUT [options]
```

| Argument | Description |
|---|---|
| `mzml_files` | One or more input `mzML` files (segments × replicate). **Required.** |
| `-o`, `--output` | Output directory. **Required.** |
| `--library FILE` | Spectral library (`MSP`/`MGF`) used for annotation, and for inferring the precursor m/z of low-response features. Optional. |
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
are reported. `--mode` and `--workers` override whatever the config file says.
The CLI always groups files by the file-name convention of
[§3](#3-preparing-input-data); explicit sample groups are a GUI feature.

---

## 7. Configuration

Without `--config`, sensible defaults are used for every stage. To override them,
pass a `JSON` file via `--config` (CLI) or edit the parameters in the setup panel
and save them (GUI). Roughly, the config surface covers:

- **General** — ionization mode, worker count.
- **MS2 detection (Stage 1)** — product-ion EIC m/z tolerance, smoothing, the
  peak detector (`msdial`, the default, or `builtin`), RT clustering tolerance,
  minimum fragments per feature, shape-correlation gate, and the weak-ion recall
  pass.
- **MS1 detection and deconvolution** — MS-DIAL peak-spotting parameters,
  MSDec/AMDIS deconvolution settings, MS2 intensity floors.
- **Inference (Stage 2.5)** — minimum fragments for inference, and the optional
  library-matching route for MS2-only precursor m/z.
- **Annotation (Stage 6.5)** — similarity threshold, minimum matched peaks,
  minimum matched percentage of reference peaks, minimum weighted dot product,
  whether RT participates in scoring, and the optional reranker.
- **Deduplication** — separate RT/correlation/mass gates for isotopes, adducts,
  spectral duplicates and ISF (see §9).
- **Alignment and gap filling** — RT and m/z tolerances, the m/z and RT weights
  of the Gaussian similarity, the reference sample, whether gap filling runs,
  and the RT expansion factor used when re-integrating.
- **Export** — MGF/MSP/report toggles.

The authoritative list of parameters and their defaults lives in
[`apps/asfam_processor/asfam/config.py`](../apps/asfam_processor/asfam/config.py);
read it for exact field names before writing a custom config. A practical
approach is to start from the defaults and override only what you need.

---

## 8. The processing pipeline

The pipeline is orchestrated by `asfam/pipeline/orchestrator.py` and runs in two
phases.

### Phase 1 — per sample (stages 0 – 6.5)

Each sample is loaded, processed to completion, spilled to disk and released
before the next one is touched.

| Stage | Name | What it does |
|---|---|---|
| 0 | **Load** | Parse this sample's `mzML` segments, separate MS1 from the 1 Da MS2 stepping windows, and index scans by RT. |
| 1 | **MS2 detection** | Build EICs for MS2 product ions, detect chromatographic peaks, and cluster co-eluting fragments into candidate MS2 spectra (MSDec or AMDIS-style component perception). A recall pass recovers weak co-eluting ions and product ions are re-centroided at the chromatographic apex. |
| 1b | **MS1-driven complementary detection** | Independently mine the MS1 channel (mass-slice ROIs + MS-DIAL peak spotting) for features the MS2-driven pass missed, and deconvolve their MS2. |
| 2 | **MS1 assignment** | Assign MS1 peaks to features exclusively, with peak-shape scoring, so one MS1 peak cannot be claimed by several features. |
| 2.5 | **MS2-only inference** | For features with fragment spectra but no MS1 peak, infer the precursor m/z from neutral-loss consensus and (optionally) library matching. Features with too few fragments are dropped here — the only stage that removes anything. |
| 3 | **Segment merge** | Merge features across the acquisition segments into one feature set for the sample. |
| 4 | **Isotope marking** | Mark isotopologues using a graph method: an isotope edge needs strict apex-RT proximity, correlated MS1 EICs, and a Δm/z on a known isotope step, reinforced by an ASFAM-specific MS2 isotope-step echo — the lighter isotopologue's top fragments reappearing mass-shifted in the heavier one's MS2. |
| 5 | **Adduct marking** | Mark adducts of the same neutral mass, validated by EIC Pearson correlation. |
| 5b | **Duplicate marking** | Mark near-identical co-eluting features by cosine similarity. |
| 6 | **ISF marking** | Mark in-source fragments from the precursor–fragment m/z relationship plus EIC correlation. |
| 6.5 | **Library annotation** | Score features against the reference library with composite similarity and attach the best candidates. |
| — | **spill** | Write `_work/<sample>.mfeat` / `.mspec` / `.json` and release this sample's raw data and features. |

Stages 4, 5, 5b and 6 are called *marking*, not *removal*, on purpose: they set
`is_duplicate` / `duplicate_type` and never delete a row.

The spectral library is loaded on first actual use, shared by every sample's
Stage 6.5, and released as soon as the per-sample loop ends — it never reaches
alignment, export or the GUI.

### Phase 2 — across samples (stages 7 – 8)

Run once, after every sample has been spilled. Raw data and library are already
out of memory; features are read back from `_work/`.

| Stage | Name | What it does |
|---|---|---|
| 7 | **Join** | Build a union master list, generate candidates by bucketing, claim greedily, and pick a representative per spot. MS2 cosine is scored only on candidate edges, with spectra streamed from the spill one at a time. |
| 7 | **Reconcile** | Merge spots that the union list produced twice for one compound, under an MS2 identity test rather than mere m/z–RT proximity, and audit that no natural peak was lost. |
| 7.1 | **Gap fill** | Reopen the raw data one sample at a time and integrate each spot's quantitation ion wherever no peak was picked, so every (feature, sample) cell has a number. Writes every chromatogram it extracts to `alignment.eic`. |
| 7.2 | **Refine** | Mark cross-sample redundancy and annotation-name duplicates, and compute the detection counts. |
| 8 | **Export** | Write the feature table, the MS2 libraries, and the processing report. |

Feature m/z and intensity centroids are intensity-weighted throughout, never a
plain arithmetic mean, so strong ions dominate the reported values.

---

## 9. Duplicate marking and co-elution rules

Isotope, adduct and ISF relations share one hard prerequisite: the two features
must **co-elute** — their apex retention times must be very close **and** their
MS1 EICs must be highly correlated. Only on top of that gate does each relation
add its own test:

- **Isotopes** — a Δm/z on a known isotope step, plus the MS2 isotope-step echo.
  Features clearly separated in RT are never related as isotopes.
- **Adducts** — additionally requires the expected neutral-mass difference for
  the adduct pair.
- **ISF** — requires a genuine precursor→fragment relationship between two
  co-eluting features.

These gates are deliberate: they prevent the over-merging that distorts feature
tables when co-elution is not actually satisfied.

**What marking means for you.** Every marked feature keeps its row, its
chromatogram and its spectrum; it simply carries `is_duplicate = True` and a
non-empty `duplicate_type` — `isotope`, `adduct`, `spectral` or `isf` from the
per-sample stages, `cross_sample_redundant` or `ms1_covered` from Stage 7.2. The
GUI hides them behind a toggle. A script must filter them itself — see the
warning in [§11](#11-output-files).

---

## 10. Alignment, quantitation ions, and gap filling

### Quantitation ions

`height` / `area` are not abstract "peak intensity": they are readings from **one
specific ion on one specific kind of chromatogram**. Gap filling has to return to
the same ion and the same chromatogram type, or the detected and filled values in
one row would not be comparable. Two routes exist and never mix within a spot:

| Feature | Quantitation ion | Chromatogram |
|---|---|---|
| MS2-only (`signal_type == "ms2_only"`) | the representative product ion | product-ion trace within the feature's 1 Da window |
| everything with an MS1 peak | the MS1 quantitation m/z (ROI centroid for MS1-driven features, apex base peak within the isolation window otherwise) | MS1 trace |

The reported `precursor_mz` of an MS2-driven feature is the intensity-weighted
centroid of its 1 Da isolation window. It is a *reporting* value and is
deliberately not the ion anything is integrated on.

### Gap filling and `gap_fill_status`

After the join, each spot is re-integrated in every sample that has no picked
peak. Each cell then carries a status:

| `gap_fill_status` | Meaning |
|---|---|
| `detected` | A peak was actually picked in that sample. |
| `filled` | No peak was picked; the quantitation ion was re-integrated and produced signal. |
| `no_signal` | No peak was picked and re-integration found nothing. |

> ⚠️ **Filled peaks enter the quantitation matrix only.** Only `detected` peaks
> contribute to `mean_height`, `mean_area`, `cv`, `sn_ratio` and `n_detected`, or
> may represent a spot. Filled values appear only in the per-sample
> `height_rep*` / `area_rep*` columns. Testing `value > 0` is *not* equivalent —
> it excludes `no_signal` but lets `filled` peaks into your statistics. Filter on
> `gap_fill_status_rep*` instead.

`n_detected` and `detection_rate` report in how many samples the peak was really
picked. Since gap filling makes every `height_rep*` cell non-empty, they are the
only way to tell.

---

## 11. Output files

All outputs are written to the directory given by `-o/--output`.

| Path | Content |
|---|---|
| `features.csv` | The feature table (below). |
| `ms2_spectra.mgf` | Reconstructed MS2 spectra, MGF. |
| `ms2_spectra.msp` | The same spectra, NIST-style MSP. |
| `alignment.eic` | Per-spot chromatograms: the quantitation-ion trace for every sample, plus the representative's top fragment traces. This is what the GUI plots from. |
| `spotmap.json` | Index from a per-sample candidate to its spot's chromatograms in `alignment.eic`. Written in lockstep with it. |
| `alignment_merge_map.json` | Source → keeper provenance for the spots reconciliation merged. |
| `_debug/processing_report.txt` | Per-stage counts and timings for the run. |
| `_work/` | Per-sample intermediate results and checkpoints — see [§12](#12-intermediate-results-checkpoints-and-resuming). |

### `features.csv`

One row per feature. The file starts with three comment lines:

```
# mode=asfam
# version=<software version>
# chromatographic_mode=lc
```

> ⚠️ **Duplicates are marked, not removed.** `len(df)` is *not* the number of
> features. Every script that reads this file must filter on
> `is_duplicate == False`, and — if it computes statistics per sample — must
> respect `gap_fill_status_rep*` as described in [§10](#10-alignment-quantitation-ions-and-gap-filling).
> Reading it with `pandas`:
> ```python
> df = pd.read_csv("features.csv", comment="#")
> real = df[~df["is_duplicate"]]
> ```

The columns, by group:

| Group | Columns |
|---|---|
| Common schema (first 16) | `mode_local`, `feature_id`, `mz`, `rt`, `height`, `area`, `n_fragments`, `score`, `name`, `formula`, `adduct`, `is_duplicate`, `duplicate_type`, `isotope_index`, `isotope_group_id`, `adduct_group_id` |
| Identity and provenance | `precursor_mz`, `rt_min`, `rt_left`, `rt_right`, `signal_type` (`ms1_detected` / `ms2_only`), `detection_source` (`ms1_driven` / `ms2_driven` / both) |
| Alignment | `alignment_mz`, `representative_rt`, `alignment_window`, `alignment_segment`, `alignment_relation`, `alignment_related_feature_id`, `gaussian_similarity` |
| Quantitation | `mean_height`, `mean_area`, `cv`, `sn_ratio`, `n_detected`, `detection_rate` |
| Annotation | `composite_score`, `total_score`, `wdp_score`, `sdp_score`, `rdp_score`, `matched_pct`, `n_matched`, `annotated` |
| Spectrum | `ms2_spectrum` |
| Per sample | `height_rep<id>`, `area_rep<id>`, `gap_fill_status_rep<id>` for every sample in the run |

`annotated` is the **high-confidence** flag: the match must clear the similarity
threshold *and* have at least `matchms_min_matched_peaks` matched peaks. Weaker
hits still fill in `name` / `score` as suggestions but come out `annotated =
False` — so filter on `annotated`, not on the presence of a name.

### `ms2_spectra.mgf`

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

```
NAME: <name or feature id>
PRECURSORMZ: <precursor m/z>
RETENTIONTIME: <retention time in minutes>
FORMULA: <formula, if any>
Num Peaks: <n>
<mz>	<intensity>
...
```

Features with no fragments are skipped in both spectral exports. Both files are
suitable for downstream identification tools or for building a project-specific
spectral library.

---

## 12. Intermediate results, checkpoints, and resuming

At the end of Stage 6.5 each sample is written to `<output_dir>/_work/`:

| File | Content |
|---|---|
| `<sample>.mfeat` | The sample's features, without the MS2 arrays (only seek pointers into the `.mspec`). |
| `<sample>.mspec` | The sample's MS2 spectra, indexed so a single spectrum can be read at random. |
| `<sample>.json` | The checkpoint manifest — written last, and published atomically. |

This buys three things:

- **Bounded memory.** Only one sample's raw scans and features are ever resident.
- **Resume after a crash.** Each manifest carries a fingerprint of the
  configuration, the library and the sample's input files. On a re-run with the
  same parameters, that sample is skipped and reused; change a parameter that
  affects it and it is recomputed.
- **Cheap re-export and re-annotation.** Changing an export setting, or trying a
  different library through **Re-annotate**, replays only stages 6.5 → 7 → 8.

`_work/` is **never deleted automatically**. Remove it with **Clear
Intermediates** in the GUI when you no longer need it, or delete the directory
by hand. If you open a project whose raw `mzML` files are no longer reachable,
gap filling is skipped rather than faked, and the missing cells stay missing.

---

## 13. Spectral libraries

A reference library (passed with `--library`) is used in two places:

1. **Annotation** (Stage 6.5) — features are scored against the library with a
   composite similarity combining a weighted dot product (WDP), a simple dot
   product (SDP) and a reverse dot product (RDP), with the fraction of matched
   reference peaks and the matched-peak count as additional gates. The best
   candidates are attached to each feature; an optional reranker can reorder
   them.
2. **Precursor m/z inference** (Stage 2.5, opt-in) — for MS2-only features,
   library matching helps infer the precursor m/z that no MS1 peak could
   provide.

**Supported formats:** `MSP` and `MGF`. Each entry should provide the precursor
m/z, the fragment peak list, and — where available — the compound name and
molecular formula, so those fields can propagate into the annotated output.
Larger, higher-quality libraries yield more and more-confident annotations.

The library is loaded once per run, only when a sample actually needs annotating,
and released before alignment. A run in which every sample hits its checkpoint
never opens it at all.

---

## 14. Repository structure

```
WT_2.0/
├── packages/
│   ├── metabo_core/        # shared, reusable algorithms and models
│   │   └── metabo_core/
│   │       ├── algorithms/     peak spotting, MSDec, similarity, dedup, smoothing, ...
│   │       ├── annotation/     library matching, confidence, reranking
│   │       ├── alignment/      joiner, refiner, gap filler, identity
│   │       ├── gcms/           deconvolution core (reused by ASFAM MS2 clustering)
│   │       ├── config/         algorithm-level config objects
│   │       ├── io/             mzML and spectral-library I/O
│   │       ├── models/         data models (scans, features, chromatography)
│   │       └── constants/      mass constants
│   └── metabo_gui/         # shared Qt widgets (theme, canvas, plots, notes dock)
├── apps/
│   └── asfam_processor/
│       ├── asfam/
│       │   ├── pipeline/       the staged pipeline (stage0 … stage8)
│       │   ├── core/           ASFAM EIC, clustering, peak detection glue
│       │   ├── io/             mzML reader, spill, EIC store, project files
│       │   ├── gui/            Qt application (panels, plots, worker, i18n)
│       │   ├── config.py       app-facing parameter surface
│       │   └── cli.py          the `asfam` command
│       ├── tests/
│       └── ASFAMProcessor.spec # PyInstaller build recipe
├── docs/USER_GUIDE.md
├── pyproject.toml          # pytest configuration
├── LICENSE
└── README.md
```

`metabo_core` never imports app or GUI code — a boundary enforced by
`packages/metabo_core/tests/test_boundaries.py`. The ASFAM MS2-clustering step
reuses the deconvolution routines in `metabo_core/gcms/`, which is why that
shared module is included here even though this repository ships no GC-MS app.

---

## 15. Development and testing

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
`pyproject.toml`, which also pins `pytest-qt` to PyQt5. The GUI tests run
headless; on a machine without a display set `QT_QPA_PLATFORM=offscreen`.

The single authoritative version string lives in
`packages/metabo_core/metabo_core/__version__.py`; every package, the GUI About
dialog, `asfam --version` and the `features.csv` header derive from it.

Several invariants described in this guide are guarded by tests rather than by
convention — the mark-never-delete gating order, the "only detected peaks enter
the statistics" rule, the release of the library after annotation, and the
Stage 7 conservation and structural audits, which abort a run rather than export
a table that lost a peak. Keep them in mind before changing those stages.

---

## 16. Troubleshooting

- **`asfam: command not found`** — the console script is registered by installing
  `apps/asfam_processor`. Re-run the editable install, or invoke the module
  directly: `python -m asfam.cli`.
- **Files were grouped into the wrong samples** — the default grouping keys on the
  trailing replicate index in the file name (see §3). Use **Edit Samples** in the
  GUI to define the groups yourself.
- **`Cannot group <file>`** — the file name carries no recognizable `N-M` segment
  range. Rename it to one of the patterns in §3.
- **Qt / GUI fails to start on a headless machine** — install the required system
  libraries, or use the CLI. For headless *tests*, set
  `QT_QPA_PLATFORM=offscreen`.
- **No MS2-only features detected** — confirm the data is genuine ASFAM (1 Da
  stepping windows) and that both MS1 and MS2 scans survived the `mzML`
  conversion.
- **Few or no annotations** — check that `--library` points to a valid `MSP`/`MGF`
  file for the correct ionization mode, that entries carry precursor m/z and peak
  lists, and remember that `annotated` also requires enough matched peaks, not
  just a high score.
- **A re-run skipped everything** — that is the checkpoint system doing its job.
  Change a parameter, use **Clear Intermediates**, or delete `_work/` to force a
  full recompute.
- **Far more rows than expected** — duplicates are marked, not deleted. Filter on
  `is_duplicate == False` (§11).
- **Over- or under-merged features** — tune the RT and correlation gates in the
  configuration (§7, §9) rather than disabling deduplication entirely.

---

## 17. Citation, license, and contact

**Citation.** If you use this software, please cite the WT 2.0 publication.
*(Full citation details will be added upon publication.)*

**License.** BSD 3-Clause with a Non-Commercial Clause: free for academic and
non-commercial use; commercial use requires written permission from the author.
See [`LICENSE`](../LICENSE).

**Contact.** Honglun Yuan — Hainan University — yuanhonglun@hotmail.com
