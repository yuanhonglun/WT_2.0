# ASFAMProcessor User Guide

## Table of Contents

1. [Installation](#1-installation)
2. [Data Preparation](#2-data-preparation)
3. [GUI Guide](#3-gui-guide)
4. [CLI Guide](#4-cli-guide)
5. [Pipeline Stages Explained](#5-pipeline-stages-explained)
6. [Parameter Reference](#6-parameter-reference)
7. [Output Files](#7-output-files)
8. [Tutorials](#8-tutorials)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Installation

### 1.1 Prerequisites

- Python 3.9 or higher
- pip package manager
- Recommended: 8 GB RAM, multi-core CPU

### 1.2 Install from Source

```bash
git clone https://github.com/yuanhonglun/WT_2.0.git
cd WT_2.0
pip install -e .
```

### 1.3 Verify Installation

```bash
# Check CLI
asfam --version

# Launch GUI
python -m asfam.gui.app
```

### 1.4 Dependencies

All dependencies are installed automatically:

- pymzml >= 2.5.0 (mzML parsing)
- numpy >= 1.21 (numerical operations)
- scipy >= 1.7 (peak detection, signal processing)
- pandas >= 1.3 (data manipulation)
- matchms >= 0.18 (spectral matching)
- msbuddy >= 0.2 (formula prediction)
- matplotlib >= 3.5 (plotting)
- PyQt5 >= 5.15 (GUI)
- lxml >= 4.6 (XML acceleration)

---

## 2. Data Preparation

### 2.1 ASFAM Data Acquisition

Each sample is measured using multiple ASFAM methods (typically 10), each covering a 100 Da m/z range with MS1 + HR-MRM scans. For example:
- Method 1: m/z 75-174
- Method 2: m/z 175-274
- ...
- Method 10: m/z 975-1075

Each method produces one mzML file per replicate. For a typical experiment with 3 replicates and 10 methods, you will have 30 mzML files.

### 2.2 Convert to mzML

Use MSConvert (from ProteoWizard) to convert vendor raw files to mzML:

1. Open MSConvert
2. Add raw files
3. Set output format to mzML
4. Check "Use zlib compression" (recommended)
5. Click "Start"

### 2.3 File Naming Convention

The parser extracts segment and replicate information from filenames. Use one of these patterns:

```
SAMPLE_100-129_rep1.mzML         # sample_seglow-seghigh_rep
SAMPLE_100-129_1_30_1.mzML       # sample_seglow-seghigh_X_step_rep
rice_075-110_1.mzML              # sample_seglow-seghigh_rep
```

Key requirements:
- The m/z range (e.g., "100-129") must appear in the filename
- Files from the same sample/replicate are automatically grouped

### 2.4 Spectral Libraries (Optional)

Prepare a reference spectral library in MSP or MGF format for annotation:

**MSP format example:**
```
NAME: Caffeine
FORMULA: C8H10N4O2
PRECURSOR_MZ: 195.0877
PRECURSOR_TYPE: [M+H]+
Num Peaks: 5
42.0338 150
69.0447 300
110.0712 999
138.0662 800
195.0877 500
```

**MGF format example:**
```
BEGIN IONS
PEPMASS=195.0877
NAME=Caffeine
FORMULA=C8H10N4O2
42.0338 150
69.0447 300
110.0712 999
138.0662 800
195.0877 500
END IONS
```

---

## 3. GUI Guide

### 3.1 Launch

```bash
python -m asfam.gui.app
```

### 3.2 Main Window Layout

```
+------------------------------------------------------------------+
| [Run] [Stop] [Re-annotate] [Save] [Open] [Export]  [View: ...]  |
+------------------------------------------------------------------+
|  Setup Panel  |  Scatter Plot (m/z vs RT)                        |
|  (Left, 280px)|  Feature Table (sortable, filterable)            |
|               |--------------------------------------------------|
|  [Files]      |  EIC Plot            |  MS2 Plot                 |
|  [Library]    |  (chromatogram)      |  (fragmentation spectrum)  |
|  [Output Dir] |                      |                            |
|  [Parameters] |                      |                            |
+------------------------------------------------------------------+
| Progress Bar: [========>          ] 45%  Stage 4: Isotope dedup  |
| Log output...                                                     |
+------------------------------------------------------------------+
```

### 3.3 Step-by-Step Processing

#### Step 1: Load Files

1. Click **Add Files** in the Setup Panel
2. Select all mzML files (from all segments and replicates)
3. Files are automatically grouped by sample and replicate
4. Verify grouping in the status label below the file list
5. Use **Edit Samples** to manually adjust grouping if needed

#### Step 2: Configure

1. Select **Output Directory**
2. Optionally load a **Spectral Library** (MSP/MGF)
3. Adjust parameters in the tabs:
   - **General**: ionization mode, number of workers
   - **Detection**: peak detection thresholds
   - **Dedup**: deduplication stringency
   - **Alignment**: cross-replicate matching tolerances
   - **Export**: output format selection

#### Step 3: Run Pipeline

1. Click **Run Pipeline** on the toolbar
2. Monitor progress in the progress bar and log panel
3. Processing stages execute sequentially:
   - Load -> MS2 Detection (with ion recall) -> MS1 Detection -> MS1 Assignment -> m/z Inference -> Segment Merge -> Isotope Dedup -> Adduct Dedup -> Spectral Dedup -> ISF Detection -> Annotation -> Alignment -> Export

#### Step 4: Explore Results

After pipeline completion:

1. **Scatter Plot**: Click on points to select features. Features are plotted by m/z (y-axis) vs RT (x-axis), colored by signal type.

2. **Feature Table**: Browse all detected features. Columns include:
   - Feature ID, m/z, RT, signal type, fragments, height, area, CV, annotation name, formula
   - Click a row to view EIC and MS2 plots for that feature

3. **EIC Plot**: Shows the extracted ion chromatogram for the selected feature, including:
   - MS1 EIC (if available)
   - MS2 product ion EIC
   - Peak boundaries marked

4. **MS2 Plot**: Shows the fragmentation spectrum with:
   - Stick spectrum of product ions
   - Library match overlay (if annotated)
   - Match score and compound name

5. **View Mode**: Switch between "Aligned (all samples)" view and individual replicate views using the dropdown.

#### Step 5: Save and Export

- **Save Project**: Saves to .asfam file for later reopening
- **Export**: Exports results to the output directory (CSV, MGF, MSP, report)

### 3.4 Re-annotation

If you want to change the spectral library or annotation parameters without rerunning the full pipeline:

1. Load a different spectral library
2. Click **Re-annotate**
3. Only stages 6.5 (annotation) -> 7 (alignment) -> 8 (export) are re-run

### 3.5 Configuration Management

- **Save Config**: Save current parameters to JSON file
- **Load Config**: Load previously saved parameters
- Config files are portable and can be shared or used with the CLI

---

## 4. CLI Guide

### 4.1 Basic Usage

```bash
asfam <mzml_files> -o <output_directory> [options]
```

### 4.2 Examples

```bash
# Process all mzML files in current directory
asfam *.mzML -o ./results

# Specify ionization mode and workers
asfam *.mzML -o ./results --mode negative --workers 8

# Use spectral library for annotation
asfam *.mzML -o ./results --library hmdb_massbank.msp

# Use custom config file
asfam *.mzML -o ./results --config optimized_params.json

# Verbose output for debugging
asfam *.mzML -o ./results -v
```

### 4.3 All Options

| Option | Description | Default |
|--------|-------------|---------|
| `mzml_files` | Input mzML files (positional, 1+) | Required |
| `-o, --output` | Output directory | Required |
| `--library` | Spectral library (MSP/MGF) | None |
| `--config` | Config JSON file | Default params |
| `--mode` | positive or negative | positive |
| `--workers` | Parallel workers (1-32) | 4 |
| `-v, --verbose` | Debug logging | Off |
| `--version` | Print version and exit | - |

### 4.4 Config File

Create a JSON config file to customize all parameters:

```json
{
    "ionization_mode": "positive",
    "n_workers": 4,
    "peak_height_threshold": 500.0,
    "peak_sn_threshold": 5.0,
    "peak_width_min": 5,
    "eic_mz_tolerance": 0.02,
    "rt_cluster_tolerance": 0.02,
    "min_fragments_per_feature": 2,
    "ms1_min_height": 100.0,
    "isotope_modified_cos_threshold": 0.85,
    "adduct_eic_pearson_threshold": 0.9,
    "isf_eic_pearson_threshold": 0.9,
    "alignment_rt_tolerance": 0.1,
    "alignment_mz_tolerance": 0.02,
    "export_mgf": true,
    "export_msp": true
}
```

---

## 5. Pipeline Stages Explained

### Stage 0: Data Loading

Parses mzML files using pymzml. Each scan cycle is organized into MS1 scan + HR-MRM product ion scans. MS2 data are centroided using a local maximum method. Files are automatically grouped by sample, segment, and replicate based on filename patterns.

### Stage 1: MS2-Driven Feature Detection

For each HR-MRM channel (1 Da precursor window):

1. **EIC Extraction**: Product ion m/z range is divided into 0.02 Da bins. An extracted ion chromatogram (EIC) is built for each bin across all scan cycles.

2. **Smoothing**: Savitzky-Golay filter (window=7, polyorder=3) is applied to each EIC.

3. **Peak Detection**: scipy.signal.find_peaks identifies chromatographic peaks with constraints:
   - Height >= 200 counts (configurable)
   - Signal-to-noise >= 5
   - Width >= 3 scans
   - Gaussian shape similarity >= 0.6

4. **RT Clustering**: Peaks within the same channel are clustered by retention time (0.02 min tolerance) to assemble MS2 spectra. Each cluster becomes a candidate feature.

5. **Ion Recall (Second Pass)**: After initial detection, a second pass revisits all product ion EICs to recover weak co-eluting ions missed by strict peak detection. For each feature's consensus apex RT, ions with raw intensity >= 50 counts and at least 2 consecutive nonzero cycles are recalled. This yields richer MS2 spectra.

6. **Product Ion m/z Refinement**: Each product ion's m/z is re-centroided using only the raw data at the chromatographic apex (+/-1 cycle), providing higher m/z accuracy than the global EIC-averaged value.

### Stage 1b: MS1-Driven Feature Detection

Captures compounds visible in MS1 but with weak or no fragmentation:

1. MS1 EIC is analyzed for each nominal m/z channel
2. MS1 peaks not overlapping with MS2-detected features are retained
3. Co-eluting MS2 product ions are extracted to assemble their spectra

### Stage 2: MS1 Assignment

Assigns precise precursor m/z from MS1 data using exclusive batch assignment.
Features are grouped by (segment, channel) and MS1 peaks are assigned using a
scoring function that combines RT proximity with peak shape overlap ratio.
Each MS1 peak can only be assigned to one MS2 feature (prevents cross-assignment).

**Pass 1 (Strict)**: MS1 height >= 100, S/N >= 5, RT tolerance 0.05 min
- Extracts isotope pattern (up to 5 isotopes, C13 spacing 1.00335 Da)
- Computes intensity-weighted centroid m/z
- Marks feature as "ms1_detected"

**Pass 2 (Relaxed)**: Height >= 50, S/N >= 2, RT tolerance 0.1 min
- Assigns m/z for weaker signals missed in Pass 1
- Feature remains "ms2_only" but with improved m/z accuracy

### Stage 2.5: m/z Inference for MS2-Only Features

For features without MS1 signal:

**Method 1 - Library Matching**:
- Searches reference library restricted to same integer m/z channel
- Composite similarity score (fragment matching + precursor weighting)
- Threshold: score >= 0.8, >= 3 matched peaks

**Method 2 - Neutral Loss Consensus**:
- For each fragment m/z, calculates candidate precursor = fragment + common neutral loss
- 26 common neutral losses: H2O (18.0106), NH3 (17.0265), CO (27.9949), CO2 (43.9898), etc.
- Clusters candidates within 0.01 Da
- Selects cluster with most supporting fragments
- Confidence: high (>= 3 fragments), medium (2), low (1)

### Stage 3: Segment Merge

Removes boundary duplicates from adjacent m/z segments:
- Compares features across segments within same replicate
- Merges if: m/z diff <= 0.02 Da, RT diff <= 0.05 min, MS2 cosine >= 0.8
- Retains feature with better MS1 data

### Stage 4: Isotope Deduplication

Graph-based algorithm identifying isotope clusters:

**Candidate pairs**: RT diff <= 0.1 min (search window), apex RT diff <= 0.05 min (hard max), peak overlap >= 80%

**Three evidence tiers for edges**:
1. **MS1 isotope pattern**: Feature j's m/z matches feature i's detected isotope peaks
2. **Modified cosine** (classic gaps: C13, N15, S34, O18, Cl37, Br81): Score >= 0.85, >= 3 matched fragments
3. **Dual metric** (relaxed near-integer gaps): Modified cosine >= 0.90 with >= 4 matches AND neutral loss cosine >= 0.85 with >= 3 matches

Connected components identify isotope groups. Components are split by RT to prevent transitive chains from linking unrelated features. Monoisotopic representative (lowest m/z, highest intensity) is retained.

### Stage 5: Adduct Deduplication

Identifies and consolidates different adducts of the same compound:

1. Group features by RT (tolerance 0.05 min)
2. Test pairs against adduct rules:
   - Positive mode: [M+H]+, [M+Na]+, [M+K]+, [M+NH4]+, [M-H2O+H]+, [M+2H]2+, etc.
   - Negative mode: [M-H]-, [M+Cl]-, [M-H2O-H]-, [M+FA-H]-, etc.
3. Check neutral mass agreement (tolerance 0.02 Da)
4. Validate by EIC Pearson correlation (>= 0.9)
5. Keep representative with highest MS1 intensity

### Stage 5b: Spectral Duplicate Detection

Detects and removes near-identical co-eluting features that escaped isotope and adduct deduplication:

- RT difference <= 0.2 min
- m/z difference <= 0.5 Da
- MS2 cosine similarity >= 0.85 with >= 3 matched fragments
- Graph-based resolution with RT-aware component splitting
- Retains feature with highest MS1 intensity per group

### Stage 6: ISF Detection

Detects in-source fragmentation artifacts using dual criteria:

1. **Precursor-fragment check**: Child feature's m/z appears in parent's MS2 product ion list (tolerance 0.02 Da)
2. **EIC correlation**: Parent and child EICs show Pearson correlation >= 0.9 over >= 10 scan cycles

Both criteria must be satisfied. Child features are removed.

### Stage 6.5: Spectral Library Annotation

Matches active features against user-supplied spectral library:

- Restricts search to same integer m/z channel for efficiency
- Composite similarity score combining fragment greedy matching and precursor ion weighting
- Threshold: score >= 0.8, >= 3 matched peaks, >= 25% of reference peaks matched
- Stores top 5 matches per feature

### Stage 7: Cross-Replicate Alignment

Aligns features across biological replicates:

1. Reference selection: replicate with most features
2. Matching: Gaussian similarity scoring
   - score = exp(-0.5 * [(mz_diff/mz_tol)^2 + (rt_diff/rt_tol)^2])
   - m/z tolerance: 0.02 Da, RT tolerance: 0.1 min
3. Gap filling: optionally fills missing values across replicates
4. Quantification: aggregates heights and areas, computes mean and CV

### Stage 8: Export

Outputs:
- **features.csv**: Complete feature table with per-replicate quantification
- **ms2_spectra.mgf**: Fragmentation spectra in MGF format
- **ms2_spectra.msp**: Fragmentation spectra in MSP format
- **processing_report.txt**: Stage-by-stage statistics

---

## 6. Parameter Reference

### General

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| ionization_mode | str | "positive" | Ionization mode |
| n_workers | int | 4 | Parallel processing workers |

### Stage 1: MS2 Detection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| eic_mz_tolerance | float | 0.02 | Product ion EIC bin width (Da) |
| eic_smoothing_method | str | "savgol" | EIC smoothing method |
| eic_smoothing_window | int | 7 | Savitzky-Golay window size |
| eic_smoothing_polyorder | int | 3 | Polynomial order |
| peak_height_threshold | float | 200.0 | Minimum peak height (counts) |
| peak_sn_threshold | float | 5.0 | Minimum signal-to-noise ratio |
| peak_width_min | int | 3 | Minimum peak width (scans) |
| peak_prominence | float | 100.0 | Minimum peak prominence |
| peak_gaussian_threshold | float | 0.6 | Gaussian shape similarity (0=off) |
| rt_cluster_tolerance | float | 0.02 | RT clustering tolerance (min) |
| min_fragments_per_feature | int | 2 | Minimum product ions per feature |

### Stage 1: MS2 Ion Recall (Second Pass)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| recall_enabled | bool | True | Enable ion recall second pass |
| recall_min_intensity | float | 50.0 | Minimum raw intensity at apex |
| recall_min_consecutive | int | 2 | Minimum consecutive nonzero cycles |
| recall_apex_window | int | 2 | +/- cycles around apex to search |

### Stage 2: MS1 Assignment

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| ms1_mz_tolerance | float | 0.01 | m/z matching tolerance (Da) |
| ms1_rt_tolerance | float | 0.05 | RT matching tolerance (min) |
| ms1_min_height | float | 100.0 | Minimum MS1 peak height |
| ms1_isotope_mz_tol | float | 0.01 | Isotope pattern m/z tolerance (Da) |
| ms1_shape_weight | float | 0.3 | Weight for peak shape in scoring |

### Stage 2.5: m/z Inference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| min_fragments_for_inference | int | 3 | Minimum fragments for inference |
| matchms_similarity_threshold | float | 0.8 | Library match score threshold |
| matchms_min_matched_peaks | int | 3 | Minimum matched peaks |
| matchms_min_matched_pct | float | 0.25 | Minimum matched percentage |
| matchms_use_rt | bool | False | Include RT in matching score |

### Stage 3: Segment Merge

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| merge_rt_tolerance | float | 0.05 | RT tolerance for merging (min) |
| merge_mz_tolerance | float | 0.02 | m/z tolerance for merging (Da) |
| merge_ms2_cosine_threshold | float | 0.8 | MS2 cosine similarity threshold |

### Stage 4: Isotope Deduplication

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| isotope_rt_tolerance | float | 0.1 | RT search window (min) |
| isotope_apex_rt_strict | float | 0.05 | Hard max apex RT difference (min) |
| isotope_overlap_ratio | float | 0.80 | Peak overlap ratio |
| isotope_mz_tolerance | float | 0.01 | m/z tolerance for classic gaps (Da) |
| isotope_integer_step_tolerance | float | 0.02 | m/z tolerance for relaxed gaps (Da) |
| isotope_modified_cos_threshold | float | 0.85 | Modified cosine threshold |
| isotope_modified_cos_relaxed | float | 0.90 | Relaxed modified cosine threshold |
| isotope_min_matches | int | 3 | Minimum matched fragments |
| isotope_min_matches_relaxed | int | 4 | Relaxed minimum matches |
| isotope_nl_cos_threshold | float | 0.85 | Neutral loss cosine threshold |
| isotope_min_nl_matches | int | 3 | Minimum NL matches |
| isotope_max_step | int | 4 | Maximum isotope step (Da) |

### Stage 5: Adduct Deduplication

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| adduct_rt_tolerance | float | 0.05 | RT tolerance (min) |
| adduct_mw_tolerance | float | 0.02 | Neutral mass tolerance (Da) |
| adduct_eic_pearson_threshold | float | 0.9 | EIC correlation threshold |

### Stage 5b: Spectral Duplicate Detection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| duplicate_rt_tolerance | float | 0.2 | RT tolerance (min) |
| duplicate_mz_tolerance | float | 0.5 | m/z tolerance (Da) |
| duplicate_cosine_threshold | float | 0.85 | MS2 cosine similarity threshold |
| duplicate_min_matched | int | 3 | Minimum matched fragments |

### Stage 6: ISF Detection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| isf_eic_pearson_threshold | float | 0.9 | EIC Pearson correlation threshold |
| isf_min_correlated_scans | int | 10 | Minimum correlated scans |
| isf_ms2_mz_tolerance | float | 0.02 | Fragment m/z matching tolerance (Da) |

### Stage 7: Alignment

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| alignment_rt_tolerance | float | 0.1 | RT tolerance (min) |
| alignment_mz_tolerance | float | 0.02 | m/z tolerance (Da) |
| alignment_mz_weight | float | 0.5 | Weight for m/z in scoring |
| alignment_rt_weight | float | 0.5 | Weight for RT in scoring |
| gap_fill_enabled | bool | True | Enable gap filling |
| gap_fill_rt_expansion | float | 1.5 | RT expansion factor |

### Stage 8: Export

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| export_mgf | bool | True | Export MGF format |
| export_msp | bool | True | Export MSP format |
| export_report | bool | True | Export processing report |

### Quality Filters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| final_height_threshold | float | 1000.0 | Final height filter |
| final_sn_threshold | float | 5.0 | Final S/N filter |
| final_gaussian_threshold | float | 0.6 | Final gaussian similarity |
| msms_intensity_threshold | float | 1000.0 | MS/MS intensity minimum |
| msms_relative_threshold | float | 0.01 | MS/MS relative intensity |
| msms_min_ions | int | 1 | Minimum MS/MS ions |

---

## 7. Output Files

### 7.1 Feature Table (features.csv)

Each row represents one aligned feature across replicates.

| Column | Description |
|--------|-------------|
| feature_id | Unique ID (F00001, F00002, ...) |
| precursor_mz | Precise precursor m/z value |
| rt_min | Retention time at peak apex (minutes) |
| rt_left | Left boundary of chromatographic peak |
| rt_right | Right boundary of chromatographic peak |
| signal_type | "ms1_detected" or "ms2_only" |
| n_fragments | Number of MS2 product ions |
| mean_height | Mean peak height across replicates |
| mean_area | Mean peak area across replicates |
| cv | Coefficient of variation across replicates |
| sn_ratio | Signal-to-noise ratio |
| formula | Molecular formula (if available) |
| adduct | Adduct type (if determined) |
| name | Annotation name from library match |
| ms2_spectrum | Product ion m/z and intensity pairs |
| height_rep1, height_rep2, ... | Per-replicate peak heights |
| area_rep1, area_rep2, ... | Per-replicate peak areas |

### 7.2 Spectral Libraries (MGF/MSP)

MS2 spectra for all features, compatible with SIRIUS, GNPS, MS-FINDER, and other annotation tools.

### 7.3 Processing Report

Text file summarizing:
- Input files and groupings
- Feature counts at each stage
- Deduplication statistics (isotopes removed, adducts merged, ISFs detected)
- Alignment statistics
- Runtime per stage

---

## 8. Tutorials

### 8.1 Basic Processing of Rice Leaf ASFAM Data

```bash
# 1. Prepare: 30 mzML files (10 segments x 3 replicates)
# Filename pattern: rice_075-174_rep1.mzML, rice_175-274_rep1.mzML, ...

# 2. Process with default parameters
asfam rice_*.mzML -o ./rice_results --mode positive --workers 4

# 3. Check results
ls ./rice_results/
# features.csv  ms2_spectra.mgf  ms2_spectra.msp  processing_report.txt
```

### 8.2 Processing with Spectral Library

```bash
# Use a combined MSP library for annotation
asfam rice_*.mzML -o ./rice_results --library combined_library.msp

# The pipeline will:
# 1. Use the library for m/z inference (Stage 2.5)
# 2. Use the library for annotation (Stage 6.5)
```

### 8.3 Optimizing Parameters for Sensitive Detection

For maximum sensitivity (more features, lower thresholds):

```json
{
    "peak_height_threshold": 100.0,
    "peak_sn_threshold": 3.0,
    "peak_width_min": 3,
    "peak_gaussian_threshold": 0.0,
    "ms1_min_height": 50.0,
    "min_fragments_for_feature": 2
}
```

For high specificity (fewer false positives):

```json
{
    "peak_height_threshold": 500.0,
    "peak_sn_threshold": 10.0,
    "peak_width_min": 5,
    "peak_gaussian_threshold": 0.7,
    "isotope_modified_cos_threshold": 0.90,
    "adduct_eic_pearson_threshold": 0.95,
    "isf_eic_pearson_threshold": 0.95
}
```

### 8.4 Using the GUI for Interactive Exploration

1. Launch GUI: `python -m asfam.gui.app`
2. Load files and run pipeline
3. After completion, use View Mode to switch between aligned and per-sample views
4. Click features in the scatter plot or table to examine:
   - EIC shape: Is this a genuine chromatographic peak?
   - MS2 spectrum: Are the fragments consistent?
   - Library match: Does the annotation make sense?
5. Use Re-annotate to try different libraries without rerunning detection

### 8.5 Downstream Analysis with SIRIUS

```bash
# 1. Run ASFAMProcessor
asfam *.mzML -o ./results

# 2. Use the exported MGF/MSP in SIRIUS
# Open SIRIUS -> Import -> ms2_spectra.mgf
# SIRIUS will predict molecular formulas, structures, and compound classes

# 3. Merge SIRIUS results with ASFAMProcessor feature table
# Match by feature_id or precursor_mz + RT
```

---

## 9. Troubleshooting

### 9.1 Common Issues

**Problem**: "No features detected"
- Check that mzML files contain both MS1 and MS2 scans
- Verify filename pattern includes m/z range (e.g., "100-129")
- Try lowering peak_height_threshold and peak_sn_threshold
- Enable verbose mode (-v) to check data loading

**Problem**: "Too many features / too much redundancy"
- Increase isotope, adduct, and ISF deduplication thresholds
- Check that ionization_mode matches your data (positive/negative)
- Increase min_fragments_per_feature to 3 or higher

**Problem**: "No annotations found"
- Verify spectral library format (MSP or MGF)
- Check that library covers your m/z range
- Try lowering matchms_similarity_threshold
- Ensure ionization mode matches between data and library

**Problem**: "Out of memory"
- Reduce n_workers (fewer parallel processes)
- Process fewer files at a time
- Close other memory-intensive applications

**Problem**: "Files not grouped correctly"
- Check filename patterns match the expected format
- Use GUI "Edit Samples" to manually assign files to groups
- Ensure replicate IDs are consistent across segments

### 9.2 Log Files

The GUI writes crash logs to `~/.asfam_logs/`. If the application crashes, check the most recent log file for details.

### 9.3 Performance Tips

- Use 4-8 workers for typical datasets (match your CPU core count)
- SSD storage improves mzML loading speed
- Close unnecessary applications to free RAM
- For very large datasets (>100 files), process in batches

### 9.4 Contact

- Bug reports: [GitHub Issues](https://github.com/yuanhonglun/WT_2.0/issues)
- Email: yuanhonglun@hotmail.com
