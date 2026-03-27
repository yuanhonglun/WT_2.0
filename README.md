# ASFAMProcessor

A dedicated data processing pipeline for All-ion Stepwise Fragmentation Acquisition Mode (ASFAM) LC-QTOF mass spectrometry data.

## Overview

ASFAMProcessor is designed to process data acquired using the ASFAM acquisition mode, which employs 1 Da HR-MRM stepping windows to capture high-resolution MS2 spectra with substantially reduced chimericity for virtually all precursor ions within the 75-1075 m/z range.

The software provides a complete feature extraction pipeline from raw mzML files to annotated feature tables, spectral libraries, and processing reports.

## Key Features

- **MS1/MS2 dual-driven feature detection**: Combines MS2-driven extraction (EIC peak detection and RT clustering) with complementary MS1-driven detection for comprehensive metabolome coverage
- **Two-pass MS1 assignment**: Strict and relaxed passes maximize the number of features with directly measured precursor m/z values
- **Multiple m/z inference strategies**: Spectral library matching and neutral loss consensus analysis infer precursor m/z for features without MS1 signal
- **Three-layer deduplication**:
  - Isotope removal: graph-based algorithm with three confidence tiers (MS1 pattern support, modified cosine, dual cosine + neutral loss cosine)
  - Adduct consolidation: neutral mass matching validated by EIC Pearson correlation
  - In-source fragmentation (ISF) detection: dual criteria (precursor-fragment m/z relationship + EIC correlation)
- **Spectral library annotation**: composite similarity scoring against MSP/MGF reference libraries
- **Cross-replicate alignment**: Gaussian similarity-based matching with gap filling
- **GUI and CLI**: interactive graphical interface and command-line interface
- **Open standard formats**: reads mzML, exports CSV, MGF, MSP

## Installation

### Requirements

- Python >= 3.9
- Windows / Linux / macOS

### Install from source

```bash
git clone https://github.com/yuanhonglun/WT_2.0.git
cd ASFAMProcessor
pip install -e .
```

### Dependencies

Automatically installed with pip:

| Package | Version | Purpose |
|---------|---------|---------|
| pymzml | >= 2.5.0 | mzML file parsing |
| numpy | >= 1.21 | Numerical computation |
| scipy | >= 1.7 | Signal processing, peak detection |
| pandas | >= 1.3 | Data tables |
| matchms | >= 0.18 | Spectral library matching |
| msbuddy | >= 0.2 | Molecular formula prediction |
| matplotlib | >= 3.5 | Plotting |
| PyQt5 | >= 5.15 | GUI framework |
| lxml | >= 4.6 | XML parsing |

## Quick Start

### GUI

```bash
python -m asfam.gui.app
```

1. Click **Add Files** to load mzML files
2. Select output directory
3. Optionally load a spectral library (MSP/MGF)
4. Adjust parameters if needed
5. Click **Run Pipeline**
6. Explore results in the feature table, EIC plot, and MS2 plot

### CLI

```bash
# Basic usage
asfam sample_100-129_rep1.mzML sample_130-159_rep1.mzML -o ./results

# With spectral library and custom config
asfam *.mzML -o ./output --library reference.msp --mode positive --workers 4 -v

# Using a config file
asfam *.mzML -o ./output --config my_config.json
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output directory (required) | - |
| `--library` | Spectral library file (MSP/MGF) | None |
| `--config` | Processing config JSON file | Default parameters |
| `--mode` | Ionization mode (positive/negative) | positive |
| `--workers` | Number of parallel workers | 4 |
| `-v, --verbose` | Enable verbose logging | Off |

## Input Data Format

### mzML Files

ASFAM data should be converted to mzML format (e.g., using MSConvert from ProteoWizard).

**Filename convention**: The parser extracts segment information from filenames:
- `SAMPLE_100-129_rep1.mzML` (sample_seglow-seghigh_rep)
- `SAMPLE_100-129_1_30_1.mzML` (sample_seglow-seghigh_X_step_rep)

Files with the same sample name and replicate ID are automatically grouped.

### Spectral Libraries

Supported formats:
- **MSP** (NIST/MassBank format): NAME, FORMULA, PRECURSOR_MZ, PRECURSOR_TYPE, Num Peaks, peak list
- **MGF** (Mascot Generic Format): BEGIN IONS / END IONS blocks with PEPMASS, peak list

## Pipeline Stages

```
Stage 0: Load          → Parse mzML files, organize scan cycles
Stage 1: MS2 Detection → Extract product ion EICs, detect peaks, cluster by RT
Stage 1b: MS1 Detection → Detect MS1-only features (weak/no fragmentation)
Stage 2: MS1 Assignment → Two-pass precise m/z assignment from MS1 spectra
Stage 2.5: m/z Inference → Library matching + neutral loss consensus for MS2-only features
Stage 3: Segment Merge  → Remove boundary duplicates across m/z segments
Stage 4: Isotope Dedup  → Graph-based isotope cluster removal (3 confidence tiers)
Stage 5: Adduct Dedup   → Neutral mass matching + EIC correlation validation
Stage 6: ISF Detection  → Dual-criteria in-source fragmentation removal
Stage 6.5: Annotation   → Spectral library matching (composite similarity)
Stage 7: Alignment      → Cross-replicate Gaussian similarity matching + gap filling
Stage 8: Export         → CSV feature table, MGF, MSP, processing report
```

## Output Files

| File | Description |
|------|-------------|
| `features.csv` | Feature table with m/z, RT, quantification, annotations |
| `ms2_spectra.mgf` | MS2 spectra in MGF format |
| `ms2_spectra.msp` | MS2 spectra in MSP format |
| `processing_report.txt` | Stage-by-stage statistics |

### Feature Table Columns

| Column | Description |
|--------|-------------|
| feature_id | Unique identifier (F00001, F00002, ...) |
| precursor_mz | Precise precursor m/z |
| rt_min | Retention time at apex (minutes) |
| signal_type | "ms1_detected" or "ms2_only" |
| n_fragments | Number of MS2 product ions |
| mean_height | Mean peak height across replicates |
| mean_area | Mean peak area across replicates |
| cv | Coefficient of variation |
| name | Annotation name (if matched) |
| formula | Molecular formula (if available) |
| height_repN / area_repN | Per-replicate quantification |

## Configuration

All parameters can be saved/loaded as JSON:

```python
from asfam.config import ProcessingConfig

# Load custom config
config = ProcessingConfig.load("my_config.json")

# Modify parameters
config.peak_height_threshold = 500.0
config.isotope_modified_cos_threshold = 0.9

# Save
config.save("updated_config.json")
```

See the [User Guide](USER_GUIDE.md) for a complete parameter reference.

## Project Files

The GUI supports saving and loading project files (`.asfam` format) that preserve:
- Processing configuration
- All detected features and candidates
- File paths and sample groupings
- Stage statistics
- Annotation results

## System Requirements

- **RAM**: 4 GB minimum, 8+ GB recommended (depends on data size)
- **CPU**: Multi-core recommended (parallel processing across files)
- **Disk**: ~50 MB per mzML file for intermediate results

## Citation

If you use ASFAMProcessor in your research, please cite:

> Yuan H, Huang S, Liu Z, et al. WT 2.0: Unveiling the "dark matter" in the metabolome using all-ion stepwise fragmentation acquisition mode (ASFAM) and dedicated feature extraction pipeline. *Nature Communications* (under review).

## License

BSD 3-Clause License with Non-Commercial Clause.

- Free for academic and non-commercial use
- Commercial use requires written permission: yuanhonglun@hotmail.com

Copyright (c) 2025-2026, Honglun Yuan, Hainan University.

## Contact

- Bug reports: [GitHub Issues](https://github.com/yuanhonglun/WT_2.0/issues)
- Email: yuanhonglun@hotmail.com
