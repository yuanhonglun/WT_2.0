# paper_scripts

Scripts behind the figures and tables of the WT 2.0 manuscript. Each script takes its
inputs from the command line (run `python <script> --help`); defaults point at the
folder layout of the MetaboLights study MTBLS15138 and of an ASFAMProcessor run
directory (`features.csv`, `ms2_spectra.msp`, `_work/`, `_debug/processing_report.txt`).
Run order follows the table.

| Script | Produces | Inputs |
|---|---|---|
| `features_io.py` | Shared: Reads features.csv without truncating names that contain '#' | — |
| `paper_palette.py` | All figures: Shared colour-blind-safe palette | — |
| `fig4_common.py` | Fig. 4: Shared readers / styles for the Fig. 4 scripts | — |
| `run_dataset.py` | Re-processing: Runs ASFAMProcessor on one manuscript data set (presets rep3_full / rep3_ms1only / rep1 / rep2 / rep123 / rice / cancer_pos / cancer_neg / exp_dilution / exp_blank / exp_mix) | mzML files (validation and rice: MetaboLights MTBLS15138; cancer discovery pools and supplementary experiments: from the corresponding author); pos.msp or Neg.msp (Zenodo, see Code availability) |
| `precursor_sources.py` | Precursor provenance (Fig. 4, Supplementary Fig. NL): mz_source × mz_confidence × ms2_only_tier tallies, nl_n_support distribution | features.csv + _work/ of a run |
| `nl_window_consistency.py` | Precursor provenance: Agreement between neutral-loss consensus and window centroid on MS2-only features | features.csv of a run |
| `window_centroid_precision.py` | Precursor provenance: Window-centroid error against the quantified ion on MS1-detected features (sets the ms1_lowres window) | _work/ of a run + its mzML files |
| `nl_self_validation.py` | Fig. 4e,f; Supplementary Table 3: Neutral-loss self-validation with the production predictor, plus the nominal-mass + mass-defect baseline | _work/ checkpoint of runs/rep3_full |
| `annotation_outcome.py` | Supplementary Fig. (precursor inference): Top-1 annotation outcome under true / inferred / centroid precursors and three NL windows | self-validation table; lib/lcms/pos.msp |
| `ablation_table.py` | Supplementary Fig. (ablation): MS1-driven-only versus full pipeline: counts, overlaps, recovered annotations | runs/rep3_full and runs/rep3_ms1only |
| `fold_audit.py` | Response letter / Methods (alignment): Audit of candidates folded into other rows by the alignment stage | runs/rep3_full |
| `gen_fig4.py` | Fig. 4: Validation against MS-DIAL, neutral-loss self-validation, annotation outcome | runs/rep3_full, cached MS-DIAL references |
| `make_fig1b_table.py` | Fig. 1: Fig. 1b fact table and panel-a recolouring (SVG) | Fig 1.svg master |
| `gen_fig2.py` | Fig. 2: Workflow schematic | — |
| `redundancy_ledger.py` | Supplementary Fig. ablation / stage ledger: Stage-by-stage feature ledger, duplicate classes, MS2-only families | features.csv, _work/, processing_report.txt of a run |
| `rerun_delta.py` | Response letter: Old-versus-new comparison of a data set | two run directories |
| `throughput_table.py` | Supplementary table (throughput): Features / annotations per injection and per hour for WT 2.0, DDA, DIA | Fig. 5 inputs (ASFAMProcessor exports, MS-DIAL exports) |
| `library_loo.py` | Supplementary Fig. decoy: Leave-one-out false-formula rate on MassBank at 0.5 / 0.05 / 0.01 Da | MassBank MSP (download) |
| `loo_collect.py` | Supplementary Fig. decoy: Collects the leave-one-out runs into one summary | library_loo outputs |
| `sample_library.py` | Supplementary Fig. decoy: Seeded 10 % sample of the spectral library for the decoy search | pos.msp |
| `spectral_decoy_fdr.py` | Supplementary Fig. decoy: Target–decoy FDR of high-score annotations on real query spectra | ms2_spectra.msp of a run; pos.msp |
| `make_fig3.py` | Fig. 3: Acquisition-mode comparison under identical MS-DIAL processing | fig 3 workbooks |
| `make_fig5.py` | Fig. 5: Complete-workflow comparison | ASFAMProcessor and MS-DIAL exports (3 replicates) |
| `analyze_rice.py` | Fig. 6, Supplementary Table 5, Source Data: Rice salt-stress statistics (BH-FDR ≤ 0.05 and |log2FC| ≥ 1), Table 5, Fig. 6 | features.csv of the rice run |
| `cancer_common.py` | Fig. 7, Supplementary Figs. 4–5, Supplementary Tables 9–10: Shared loaders and statistics for the cohort analyses | Supplementary Tables 8–9, clinical table, split lists |
| `cancer_metrics.py` | Fig. 7e, Supplementary Fig. 5: AUC with CIs, cut-offs, LC vs BLD held-out, Cyfra21-1, covariate models for the published models | per-sample probabilities of the published models |
| `cancer_classifier.py` | Fig. 7e, Supplementary Fig. 5, new metrics table: Training-set-only redo of the classifier (LC vs BLD main; LC vs HC and LC vs non-LC exploratory) | Supplementary Table 9, split lists, clinical table |
| `cancer_cohort_tables.py` | Fig. 7c,d, Supplementary Fig. 4, SI tables: Panel-member provenance, outlier sensitivity, exclusion counts, internal standards, missingness | Supplementary Tables 8–9, collaborator workbook |
| `make_fig7.py` | Fig. 7: Cohort figure | analysis/cancer outputs |
| `make_supplementary_figures.py` | Supplementary Figs. 1–3: Standards evidence and other SI figures | see script |
| `make_supplementary_figures_cancer.py` | Supplementary Figs. 4–5: Cohort SI figures | analysis/cancer outputs |
| `cvd_check.py` | All figures: Deuteranopia / protanopia simulation of every figure | PNG figures |

Requirements: Python ≥ 3.12, numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib,
openpyxl, colorspacious, and the ASFAMProcessor packages of this repository installed in
editable mode (`pip install -e packages/metabo_core -e packages/metabo_gui -e apps/asfam_processor`).
