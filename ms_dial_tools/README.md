# MS-DIAL results integration tools

Two standalone helper scripts used in the WT 2.0 study to integrate and
de-duplicate the **per-segment peak lists exported by MS-DIAL**, so that a
segmented acquisition (ASFAM, DDA or DIA) processed in MS-DIAL can be compared
with ASFAMProcessor output on equal terms.

| script | use |
| --- | --- |
| `MS-DIAL_results_integration_tool_260320_1.py` | single run: choose one set of per-segment MS-DIAL exports and write the integrated, de-duplicated table |
| `MS-DIAL_results_integration_tool_batch_260320_1.py` | batch version: process several sample/segment sets in one pass, with a selection dialog |

Both scripts open a small Tkinter window, so run them in a desktop session:

```bash
python -m pip install pandas
python MS-DIAL_results_integration_tool_260320_1.py
```

## What they do

1. Read the MS-DIAL alignment / peak-list export of each precursor segment
   (`.txt` or `.csv`) and tag every row with the segment it came from.
2. Remove MS-DIAL-flagged isotopes and rows failing the Peak Height or S/N
   filter.
3. Consolidate features that share a neutral mass and retention time across
   segments.
4. Clean MS2 spectra with absolute and relative intensity thresholds and drop
   records with too few fragment ions.
5. Run a final graph pass that links co-eluting residual isotopes,
   halogen-isotope patterns and spectral duplicates by precursor spacing or by
   modified-cosine / neutral-loss similarity, keeping the highest-intensity
   representative of each group.

## Settings used in the paper

The strict MS-DIAL reference set used for the software benchmark (Fig. 4)
required Peak Height >= 10,000, S/N >= 5 and at least four fragment ions. The
lenient reference used for the sensitivity and height-stratified recall checks
(Supplementary Fig. 1) was built without the Peak Height and S/N filters. All
other parameters were left at the defaults shown in the dialogs.
