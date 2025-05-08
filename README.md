# 1. Installation
WT_2 requires Python 3.10. We recommend creating a virtual environment using conda:
```bash
conda create --name WT2 python=3.10
conda activate WT2
pip install WT_2
```

# 2. Usage Guide
## 2.1 Data Preparation
- Each sample requires 10 raw MGF files and MS-DIAL processing results.
- Place all MGF files and MS-DIAL results for the same sample into a folder named after the sample.
- Multiple samples should be organized in separate folders.
- Refer to the sample1 folder in test_data as an example.


## 2.2 Peak Process
```python
from WT_2 import MultiprocessingManager

sample_folder = "./test_data/sample1" 
out_dir = "./test_data/sample1" 

manager = MultiprocessingManager(
    outer_max_workers=1,
    inner_max_workers=8,
    mgf_folder=sample_folder,
    out_dir=out_dir,
)
manager.process_mgf_files()

```
- Parameters:
  - process_mgf_files() will automatically create a result folder in out_dir to store peak extraction and clustering results.
  - sample_folder: Sample directory path.
  - out_dir: Output directory path.
  - outer_max_workers: Number of outer processes for MGF file processing (default: 1).
  - inner_max_workers: Number of inner processes for m/z processing (default: 8).
  - RT_start: Left boundary of retention time (RT) range (seconds).
  - RT_end: Right boundary of RT range (seconds).
  - fp_wid: Peak detection window width (default: 6).
  - fp_sigma: Peak detection sigma (default: 2).
  - fp_min_noise: Noise threshold for peak detection (default: 200).
  - group_wid: Peak clustering window width (default: 6).
  - group_sigma:  Peak clustering sigma (default: 0.5).
  

## 2.3 Peak Deduplication
```python
from WT_2 import Deduplicator


sample_name = os.path.basename(sample_folder)
msdial_path = "./test_data/sample1/sample1_Q1_peak_df.csv"


deduplicator = Deduplicator(
    peak_result_dir=os.path.join(out_dir, "result"),
    msdial_out_path=msdial_path,
    sample_name=sample_name,
    useHrMs1=True,
    HrMs1model_path=None
)
deduplicator.remove_msdial_duplicate()
peak_outpath, group_outpath = deduplicator.filter_p3_group()
```

- Parameters:
    - peak_result_dir: Directory containing peak extraction results (default: result folder in out_dir).
    - msdial_out_path: Path to MS-DIAL input file.
    - sample_name: Sample name (default: folder name).
    - useHrMs1: Whether to use high-resolution MS1 model (True for high-res, False for low-res, default: False).
    - HrMs1model_path: Path to high-resolution MS1 prediction model. If None, the pretrained model will be downloaded automatically. Pretrained model available at test_data/models/HrMs1.pth or [Hugging Face](https://huggingface.co/liuzhenhuan123/HrMs1/tree/main)

## 2.4 Metabolite Identification
```python
from WT_2 import MspGenerator, MspFileLibraryMatcher
import pandas as pd

# Generate MSP file from deduplicated P3 group results
df = pd.read_csv(group_outpath)
out_msp_path = os.path.join(os.path.dirname(group_outpath), sample_name + ".msp")
msp_generator = MspGenerator(df, out_msp_path, useHrMs1=False)


# Library matching (requires MSP-format library)
out_match_path = os.path.join(os.path.dirname(group_outpath), sample_name + "_match_library_out.csv")

library_matcher = MspFileLibraryMatcher(
    query_msp_path=out_msp_path,
    library_msp_path="./test_data/library_msp",
    out_path=out_match_path,
    num=1
)
library_matcher.calculateCosineBoth()

```

- MspGenerator Parameters:
  - df: Deduplicated P3 group dataframe.
  - out_msp_path: Output path for MSP file.
  - useHrMs1: Whether to use high-resolution MS1 data (True for high-res, False for low-res, default: False). If set to true, ensure that the high-resolution MS1 model is used during the Peak Deduplication step.

- MspFileLibraryMatcher Parameters:
    - query_msp_path: MSP file generated from P3 group results (standard MSP format).
    - library_msp_path: Path to reference MSP library.
    - out_path: Output path for matching results.
    - num: Number of top matches to keep (default: 1).


## 2.5 Metabolite quantification
```python
from WT_2 import SampleQuantity


quantity_folder = "./test_data/quantity_prepared"


quantifier = SampleQuantity(
    quantity_folder=quantity_folder,
    quantity_out_path=quantity_folder,
    ref_file=None,
    useHrMs1=True,
    uesSampleAligmentmodel=True,
    SampleAligmentmodel_path=None
)
quantifier.quantity_processor()

```

- Parameters:
    - quantity_folder: Directory containing deduplicated P3 results for all samples.
    - quantity_out_path: Output directory.
    - ref_file: Reference sample file (uses first sample as reference if None).
    - useHrMs1:  Whether to use high-resolution MS1 data (requires prior high-res processing).
    - uesSampleAligmentmodel: Enable sample alignment model.
    - SampleAligmentmodel_path: Path to sample alignment model. If None, downloads pretrained model. Pretrained model available at test_data/models/samplealigment.pth or [Hugging Face](https://huggingface.co/liuzhenhuan123/Samplealigment/tree/main)
