
import numpy as np
import pickle

import pandas as pd
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")

def parse_ms2(ms2_str):
    """
    简化版：将 ms2_str 解析为列表 (mz, intensity)，并做最大值归一化到[0,1]。
    """
    if not ms2_str or pd.isna(ms2_str):
        return []
    pairs_str = ms2_str.strip().split(";")
    mz_list = []
    intensity_list = []
    for pair in pairs_str:
        mz_s, it_s = pair.strip().split()
        mz_val, it_val = float(mz_s), float(it_s)
        mz_list.append(mz_val)
        intensity_list.append(it_val)
    max_inten = max(intensity_list) if len(intensity_list) > 0 else 1
    if max_inten == 0:
        max_inten = 1
    intensity_list = [v / max_inten for v in intensity_list]

    return mz_list, intensity_list


def accumulate_intensities(ms2_mz, ms2_intensity, bins):
    """Accumulates intensities into specified bins."""
    sparse_data = np.zeros(len(bins) + 1)
    indices = np.digitize(ms2_mz, bins)

    for index, intensity in zip(indices, ms2_intensity):
        if 0 < index < len(bins):
            sparse_data[index - 1] += intensity

    return sparse_data


def process_batch(batch_data, all_data, key):
    """Processes and appends a batch of data to the all_data dictionary."""
    if len(batch_data) > 0:
        batch_sparse_matrix = sp.csr_matrix(batch_data)
        if all_data[key] is None:
            all_data[key] = batch_sparse_matrix
        else:
            all_data[key] = sp.vstack([all_data[key], batch_sparse_matrix])


def process_spectra_to_pickle(df, pickle_file, mz_min=50, mz_max=1076, bin_width=0.02, batch_size=10000):
    """Processes spectral data and saves it to a pickle file."""

    # Create bins
    bins = np.round(np.arange(mz_min, mz_max + bin_width, bin_width), 2)

    # Initialize all_data dictionary
    all_data = {
        'left_intensities': None,
        'right_intensities': None,
        'left_rt': [],
        'right_rt': [],
        'labels': [],
        'mz_bins': np.append(bins[:-1], [0, 1])
    }


    left_batch_data, right_batch_data = [], []
    batch_labels, left_batch_rt, right_batch_rt = [], [], []

    for i, row in df.iterrows():
        left_ms2_mz, left_ms2_intensity = parse_ms2(row[2])
        right_ms2_mz, right_ms2_intensity = parse_ms2(row[5])

        # Accumulate left intensities
        left_batch_data.append(accumulate_intensities(left_ms2_mz, left_ms2_intensity, bins))
        left_batch_rt.append(row[1])
        batch_labels.append(row[6])

        # Process left batch
        if (i + 1) % batch_size == 0:
            process_batch(left_batch_data, all_data, 'left_intensities')
            all_data['labels'].extend(batch_labels)
            all_data['left_rt'].extend(left_batch_rt)
            left_batch_data, batch_labels, left_batch_rt = [], [], []

            # Accumulate right intensities
        right_batch_data.append(accumulate_intensities(right_ms2_mz, right_ms2_intensity, bins))
        right_batch_rt.append(row[4])

        # Process right batch
        if (i + 1) % batch_size == 0:
            process_batch(right_batch_data, all_data, 'right_intensities')
            all_data['right_rt'].extend(right_batch_rt)
            right_batch_data, right_batch_rt = [], []

            # Process remaining data after the loop
    process_batch(left_batch_data, all_data, 'left_intensities')
    process_batch(right_batch_data, all_data, 'right_intensities')
    all_data['labels'].extend(batch_labels)
    all_data['left_rt'].extend(left_batch_rt)
    all_data['right_rt'].extend(right_batch_rt)

    # Save data to pickle
    with open(pickle_file, 'wb') as f:
        pickle.dump(all_data, f)

    #print(f'Data has been saved to {pickle_file}')

# # 使用示例
# import time
# t1 = time.time()
# df = pd.read_csv(r"D:\work\WT2.0\peakalignment\val\select_p1.csv")
# process_spectra_to_pickle(df, "./select_p1.pkl")
# t2 = time.time()
# print(t2 - t1)

