import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Toplevel, Listbox, MULTIPLE

import pandas as pd


# =========================
# Helper functions
# =========================

PROTON_MASS = 1.0072766
MASS_H2O = 18.0105647
MASS_NH4 = 18.0338257
MASS_NA = 22.9892213
MASS_K = 38.9631585
C13_DELTA = 1.003355
CL37_DELTA = 1.99705
BR81_DELTA = 1.99795

# Default parameters for the additional isotope-like deduplication step
ADDITIONAL_ISOTOPE_RT_TOL = 0.01          # min
ADDITIONAL_ISOTOPE_OVERLAP_RATIO = 0.80   # overlap / min(width1, width2)
ADDITIONAL_ISOTOPE_MZ_TOL = 0.01          # Da, for classic isotope gaps
ADDITIONAL_INTEGER_STEP_TOL = 0.02        # Da, for relaxed near-integer gaps
ADDITIONAL_FRAGMENT_MZ_TOL = 0.02         # Da
ADDITIONAL_PRECURSOR_EXCLUSION = 1.5      # Da, remove precursor-region ions before similarity
ADDITIONAL_MODIFIED_COS_THRESHOLD = 0.85
ADDITIONAL_MODIFIED_COS_THRESHOLD_RELAXED = 0.90
ADDITIONAL_MIN_SHIFTED_MATCHES = 3
ADDITIONAL_MIN_SHIFTED_MATCHES_RELAXED = 4
ADDITIONAL_NEUTRAL_LOSS_THRESHOLD = 0.85
ADDITIONAL_MIN_NL_MATCHES = 3
MAX_ISOTOPE_STEP = 4


def safe_to_float(x, default=math.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def read_msdial_alignment_table(file_path):
    """Read an MS-DIAL alignment-export table, supporting TXT and CSV."""
    candidates = []

    if file_path.lower().endswith('.txt'):
        candidates = [
            dict(sep='\t', skiprows=4),
            dict(sep='\t', skiprows=0),
        ]
    else:
        candidates = [
            dict(sep=None, engine='python', skiprows=4),
            dict(sep=None, engine='python', skiprows=0),
            dict(sep=',', skiprows=4),
            dict(sep=',', skiprows=0),
            dict(sep='\t', skiprows=4),
            dict(sep='\t', skiprows=0),
        ]

    last_error = None
    for kwargs in candidates:
        try:
            df = pd.read_csv(file_path, on_bad_lines='skip', **kwargs)
            if 'Alignment ID' in df.columns:
                return df
        except Exception as e:
            last_error = e

    if last_error is not None:
        raise last_error
    raise ValueError(f"Failed to read a valid MS-DIAL alignment table: {file_path}")


def read_first_lines(file_path, n=4):
    lines = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as handle:
        for _ in range(n):
            line = handle.readline()
            if line == '':
                break
            lines.append(line.rstrip('\n'))
    return lines


def get_true_sample_columns(file_path, df):
    """
    Identify actual sample columns after 'MS/MS spectrum'.
    This excludes MS-DIAL summary columns such as Average/Stdev when present.
    """
    if 'MS/MS spectrum' not in df.columns:
        return []

    msms_idx = df.columns.get_loc('MS/MS spectrum')
    header_columns = list(df.columns)
    trailing_columns = header_columns[msms_idx + 1:]

    metadata_lines = read_first_lines(file_path, 4)
    if len(metadata_lines) >= 2:
        file_type_parts = metadata_lines[1].split('\t')
        if len(file_type_parts) >= len(header_columns):
            sample_columns = [
                header_columns[i]
                for i in range(msms_idx + 1, len(header_columns))
                if file_type_parts[i].strip().lower() == 'sample'
            ]
            if sample_columns:
                return sample_columns

    fallback_columns = []
    for col in trailing_columns:
        col_str = str(col).strip()
        if col_str.lower().startswith('unnamed'):
            continue
        numeric_like = col_str.replace('.', '', 1).isdigit()
        if numeric_like:
            continue
        fallback_columns.append(col)
    return fallback_columns


def parse_positive_int_list(text, field_name):
    raw_items = [item.strip() for item in str(text).split(',') if item.strip()]
    if not raw_items:
        raise ValueError(f"{field_name} must not be empty.")

    values = []
    for item in raw_items:
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"{field_name} must contain integers only.") from exc
        if value <= 0:
            raise ValueError(f"{field_name} must contain positive integers only.")
        values.append(value)
    return values


def parse_optional_label_list(text, expected_count):
    labels = [item.strip() for item in str(text).split(',') if item.strip()]
    if not labels:
        return [f'Batch{i + 1}' for i in range(expected_count)]
    if len(labels) != expected_count:
        raise ValueError("The number of combined total-batch labels must match the number of replicate-count entries.")
    return labels


def compact_nonzero_values(row_values, keep_count):
    compacted = []
    for value in row_values:
        numeric_value = safe_to_float(value, default=0.0)
        if not math.isnan(numeric_value) and abs(numeric_value) > 0:
            compacted.append(numeric_value)
    compacted = compacted[:keep_count]
    while len(compacted) < keep_count:
        compacted.append(0.0)
    return compacted


def compress_total_batch_table(file_path, df, replicate_counts, dataset_labels):
    """
    Compress a single MS-DIAL total-batch table.

    Example:
    original columns: seg1_R1 seg1_R2 seg1_R3 seg2_R1 seg2_R2 seg2_R3
    compressed to:    Batch1_R1 Batch1_R2 Batch1_R3

    When multiple datasets are present in the same MS-DIAL total batch,
    the function assumes their sample blocks are consecutive.
    """
    sample_columns = get_true_sample_columns(file_path, df)
    if not sample_columns:
        raise ValueError("No valid sample columns were detected after 'MS/MS spectrum'.")

    total_actual_replicates = sum(replicate_counts)
    if len(sample_columns) % total_actual_replicates != 0:
        raise ValueError(
            "For combined total-batch mode, the number of detected sample columns is not divisible by the total actual replicate count. "
            f"Detected sample columns: {len(sample_columns)}; actual replicate count sum: {total_actual_replicates}."
        )

    segment_count = len(sample_columns) // total_actual_replicates
    if segment_count < 1:
        raise ValueError("The inferred number of ASFAM segments is invalid.")

    print(f"Combined total-batch mode: detected {len(sample_columns)} sample columns, actual replicate count sum = {total_actual_replicates}, inferred segment count = {segment_count}")

    metadata_end_idx = df.columns.get_loc('MS/MS spectrum') + 1
    metadata_columns = list(df.columns[:metadata_end_idx])
    compressed_df = df[metadata_columns].copy()

    start = 0
    output_sample_columns = []

    for dataset_idx, (replicate_count, dataset_label) in enumerate(zip(replicate_counts, dataset_labels), start=1):
        block_size = replicate_count * segment_count
        block_columns = sample_columns[start:start + block_size]
        if len(block_columns) != block_size:
            raise ValueError(f"Dataset block {dataset_idx} does not contain the expected number of columns.")

        block_matrix = df[block_columns].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
        compressed_rows = [compact_nonzero_values(row, replicate_count) for row in block_matrix]

        new_column_names = [f"{dataset_label}_R{i + 1}" for i in range(replicate_count)]
        compressed_block_df = pd.DataFrame(compressed_rows, columns=new_column_names, index=df.index)
        compressed_df = pd.concat([compressed_df, compressed_block_df], axis=1)
        output_sample_columns.extend(new_column_names)

        print(f"Compressed dataset block {dataset_idx}: original columns = {block_columns}; output columns = {new_column_names}")
        start += block_size

    return compressed_df, output_sample_columns


def merge_response_and_snr_tables(resp_df, snr_df, response_sample_columns, snr_sample_columns):
    if 'Alignment ID' not in resp_df.columns or 'Alignment ID' not in snr_df.columns:
        raise ValueError("Column 'Alignment ID' was not found. Please check whether the MS-DIAL table header is correct.")

    if len(response_sample_columns) != len(snr_sample_columns):
        raise ValueError(
            "The numbers of detected response and S/N sample columns do not match after preprocessing. "
            f"Response columns: {len(response_sample_columns)}; S/N columns: {len(snr_sample_columns)}."
        )

    snr_subset = snr_df[['Alignment ID'] + snr_sample_columns].copy()
    rename_map = {
        snr_col: f"SNR_{resp_col}"
        for resp_col, snr_col in zip(response_sample_columns, snr_sample_columns)
    }
    snr_subset.rename(columns=rename_map, inplace=True)

    merged_df = resp_df.merge(snr_subset, on='Alignment ID', how='left')
    return merged_df


def parse_msms_spectrum(spectrum_str):
    """Parse MS-DIAL spectrum text, supporting both space and semicolon separators."""
    if pd.isna(spectrum_str) or not str(spectrum_str).strip():
        return []

    raw = str(spectrum_str).strip().replace(';', ' ')
    peaks = []
    for token in raw.split():
        if ':' not in token:
            continue
        mz_str, intensity_str = token.split(':', 1)
        try:
            mz = float(mz_str)
            intensity = float(intensity_str)
            peaks.append((mz, intensity))
        except ValueError:
            continue
    return peaks


def spectrum_to_string(peaks):
    if not peaks:
        return ''
    return ' '.join([f'{mz:.5f}:{int(round(intensity))}' for mz, intensity in peaks])


def clean_msms_peaks(peaks, intensity_threshold, response_percentage):
    if not peaks:
        return []

    peaks = [(mz, intensity) for mz, intensity in peaks if intensity >= intensity_threshold]
    if not peaks:
        return []

    base_intensity = max(intensity for _, intensity in peaks)
    relative_cutoff = base_intensity * response_percentage
    peaks = [(mz, intensity) for mz, intensity in peaks if intensity >= relative_cutoff]
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks


def remove_precursor_region(peaks, precursor_mz, exclusion_da):
    return [(mz, intensity) for mz, intensity in peaks if abs(mz - precursor_mz) > exclusion_da]


def vector_norm(peaks):
    return math.sqrt(sum(intensity * intensity for _, intensity in peaks))


def greedy_match(peaks_a, peaks_b, mz_tolerance, shift=0.0):
    """
    Match peaks_a to peaks_b with optional shift applied to peaks_a mz.
    Returns (sum_products, matched_count).
    Greedy matching by descending product score.
    """
    candidates = []
    for i, (mz_a, int_a) in enumerate(peaks_a):
        target = mz_a + shift
        for j, (mz_b, int_b) in enumerate(peaks_b):
            if abs(target - mz_b) <= mz_tolerance:
                candidates.append((int_a * int_b, i, j))

    candidates.sort(reverse=True, key=lambda x: x[0])
    used_a = set()
    used_b = set()
    sum_products = 0.0
    matched_count = 0
    for score, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        sum_products += score
        matched_count += 1
    return sum_products, matched_count


def modified_cosine(peaks_a, peaks_b, precursor_a, precursor_b, mz_tolerance):
    if not peaks_a or not peaks_b:
        return 0.0, 0

    shift = precursor_b - precursor_a
    sum_direct, count_direct = greedy_match(peaks_a, peaks_b, mz_tolerance, shift=0.0)
    sum_shifted, count_shifted = greedy_match(peaks_a, peaks_b, mz_tolerance, shift=shift)

    if sum_shifted >= sum_direct:
        sum_products = sum_shifted
        matched_count = count_shifted
    else:
        sum_products = sum_direct
        matched_count = count_direct

    denom = vector_norm(peaks_a) * vector_norm(peaks_b)
    if denom == 0:
        return 0.0, matched_count
    return sum_products / denom, matched_count


def neutral_loss_cosine(peaks_a, peaks_b, precursor_a, precursor_b, mz_tolerance):
    if not peaks_a or not peaks_b:
        return 0.0, 0

    nl_a = [(precursor_a - mz, intensity) for mz, intensity in peaks_a if precursor_a - mz > 0]
    nl_b = [(precursor_b - mz, intensity) for mz, intensity in peaks_b if precursor_b - mz > 0]
    if not nl_a or not nl_b:
        return 0.0, 0

    sum_products, matched_count = greedy_match(nl_a, nl_b, mz_tolerance, shift=0.0)
    denom = vector_norm(nl_a) * vector_norm(nl_b)
    if denom == 0:
        return 0.0, matched_count
    return sum_products / denom, matched_count


def peak_overlap_ratio(row_a, row_b):
    """
    Calculate chromatographic overlap ratio.
    Prefer RT left/right boundaries when available; otherwise, fall back to RT-only proximity.
    """
    left_a = safe_to_float(row_a.get('RT left(min)'))
    right_a = safe_to_float(row_a.get('RT right (min)'))
    left_b = safe_to_float(row_b.get('RT left(min)'))
    right_b = safe_to_float(row_b.get('RT right (min)'))

    if not any(math.isnan(v) for v in [left_a, right_a, left_b, right_b]):
        width_a = max(0.0, right_a - left_a)
        width_b = max(0.0, right_b - left_b)
        if width_a > 0 and width_b > 0:
            overlap = max(0.0, min(right_a, right_b) - max(left_a, left_b))
            return overlap / min(width_a, width_b)

    rt_a = safe_to_float(row_a.get('Average Rt(min)'))
    rt_b = safe_to_float(row_b.get('Average Rt(min)'))
    if math.isnan(rt_a) or math.isnan(rt_b):
        return 0.0
    return 1.0 if abs(rt_a - rt_b) <= ADDITIONAL_ISOTOPE_RT_TOL else 0.0


def calculate_mw(row, mode):
    precursor_mz = safe_to_float(row['Average Mz'])
    adduct = str(row.get('Adduct type', '')).strip()

    if math.isnan(precursor_mz):
        return 0.0

    if mode == 'Pos_Mode':
        if adduct == '[M+H]+':
            return precursor_mz - PROTON_MASS
        elif adduct == '[M+H-H2O]+':
            return precursor_mz + (MASS_H2O - PROTON_MASS)
        elif adduct == '[M+H -H2O]+':
            return precursor_mz + (MASS_H2O - PROTON_MASS)
        elif adduct == '[M+NH4]+':
            return precursor_mz - MASS_NH4
        elif adduct == '[M+Na]+':
            return precursor_mz - MASS_NA
        elif adduct == '[M+K]+':
            return precursor_mz - MASS_K
        elif adduct == '[M]+':
            return precursor_mz
        else:
            return 0.0
    elif mode == 'Neg_Mode':
        if adduct == '[M-2H]2-':
            return precursor_mz * 2 + 2 * PROTON_MASS
        elif adduct == '[M-H]-':
            return precursor_mz + PROTON_MASS
        elif adduct == '[M-H2O-H]-':
            return precursor_mz + MASS_H2O + PROTON_MASS
        else:
            return 0.0
    return 0.0


def ms1_isotope_support(ms1_iso_a, ms1_iso_b, precursor_a, precursor_b, tol=0.01):
    def extract_mzs(ms1_iso_str):
        if pd.isna(ms1_iso_str):
            return []
        mzs = []
        for token in str(ms1_iso_str).strip().split():
            if ':' not in token:
                continue
            mz_str = token.split(':', 1)[0]
            try:
                mzs.append(float(mz_str))
            except ValueError:
                continue
        return mzs

    mzs_a = extract_mzs(ms1_iso_a)
    mzs_b = extract_mzs(ms1_iso_b)

    a_supports_b = any(abs(mz - precursor_b) <= tol for mz in mzs_a)
    b_supports_a = any(abs(mz - precursor_a) <= tol for mz in mzs_b)
    return a_supports_b or b_supports_a


def classify_isotope_gap(delta_mz):
    """
    Return 'classic' for canonical isotope-like gaps,
    'relaxed' for near-integer gaps (1-4 Da), otherwise None.
    """
    if delta_mz <= 0:
        return None

    for n in range(1, MAX_ISOTOPE_STEP + 1):
        if abs(delta_mz - n * C13_DELTA) <= ADDITIONAL_ISOTOPE_MZ_TOL:
            return 'classic'

    if abs(delta_mz - CL37_DELTA) <= ADDITIONAL_ISOTOPE_MZ_TOL:
        return 'classic'
    if abs(delta_mz - BR81_DELTA) <= ADDITIONAL_ISOTOPE_MZ_TOL:
        return 'classic'

    rounded = round(delta_mz)
    if 1 <= rounded <= MAX_ISOTOPE_STEP and abs(delta_mz - rounded) <= ADDITIONAL_INTEGER_STEP_TOL:
        return 'relaxed'

    return None


def build_isotope_graph(df):
    """Return adjacency list of isotope-like feature relationships."""
    adjacency = {idx: set() for idx in df.index}
    if df.empty:
        return adjacency

    records = df.to_dict('index')
    sorted_indices = list(df.sort_values(['Average Rt(min)', 'Average Mz']).index)

    for pos, idx_i in enumerate(sorted_indices):
        row_i = records[idx_i]
        rt_i = safe_to_float(row_i['Average Rt(min)'])
        mz_i = safe_to_float(row_i['Average Mz'])
        adduct_i = str(row_i.get('Adduct type', '')).strip()
        peaks_i = row_i.get('_similarity_ms2', [])
        if math.isnan(rt_i) or math.isnan(mz_i):
            continue

        for idx_j in sorted_indices[pos + 1:]:
            row_j = records[idx_j]
            rt_j = safe_to_float(row_j['Average Rt(min)'])
            mz_j = safe_to_float(row_j['Average Mz'])
            adduct_j = str(row_j.get('Adduct type', '')).strip()
            peaks_j = row_j.get('_similarity_ms2', [])
            if math.isnan(rt_j) or math.isnan(mz_j):
                continue

            if rt_j - rt_i > ADDITIONAL_ISOTOPE_RT_TOL:
                break

            if adduct_i != adduct_j:
                continue
            if abs(rt_i - rt_j) > ADDITIONAL_ISOTOPE_RT_TOL:
                continue
            if peak_overlap_ratio(row_i, row_j) < ADDITIONAL_ISOTOPE_OVERLAP_RATIO:
                continue

            delta_mz = abs(mz_j - mz_i)
            gap_type = classify_isotope_gap(delta_mz)
            if gap_type is None:
                continue

            if ms1_isotope_support(row_i.get('MS1 isotopic spectrum'), row_j.get('MS1 isotopic spectrum'), mz_i, mz_j, tol=0.01):
                adjacency[idx_i].add(idx_j)
                adjacency[idx_j].add(idx_i)
                continue

            mod_cos, mod_matches = modified_cosine(peaks_i, peaks_j, mz_i, mz_j, ADDITIONAL_FRAGMENT_MZ_TOL)
            nl_cos, nl_matches = neutral_loss_cosine(peaks_i, peaks_j, mz_i, mz_j, ADDITIONAL_FRAGMENT_MZ_TOL)

            if gap_type == 'classic':
                if mod_cos >= ADDITIONAL_MODIFIED_COS_THRESHOLD and mod_matches >= ADDITIONAL_MIN_SHIFTED_MATCHES:
                    adjacency[idx_i].add(idx_j)
                    adjacency[idx_j].add(idx_i)
            else:
                if (
                    mod_cos >= ADDITIONAL_MODIFIED_COS_THRESHOLD_RELAXED
                    and mod_matches >= ADDITIONAL_MIN_SHIFTED_MATCHES_RELAXED
                    and nl_cos >= ADDITIONAL_NEUTRAL_LOSS_THRESHOLD
                    and nl_matches >= ADDITIONAL_MIN_NL_MATCHES
                ):
                    adjacency[idx_i].add(idx_j)
                    adjacency[idx_j].add(idx_i)

    return adjacency


def connected_components(adjacency):
    visited = set()
    components = []

    for node in adjacency:
        if node in visited:
            continue
        stack = [node]
        component = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for nei in adjacency[current]:
                if nei not in visited:
                    stack.append(nei)
        components.append(component)
    return components


def additional_isotope_deduplication(df):
    if df.empty:
        return df

    df = df.copy()
    df['_similarity_ms2'] = df.apply(
        lambda row: remove_precursor_region(
            row['_processed_msms_peaks'], safe_to_float(row['Average Mz']), ADDITIONAL_PRECURSOR_EXCLUSION
        ),
        axis=1,
    )

    adjacency = build_isotope_graph(df)
    components = connected_components(adjacency)

    keep_indices = []
    removed_count = 0
    for comp in components:
        if len(comp) == 1 and len(adjacency[comp[0]]) == 0:
            keep_indices.append(comp[0])
            continue

        comp_df = df.loc[comp].copy()
        comp_df['Average Mz'] = pd.to_numeric(comp_df['Average Mz'], errors='coerce')
        comp_df['_representative_intensity'] = pd.to_numeric(comp_df['_representative_intensity'], errors='coerce')

        # Keep the lowest precursor m/z; if tied, keep the highest representative intensity
        comp_df = comp_df.sort_values(['Average Mz', '_representative_intensity'], ascending=[True, False])
        representative = comp_df.index[0]
        keep_indices.append(representative)
        removed_count += (len(comp) - 1)

    result = df.loc[sorted(set(keep_indices))].copy()
    result.drop(columns=['_similarity_ms2'], inplace=True, errors='ignore')
    print(f"After additional isotope-like deduplication: {len(result)} (removed {removed_count})")
    return result


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("MS-DIAL Batch Table Integration and Deduplication v0.3.260320")

        self.debug_mode = tk.BooleanVar(value=False)
        self.additional_isotope_dedup_var = tk.BooleanVar(value=True)
        self.analysis_mode_var = tk.StringVar(value="segmented_batches")

        self.response_files = []
        self.snr_files = []
        self.merged_df = None
        self.blank_group_columns = None
        self.group_columns = None

        self.debug_checkbox = tk.Checkbutton(root, text="Debug Mode", variable=self.debug_mode)
        self.debug_checkbox.pack(pady=5)

        self.analysis_mode_label = tk.Label(root, text="Input Organization Mode:")
        self.analysis_mode_label.pack(pady=5)

        self.segmented_mode_radio = tk.Radiobutton(
            root,
            text="Segmented Batch Files (original mode)",
            variable=self.analysis_mode_var,
            value="segmented_batches",
            command=self.on_analysis_mode_changed,
        )
        self.segmented_mode_radio.pack(pady=2)

        self.total_batch_mode_radio = tk.Radiobutton(
            root,
            text="Combined Total Batch (left-shift non-zero values within each dataset block)",
            variable=self.analysis_mode_var,
            value="combined_total_batch",
            command=self.on_analysis_mode_changed,
        )
        self.total_batch_mode_radio.pack(pady=2)

        self.import_mode_hint_label = tk.Label(
            root,
            text="Original mode: import one response table + one S/N table for each ASFAM segment.",
            wraplength=760,
            justify='left',
        )
        self.import_mode_hint_label.pack(pady=5)

        self.total_batch_replicate_counts_label = tk.Label(
            root,
            text="Combined Total-Batch Actual Replicate Counts per Dataset Block (comma-separated; example: 3 or 3,3):"
        )
        self.total_batch_replicate_counts_label.pack(pady=5)
        self.total_batch_replicate_counts_entry = tk.Entry(root, width=60)
        self.total_batch_replicate_counts_entry.insert(0, "3")
        self.total_batch_replicate_counts_entry.pack(pady=5)

        self.total_batch_labels_label = tk.Label(
            root,
            text="Combined Total-Batch Dataset Labels (optional, comma-separated; example: Blank,Group1):"
        )
        self.total_batch_labels_label.pack(pady=5)
        self.total_batch_labels_entry = tk.Entry(root, width=60)
        self.total_batch_labels_entry.pack(pady=5)

        self.btn_import_response = tk.Button(root, text="Import Peak Height Tables", command=self.import_response)
        self.btn_import_response.pack(pady=5)

        self.btn_import_snr = tk.Button(root, text="Import S/N Tables", command=self.import_snr)
        self.btn_import_snr.pack(pady=5)

        self.btn_concat = tk.Button(root, text="Prepare Tables", command=self.concat_tables)
        self.btn_concat.pack(pady=5)

        self.btn_select_blank = tk.Button(root, text="Select Blank Group", command=self.select_blank_groups)
        self.btn_select_blank.pack(pady=5)

        self.btn_group_selection = tk.Button(root, text="Set Number of Sample Groups", command=self.group_selection)
        self.btn_group_selection.pack(pady=5)

        self.blank_threshold_label = tk.Label(root, text="Blank Fold Threshold (default: 5):")
        self.blank_threshold_label.pack(pady=5)
        self.blank_threshold_entry = tk.Entry(root)
        self.blank_threshold_entry.insert(0, "5")
        self.blank_threshold_entry.pack(pady=5)

        self.intensity_threshold_label = tk.Label(root, text="Minimum Peak Height Threshold (default: 1000):")
        self.intensity_threshold_label.pack(pady=5)
        self.intensity_threshold_entry = tk.Entry(root)
        self.intensity_threshold_entry.insert(0, "1000")
        self.intensity_threshold_entry.pack(pady=5)

        self.snr_threshold_label = tk.Label(root, text="Minimum S/N Threshold (default: 5):")
        self.snr_threshold_label.pack(pady=5)
        self.snr_threshold_entry = tk.Entry(root)
        self.snr_threshold_entry.insert(0, "5")
        self.snr_threshold_entry.pack(pady=5)

        self.mode_var = tk.StringVar(value="Pos_Mode")
        self.mode_label = tk.Label(root, text="Select Ionization Mode:")
        self.mode_label.pack(pady=5)
        self.mode_pos = tk.Radiobutton(root, text="Positive Ion Mode", variable=self.mode_var, value="Pos_Mode")
        self.mode_pos.pack(pady=5)
        self.mode_neg = tk.Radiobutton(root, text="Negative Ion Mode", variable=self.mode_var, value="Neg_Mode")
        self.mode_neg.pack(pady=5)

        self.mw_rt_window_label = tk.Label(root, text="RT Window for MW-Based Deduplication ± (min) (default: 0.05):")
        self.mw_rt_window_label.pack(pady=5)
        self.mw_rt_window_entry = tk.Entry(root)
        self.mw_rt_window_entry.insert(0, "0.05")
        self.mw_rt_window_entry.pack(pady=5)

        self.mw_diff_threshold_label = tk.Label(root, text="MW Difference Tolerance for Deduplication (default: 0.02):")
        self.mw_diff_threshold_label.pack(pady=5)
        self.mw_diff_threshold_entry = tk.Entry(root)
        self.mw_diff_threshold_entry.insert(0, "0.02")
        self.mw_diff_threshold_entry.pack(pady=5)

        self.ms2_intensity_threshold_label = tk.Label(root, text="Absolute MS/MS Intensity Threshold (default: 200):")
        self.ms2_intensity_threshold_label.pack(pady=5)
        self.ms2_intensity_threshold_entry = tk.Entry(root)
        self.ms2_intensity_threshold_entry.insert(0, "200")
        self.ms2_intensity_threshold_entry.pack(pady=5)

        self.ms2_response_percentage_label = tk.Label(root, text="Relative MS/MS Intensity Cutoff (% of base peak) (default: 5):")
        self.ms2_response_percentage_label.pack(pady=5)
        self.ms2_response_percentage_entry = tk.Entry(root)
        self.ms2_response_percentage_entry.insert(0, "5")
        self.ms2_response_percentage_entry.pack(pady=5)

        self.ms2_ion_threshold_label = tk.Label(root, text="Minimum Retained MS/MS Ion Count (default: 2):")
        self.ms2_ion_threshold_label.pack(pady=5)
        self.ms2_ion_threshold_entry = tk.Entry(root)
        self.ms2_ion_threshold_entry.insert(0, "2")
        self.ms2_ion_threshold_entry.pack(pady=5)

        self.additional_isotope_dedup_checkbox = tk.Checkbutton(
            root,
            text="Additional Isotope-Like Redundancy Removal",
            variable=self.additional_isotope_dedup_var
        )
        self.additional_isotope_dedup_checkbox.pack(pady=5)

        self.gaussian_similarity_threshold_label = tk.Label(root, text="Gaussian Similarity Threshold (default: 0.6):")
        self.gaussian_similarity_threshold_label.pack(pady=5)
        self.gaussian_similarity_threshold_entry = tk.Entry(root)
        self.gaussian_similarity_threshold_entry.insert(0, "0.6")
        self.gaussian_similarity_threshold_entry.pack(pady=5)

        self.process_data_button = tk.Button(self.root, text="Process Data", command=self.process_data)
        self.process_data_button.pack(pady=10)

        self.on_analysis_mode_changed()

    def clear_loaded_tables(self):
        self.response_files = []
        self.snr_files = []
        self.merged_df = None
        self.blank_group_columns = None
        self.group_columns = None

    def on_analysis_mode_changed(self):
        self.clear_loaded_tables()
        if self.analysis_mode_var.get() == "segmented_batches":
            self.import_mode_hint_label.config(
                text="Original mode: import one response table + one S/N table for each ASFAM segment, then the tool merges each pair and concatenates all segment tables vertically."
            )
            self.btn_import_response.config(text="Import Peak Height Tables")
            self.btn_import_snr.config(text="Import S/N Tables")
        else:
            self.import_mode_hint_label.config(
                text=(
                    "Combined total-batch mode: import one MS-DIAL response table and one MS-DIAL S/N table generated from a single total batch. "
                    "The tool will compress each dataset block by moving all non-zero values left within that block, keep only the actual replicate count, and then continue with the same downstream workflow."
                )
            )
            self.btn_import_response.config(text="Import Total-Batch Peak Height Table")
            self.btn_import_snr.config(text="Import Total-Batch S/N Table")

    def import_response(self):
        selected_files = filedialog.askopenfilenames(
            title=self.btn_import_response.cget('text'),
            filetypes=[("MS-DIAL tables", "*.txt *.csv"), ("TXT Files", "*.txt"), ("CSV Files", "*.csv")]
        )
        if not selected_files:
            return

        if self.analysis_mode_var.get() == "combined_total_batch" and len(selected_files) != 1:
            messagebox.showerror("Error", "Combined total-batch mode accepts exactly one peak height table per run.")
            return

        self.clear_loaded_tables()
        self.response_files = list(selected_files)
        messagebox.showinfo("Success", f"Successfully imported {len(self.response_files)} peak height table(s)")

    def import_snr(self):
        selected_files = filedialog.askopenfilenames(
            title=self.btn_import_snr.cget('text'),
            filetypes=[("MS-DIAL tables", "*.txt *.csv"), ("TXT Files", "*.txt"), ("CSV Files", "*.csv")]
        )
        if not selected_files:
            return

        if self.analysis_mode_var.get() == "combined_total_batch" and len(selected_files) != 1:
            messagebox.showerror("Error", "Combined total-batch mode accepts exactly one S/N table per run.")
            return

        self.snr_files = list(selected_files)
        if self.response_files and len(self.snr_files) != len(self.response_files):
            messagebox.showerror("Error", "The numbers of peak height tables and S/N tables do not match!")
        else:
            messagebox.showinfo("Success", f"Successfully imported {len(self.snr_files)} S/N table(s)")

    def prepare_segmented_tables(self):
        merged_tables = []

        for idx, (resp_file, snr_file) in enumerate(zip(self.response_files, self.snr_files), start=1):
            resp_df_raw = read_msdial_alignment_table(resp_file)
            snr_df = read_msdial_alignment_table(snr_file)

            response_sample_columns = get_true_sample_columns(resp_file, resp_df_raw)
            snr_sample_columns = get_true_sample_columns(snr_file, snr_df)

            metadata_end_idx = resp_df_raw.columns.get_loc('MS/MS spectrum') + 1
            response_keep_columns = list(resp_df_raw.columns[:metadata_end_idx]) + response_sample_columns
            resp_df = resp_df_raw[response_keep_columns].copy()

            merged_df = merge_response_and_snr_tables(resp_df, snr_df, response_sample_columns, snr_sample_columns)
            merged_df.reset_index(drop=True, inplace=True)
            print(f"Peak height table and S/N table {idx} were merged successfully")

            if self.debug_mode.get():
                merged_df.to_csv(f"intensity_SNR_{idx}.csv", index=False)

            merged_tables.append(merged_df)

        first_table_columns = merged_tables[0].columns
        merged_df = merged_tables[0]
        for df in merged_tables[1:]:
            df.columns = first_table_columns
            df.reset_index(drop=True, inplace=True)
            merged_df = pd.concat([merged_df, df], ignore_index=True)

        print("All segmented tables were concatenated successfully")
        merged_df.sort_values(by="Average Rt(min)", inplace=True)
        return merged_df

    def prepare_combined_total_batch_table(self):
        if len(self.response_files) != 1 or len(self.snr_files) != 1:
            raise ValueError("Combined total-batch mode requires exactly one response table and one S/N table.")

        replicate_counts = parse_positive_int_list(
            self.total_batch_replicate_counts_entry.get(),
            "Combined total-batch actual replicate counts"
        )
        dataset_labels = parse_optional_label_list(
            self.total_batch_labels_entry.get(),
            len(replicate_counts)
        )

        resp_file = self.response_files[0]
        snr_file = self.snr_files[0]

        resp_df_raw = read_msdial_alignment_table(resp_file)
        snr_df_raw = read_msdial_alignment_table(snr_file)

        resp_df, response_sample_columns = compress_total_batch_table(
            resp_file, resp_df_raw, replicate_counts, dataset_labels
        )
        snr_df, snr_sample_columns = compress_total_batch_table(
            snr_file, snr_df_raw, replicate_counts, dataset_labels
        )

        merged_df = merge_response_and_snr_tables(resp_df, snr_df, response_sample_columns, snr_sample_columns)
        merged_df.reset_index(drop=True, inplace=True)
        print("Combined total-batch table was compressed and merged successfully")

        if self.debug_mode.get():
            resp_df.to_csv("combined_total_batch_response_compressed.csv", index=False)
            snr_df.to_csv("combined_total_batch_snr_compressed.csv", index=False)
            merged_df.to_csv("combined_total_batch_merged.csv", index=False)

        merged_df.sort_values(by="Average Rt(min)", inplace=True)
        return merged_df

    def concat_tables(self):
        if not self.response_files or not self.snr_files:
            messagebox.showerror("Error", "Please import peak height tables and S/N tables first!")
            return
        if len(self.response_files) != len(self.snr_files):
            messagebox.showerror("Error", "The numbers of peak height tables and S/N tables do not match!")
            return

        try:
            if self.analysis_mode_var.get() == "segmented_batches":
                merged_df = self.prepare_segmented_tables()
            else:
                merged_df = self.prepare_combined_total_batch_table()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        if self.debug_mode.get():
            merged_df.to_csv("merge_df.csv", index=False)

        self.merged_df = merged_df
        messagebox.showinfo("Success", "Table preparation completed successfully")

    def select_blank_groups(self):
        if self.merged_df is None:
            messagebox.showerror("Error", "Please prepare the tables first!")
            return

        columns = list(self.merged_df.columns)
        blank_window = Toplevel(self.root)
        blank_window.title("Select Blank Group")

        lbl_response = tk.Label(blank_window, text="Blank-Group Peak Height Columns")
        lbl_response.grid(row=0, column=0, padx=10, pady=10)
        lbl_snr = tk.Label(blank_window, text="Blank-Group S/N Columns")
        lbl_snr.grid(row=0, column=1, padx=10, pady=10)

        self.blank_response_listbox = Listbox(blank_window, selectmode=MULTIPLE, height=10, exportselection=False)
        self.blank_response_listbox.grid(row=1, column=0, padx=10, pady=10)

        self.blank_snr_listbox = Listbox(blank_window, selectmode=MULTIPLE, height=10, exportselection=False)
        self.blank_snr_listbox.grid(row=1, column=1, padx=10, pady=10)

        for col in columns:
            self.blank_response_listbox.insert(tk.END, col)
            self.blank_snr_listbox.insert(tk.END, col)

        submit_btn = tk.Button(blank_window, text="Confirm Selection", command=self.save_blank_group_selection)
        submit_btn.grid(row=2, column=0, columnspan=2, pady=10)

    def save_blank_group_selection(self):
        blank_response_indices = self.blank_response_listbox.curselection()
        blank_snr_indices = self.blank_snr_listbox.curselection()

        blank_response_columns = [self.blank_response_listbox.get(i) for i in blank_response_indices]
        blank_snr_columns = [self.blank_snr_listbox.get(i) for i in blank_snr_indices]

        if not blank_response_columns and not blank_snr_columns:
            messagebox.showerror("Error", "Please select at least one peak-height column or one S/N column")
            return

        print(f"Blank-group peak height columns: {blank_response_columns}")
        print(f"Blank-group S/N columns: {blank_snr_columns}")

        self.blank_group_columns = {
            "response": blank_response_columns,
            "snr": blank_snr_columns
        }

        messagebox.showinfo("Success", "Blank-group selection completed successfully")

    def group_selection(self):
        if self.merged_df is None:
            messagebox.showerror("Error", "Please prepare the tables first!")
            return

        group_count = simpledialog.askinteger("Set Number of Sample Groups", "Please enter the number of sample groups:")
        if group_count is None or group_count <= 0:
            messagebox.showerror("Error", "Please enter a valid number of sample groups!")
            return

        group_window = Toplevel(self.root)
        group_window.title(f"Select Columns for {group_count} Group(s)")

        self.group_response_listboxes = []
        self.group_snr_listboxes = []

        for i in range(group_count):
            lbl_response = tk.Label(group_window, text=f"Group {i + 1} Peak Height Columns")
            lbl_response.grid(row=0, column=2 * i, padx=10, pady=10)
            lbl_snr = tk.Label(group_window, text=f"Group {i + 1} S/N Columns")
            lbl_snr.grid(row=0, column=2 * i + 1, padx=10, pady=10)

            response_listbox = Listbox(group_window, selectmode=MULTIPLE, height=10, exportselection=False)
            response_listbox.grid(row=1, column=2 * i, padx=10, pady=10)
            self.group_response_listboxes.append(response_listbox)

            snr_listbox = Listbox(group_window, selectmode=MULTIPLE, height=10, exportselection=False)
            snr_listbox.grid(row=1, column=2 * i + 1, padx=10, pady=10)
            self.group_snr_listboxes.append(snr_listbox)

            for col in self.merged_df.columns:
                response_listbox.insert(tk.END, col)
                snr_listbox.insert(tk.END, col)

        submit_btn = tk.Button(group_window, text="Confirm Selection", command=self.save_group_selection)
        submit_btn.grid(row=2, column=0, columnspan=2 * group_count, pady=10)

    def save_group_selection(self):
        group_responses = []
        group_snrs = []

        for response_listbox, snr_listbox in zip(self.group_response_listboxes, self.group_snr_listboxes):
            response_indices = response_listbox.curselection()
            snr_indices = snr_listbox.curselection()

            response_columns = [response_listbox.get(i) for i in response_indices]
            snr_columns = [snr_listbox.get(i) for i in snr_indices]

            if response_columns and snr_columns:
                group_responses.append(response_columns)
                group_snrs.append(snr_columns)
            else:
                messagebox.showerror("Error", "Please select at least one peak-height column and one S/N column for every group")
                return

        print(f"Group peak height columns: {group_responses}")
        print(f"Group S/N columns: {group_snrs}")

        self.group_columns = {
            "response": group_responses,
            "snr": group_snrs
        }

        messagebox.showinfo("Success", "Group selection completed successfully")

    def get_thresholds(self):
        blank_threshold = float(self.blank_threshold_entry.get())
        intensity_threshold = float(self.intensity_threshold_entry.get())
        snr_threshold = float(self.snr_threshold_entry.get())
        rt_window = float(self.mw_rt_window_entry.get())
        mw_diff_threshold = float(self.mw_diff_threshold_entry.get())
        ms2_intensity_threshold = float(self.ms2_intensity_threshold_entry.get())
        ms2_response_percentage = float(self.ms2_response_percentage_entry.get()) / 100
        ms2_ion_threshold = int(self.ms2_ion_threshold_entry.get())
        gaussian_similarity_threshold = float(self.gaussian_similarity_threshold_entry.get())

        return {
            "blank_threshold": blank_threshold,
            "intensity_threshold": intensity_threshold,
            "snr_threshold": snr_threshold,
            "rt_window": rt_window,
            "mw_diff_threshold": mw_diff_threshold,
            "ms2_intensity_threshold": ms2_intensity_threshold,
            "ms2_response_percentage": ms2_response_percentage,
            "ms2_ion_threshold": ms2_ion_threshold,
            "gaussian_similarity_threshold": gaussian_similarity_threshold,
        }

    def process_data(self):
        if self.merged_df is None:
            messagebox.showerror("Error", "Please prepare the tables first!")
            return
        if self.group_columns is None:
            messagebox.showerror("Error", "Please define the sample groups first!")
            return

        thresholds = self.get_thresholds()
        mode = self.mode_var.get()
        df = self.merged_df.copy()

        print(f"Initial table contains {len(df)} rows")

        numeric_columns = [
            'Average Rt(min)', 'Average Mz', 'Isotope tracking weight number',
            'Gaussian similarity', 'RT left(min)', 'RT right (min)'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if self.blank_group_columns:
            df['blank_ave_intensity'] = df[self.blank_group_columns['response']].mean(axis=1)
            df['blank_ave_SNR'] = df[self.blank_group_columns['snr']].mean(axis=1)

        for i, (response_cols, snr_cols) in enumerate(zip(self.group_columns['response'], self.group_columns['snr']), start=1):
            df[f'Group_{i}_ave_intensity'] = df[response_cols].mean(axis=1)
            df[f'Group_{i}_ave_SNR'] = df[snr_cols].mean(axis=1)

        group_intensity_cols = [f'Group_{i}_ave_intensity' for i in range(1, len(self.group_columns['response']) + 1)]
        df['_representative_intensity'] = df[group_intensity_cols].max(axis=1)

        if 'blank_ave_intensity' in df.columns:
            def check_blank_threshold(row):
                blank_value = row['blank_ave_intensity']
                if pd.isna(blank_value) or blank_value <= 0:
                    blank_value = 0.1
                for i in range(1, len(self.group_columns['response']) + 1):
                    factor = row[f'Group_{i}_ave_intensity'] / blank_value
                    if factor >= thresholds['blank_threshold']:
                        return False
                return True

            df = df[~df.apply(check_blank_threshold, axis=1)].copy()
            print(f"{len(df)} rows remain after blank filtering")

        def check_intensity_threshold(row):
            for i in range(1, len(self.group_columns['response']) + 1):
                if row[f'Group_{i}_ave_intensity'] >= thresholds['intensity_threshold']:
                    return False
            return True

        df = df[~df.apply(check_intensity_threshold, axis=1)].copy()
        print(f"{len(df)} rows remain after minimum peak height filtering")

        def check_snr_threshold(row):
            for i in range(1, len(self.group_columns['snr']) + 1):
                if row[f'Group_{i}_ave_SNR'] >= thresholds['snr_threshold']:
                    return False
            return True

        df = df[~df.apply(check_snr_threshold, axis=1)].copy()
        print(f"{len(df)} rows remain after minimum S/N filtering")

        if 'Isotope tracking weight number' in df.columns:
            df = df[df['Isotope tracking weight number'] == 0].copy()
            print(f"{len(df)} rows remain after removing rows with non-zero 'Isotope tracking weight number'")

        df['MW'] = df.apply(lambda row: calculate_mw(row, mode), axis=1)
        df = df[df['MW'] != 0].copy()
        print(f"{len(df)} rows remain after MW calculation and removing rows with MW = 0")

        rt_window = thresholds['rt_window']
        mw_diff_threshold = thresholds['mw_diff_threshold']
        to_delete_indices = set()
        df = df.sort_values(['Average Rt(min)', 'MW', '_representative_intensity'], ascending=[True, True, False]).copy()
        idx_list = list(df.index)

        for pos, idx in enumerate(idx_list):
            if idx in to_delete_indices:
                continue

            target_row = df.loc[idx]
            target_rt = safe_to_float(target_row['Average Rt(min)'])
            target_mw = safe_to_float(target_row['MW'])
            if math.isnan(target_rt) or math.isnan(target_mw):
                continue

            candidate_indices = []
            for idx2 in idx_list[pos + 1:]:
                if idx2 in to_delete_indices:
                    continue
                row2 = df.loc[idx2]
                rt2 = safe_to_float(row2['Average Rt(min)'])
                mw2 = safe_to_float(row2['MW'])
                if math.isnan(rt2) or math.isnan(mw2):
                    continue
                if rt2 - target_rt > rt_window:
                    break
                if abs(rt2 - target_rt) <= rt_window and abs(mw2 - target_mw) <= mw_diff_threshold:
                    candidate_indices.append(idx2)

            if candidate_indices:
                all_group_indices = [idx] + candidate_indices
                group_df = df.loc[all_group_indices]
                keep_idx = group_df['_representative_intensity'].idxmax()
                for rm_idx in group_df.index:
                    if rm_idx != keep_idx:
                        to_delete_indices.add(rm_idx)

        df = df.drop(index=list(to_delete_indices)).copy()
        print(f"{len(df)} rows remain after MW-based deduplication")

        parsed_peaks = df['MS/MS spectrum'].apply(parse_msms_spectrum)
        processed_peaks = parsed_peaks.apply(
            lambda peaks: clean_msms_peaks(
                peaks,
                thresholds['ms2_intensity_threshold'],
                thresholds['ms2_response_percentage']
            )
        )
        df['_processed_msms_peaks'] = processed_peaks
        df['MS/MS spectrum'] = processed_peaks.apply(spectrum_to_string)
        print("MS/MS spectrum processing completed")

        if self.additional_isotope_dedup_var.get():
            df = additional_isotope_deduplication(df)

        if 'Gaussian similarity' in df.columns:
            df = df[df['Gaussian similarity'] >= thresholds['gaussian_similarity_threshold']].copy()
            print(f"{len(df)} rows remain after Gaussian similarity filtering")

        output_dir = os.path.dirname(self.response_files[0]) if self.response_files else os.getcwd()
        output_file = os.path.join(output_dir, 'final_processed_data.csv')

        df.drop(columns=['_processed_msms_peaks', '_representative_intensity'], inplace=True, errors='ignore')
        df.to_csv(output_file, index=False)

        self.merged_df = df
        print(f"Data processing completed. Results were saved as {output_file}")
        messagebox.showinfo("Success", f"Data processing completed. Results were saved as:\n{output_file}")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
