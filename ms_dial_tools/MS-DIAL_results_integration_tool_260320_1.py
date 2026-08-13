import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox

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


def parse_msms_spectrum(spectrum_str):
    """Parse MS-DIAL text format: 'mz:intensity mz:intensity ...'."""
    if pd.isna(spectrum_str) or not str(spectrum_str).strip():
        return []

    peaks = []
    for token in str(spectrum_str).strip().split():
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
    left_a = safe_to_float(row_a.get('RT left(min)'))
    right_a = safe_to_float(row_a.get('RT right (min)'))
    left_b = safe_to_float(row_b.get('RT left(min)'))
    right_b = safe_to_float(row_b.get('RT right (min)'))

    if any(math.isnan(v) for v in [left_a, right_a, left_b, right_b]):
        return 0.0

    width_a = max(0.0, right_a - left_a)
    width_b = max(0.0, right_b - left_b)
    if width_a == 0 or width_b == 0:
        return 0.0

    overlap = max(0.0, min(right_a, right_b) - max(left_a, left_b))
    return overlap / min(width_a, width_b)


def calculate_mw(row, mode):
    precursor_mz = safe_to_float(row['Precursor m/z'])
    adduct = str(row.get('Adduct', '')).strip()

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
    adjacency = {idx: set() for idx in df.index}
    if df.empty:
        return adjacency

    records = df.to_dict('index')
    sorted_indices = list(df.sort_values(['RT (min)', 'Precursor m/z']).index)

    for pos, idx_i in enumerate(sorted_indices):
        row_i = records[idx_i]
        rt_i = safe_to_float(row_i['RT (min)'])
        mz_i = safe_to_float(row_i['Precursor m/z'])
        adduct_i = str(row_i.get('Adduct', '')).strip()
        peaks_i = row_i.get('_similarity_ms2', [])
        if math.isnan(rt_i) or math.isnan(mz_i):
            continue

        for idx_j in sorted_indices[pos + 1:]:
            row_j = records[idx_j]
            rt_j = safe_to_float(row_j['RT (min)'])
            mz_j = safe_to_float(row_j['Precursor m/z'])
            adduct_j = str(row_j.get('Adduct', '')).strip()
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

            if ms1_isotope_support(row_i.get('MS1 isotopes'), row_j.get('MS1 isotopes'), mz_i, mz_j, tol=0.01):
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

    df['_similarity_ms2'] = df.apply(
        lambda row: remove_precursor_region(
            row['_processed_msms_peaks'], safe_to_float(row['Precursor m/z']), ADDITIONAL_PRECURSOR_EXCLUSION
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
        comp_df['Precursor m/z'] = pd.to_numeric(comp_df['Precursor m/z'], errors='coerce')
        comp_df['Height'] = pd.to_numeric(comp_df['Height'], errors='coerce')

        comp_df = comp_df.sort_values(['Precursor m/z', 'Height'], ascending=[True, False])
        representative = comp_df.index[0]
        keep_indices.append(representative)
        removed_count += (len(comp) - 1)

    result = df.loc[sorted(set(keep_indices))].copy()
    result.drop(columns=['_similarity_ms2'], inplace=True, errors='ignore')
    print(f"After additional isotope-like deduplication: {len(result)} (removed {removed_count})")
    return result


# =========================
# Main processing function
# =========================

def process_files(
    file_paths,
    mode,
    output_dir,
    remove_isotopes,
    remove_different_adducts,
    process_secondary_data,
    height_threshold,
    sn_threshold,
    intensity_threshold,
    response_percentage,
    ion_threshold,
    rt_window,
    mw_diff_threshold,
    gaussian_similarity_threshold,
    additional_isotope_dedup=True,
):
    all_dfs = []

    for file in file_paths:
        df = pd.read_csv(file, sep='\t')
        df['ID'] = df.apply(lambda row: f"{os.path.basename(file).replace('.txt', '')}_{row['Peak ID']}", axis=1)
        all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True)

    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    print(f"Initial merged rows: {len(merged_df)}")

    numeric_columns = [
        'Isotope', 'Height', 'S/N', 'RT (min)', 'Precursor m/z',
        'RT left(min)', 'RT right (min)', 'Gaussian similarity'
    ]
    for col in numeric_columns:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

    # Option 1: Remove isotopic features identified by MS-DIAL
    if remove_isotopes and 'Isotope' in merged_df.columns:
        merged_df = merged_df[merged_df['Isotope'] == 0].copy()
        print(f"After filtering Isotope != 0: {len(merged_df)}")

    # Option 2: Simple signal filtering aligned with the batch version
    merged_df = merged_df[
        (merged_df['Height'] >= height_threshold) &
        (merged_df['S/N'] >= sn_threshold)
    ].copy()
    print(f"After simple Height and S/N filtering: {len(merged_df)}")

    # Option 3: Remove redundant features derived from different adduct forms
    if remove_different_adducts:
        merged_df['MW'] = merged_df.apply(lambda row: calculate_mw(row, mode), axis=1)
        merged_df = merged_df[merged_df['MW'] != 0].copy()
        print(f"After filtering MW == 0: {len(merged_df)}")

        to_delete_indices = set()
        merged_df = merged_df.sort_values(['RT (min)', 'MW', 'Height'], ascending=[True, True, False]).copy()
        idx_list = list(merged_df.index)

        for pos, idx in enumerate(idx_list):
            if idx in to_delete_indices:
                continue
            target_row = merged_df.loc[idx]
            target_rt = safe_to_float(target_row['RT (min)'])
            target_mw = safe_to_float(target_row['MW'])
            if math.isnan(target_rt) or math.isnan(target_mw):
                continue

            candidate_indices = []
            for idx2 in idx_list[pos + 1:]:
                if idx2 in to_delete_indices:
                    continue
                row2 = merged_df.loc[idx2]
                rt2 = safe_to_float(row2['RT (min)'])
                mw2 = safe_to_float(row2['MW'])
                if math.isnan(rt2) or math.isnan(mw2):
                    continue
                if rt2 - target_rt > rt_window:
                    break
                if abs(rt2 - target_rt) <= rt_window and abs(mw2 - target_mw) <= mw_diff_threshold:
                    candidate_indices.append(idx2)

            if candidate_indices:
                all_group_indices = [idx] + candidate_indices
                group_df = merged_df.loc[all_group_indices]
                keep_idx = group_df['Height'].idxmax()
                for rm_idx in group_df.index:
                    if rm_idx != keep_idx:
                        to_delete_indices.add(rm_idx)

        merged_df = merged_df.drop(index=list(to_delete_indices)).copy()
        print(f"After RT and MW filtering: {len(merged_df)}")

    # Option 4: Process MS/MS spectral data
    if process_secondary_data:
        parsed_peaks = merged_df['MSMS spectrum'].apply(parse_msms_spectrum)
        processed_peaks = parsed_peaks.apply(
            lambda peaks: clean_msms_peaks(peaks, intensity_threshold, response_percentage)
        )

        merged_df['_processed_msms_peaks'] = processed_peaks
        merged_df['MSMS spectrum'] = processed_peaks.apply(spectrum_to_string)
        merged_df = merged_df[merged_df['_processed_msms_peaks'].apply(len) > ion_threshold].copy()
        print(f"After MS/MS filtering: {len(merged_df)}")
    else:
        merged_df['_processed_msms_peaks'] = [[] for _ in range(len(merged_df))]

    # Option 5: Additional isotope-like redundancy removal after MS-DIAL isotope/adduct cleanup
    if additional_isotope_dedup and process_secondary_data:
        merged_df = additional_isotope_deduplication(merged_df)

    # Option 6: Gaussian similarity filter
    if 'Gaussian similarity' in merged_df.columns:
        merged_df = merged_df[merged_df['Gaussian similarity'] >= gaussian_similarity_threshold].copy()
        print(f"After Gaussian similarity filtering: {len(merged_df)}")

    merged_df.drop(columns=['_processed_msms_peaks'], inplace=True, errors='ignore')

    output_file = os.path.join(output_dir, 'merged_output.csv')
    merged_df.to_csv(output_file, index=False)

    return output_file


# =========================
# GUI
# =========================

def main():
    root = tk.Tk()
    root.title("MS-DIAL Result Integration and Deduplication Tool v0.3.260320")

    global selected_files, mode
    selected_files = []
    mode = tk.StringVar(value='Pos_Mode')

    remove_isotopes = tk.BooleanVar(value=True)
    remove_different_adducts = tk.BooleanVar(value=True)
    process_secondary_data = tk.BooleanVar(value=True)
    additional_isotope_dedup_var = tk.BooleanVar(value=True)

    # Input variables
    height_threshold_var = tk.DoubleVar(value=1000.0)
    sn_threshold_var = tk.DoubleVar(value=5.0)
    intensity_threshold_var = tk.IntVar(value=200)
    response_percentage_var = tk.DoubleVar(value=0.05)
    ion_threshold_var = tk.IntVar(value=1)
    rt_window_var = tk.DoubleVar(value=0.05)
    mw_diff_threshold_var = tk.DoubleVar(value=0.02)
    gaussian_similarity_threshold_var = tk.DoubleVar(value=0.6)

    def select_files():
        global selected_files
        selected_files = filedialog.askopenfilenames(filetypes=[("TXT files", "*.txt")])
        if selected_files:
            messagebox.showinfo("File Selection", f"{len(selected_files)} file(s) selected.")

    def run_process():
        if not selected_files:
            messagebox.showwarning("Warning", "Please select TXT files first.")
            return

        output_dir = os.path.dirname(selected_files[0])
        try:
            result_file = process_files(
                selected_files,
                mode.get(),
                output_dir,
                remove_isotopes.get(),
                remove_different_adducts.get(),
                process_secondary_data.get(),
                height_threshold_var.get(),
                sn_threshold_var.get(),
                intensity_threshold_var.get(),
                response_percentage_var.get(),
                ion_threshold_var.get(),
                rt_window_var.get(),
                mw_diff_threshold_var.get(),
                gaussian_similarity_threshold_var.get(),
                additional_isotope_dedup_var.get(),
            )
            messagebox.showinfo("Completed", f"Processing completed.\nResults saved to:\n{result_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed:\n{e}")

    # Create interface widgets
    import_button = tk.Button(root, text="Import TXT Files", command=select_files)
    import_button.pack(pady=20)

    mode_label = tk.Label(root, text="Ionization Mode:")
    mode_label.pack()

    mode_option_menu = tk.OptionMenu(root, mode, 'Pos_Mode', 'Neg_Mode')
    mode_option_menu.pack(pady=10)

    remove_isotopes_check = tk.Checkbutton(root, text="Remove MS-DIAL-Flagged Isotopic Features", variable=remove_isotopes)
    remove_isotopes_check.pack(pady=5)

    remove_different_adducts_check = tk.Checkbutton(
        root,
        text="Remove Redundant Features from Different Adduct Forms",
        variable=remove_different_adducts
    )
    remove_different_adducts_check.pack(pady=5)

    process_secondary_data_check = tk.Checkbutton(
        root,
        text="Process MS/MS Spectra",
        variable=process_secondary_data
    )
    process_secondary_data_check.pack(pady=5)

    additional_isotope_dedup_check = tk.Checkbutton(
        root,
        text="Additional Isotope-Like Redundancy Removal",
        variable=additional_isotope_dedup_var
    )
    additional_isotope_dedup_check.pack(pady=5)

    height_threshold_label = tk.Label(root, text="Minimum Peak Height Threshold:")
    height_threshold_label.pack()
    height_threshold_entry = tk.Entry(root, textvariable=height_threshold_var)
    height_threshold_entry.pack()

    sn_threshold_label = tk.Label(root, text="Minimum S/N Threshold:")
    sn_threshold_label.pack()
    sn_threshold_entry = tk.Entry(root, textvariable=sn_threshold_var)
    sn_threshold_entry.pack()

    intensity_label = tk.Label(root, text="MS/MS Intensity Threshold:")
    intensity_label.pack()
    intensity_entry = tk.Entry(root, textvariable=intensity_threshold_var)
    intensity_entry.pack()

    response_percentage_label = tk.Label(root, text="Relative MS/MS Intensity Threshold (Base Peak Fraction):")
    response_percentage_label.pack()
    response_percentage_entry = tk.Entry(root, textvariable=response_percentage_var)
    response_percentage_entry.pack()

    ion_threshold_label = tk.Label(root, text="Minimum Number of MS/MS Fragment Ions:")
    ion_threshold_label.pack()
    ion_threshold_entry = tk.Entry(root, textvariable=ion_threshold_var)
    ion_threshold_entry.pack()

    rt_window_label = tk.Label(root, text="RT Window for Adduct Deduplication (min):")
    rt_window_label.pack()
    rt_window_entry = tk.Entry(root, textvariable=rt_window_var)
    rt_window_entry.pack()

    mw_diff_threshold_label = tk.Label(root, text="MW Difference Threshold for Adduct Deduplication:")
    mw_diff_threshold_label.pack()
    mw_diff_threshold_entry = tk.Entry(root, textvariable=mw_diff_threshold_var)
    mw_diff_threshold_entry.pack()

    gaussian_similarity_threshold_label = tk.Label(root, text="Gaussian Similarity Threshold:")
    gaussian_similarity_threshold_label.pack()
    gaussian_similarity_threshold_entry = tk.Entry(root, textvariable=gaussian_similarity_threshold_var)
    gaussian_similarity_threshold_entry.pack()

    run_button = tk.Button(root, text="Run", command=run_process)
    run_button.pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    main()