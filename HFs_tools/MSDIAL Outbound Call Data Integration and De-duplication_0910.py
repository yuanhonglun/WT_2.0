import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# Merge all txt files and perform relevant processing
def process_files(file_paths, mode, output_dir, remove_isotopes, remove_different_adducts, process_secondary_data, sn_threshold_high, sn_threshold_low, height_threshold, height_min_threshold, intensity_threshold, response_percentage, ion_threshold, rt_window, mw_diff_threshold):
    all_dfs = []
    
    for file in file_paths:
        df = pd.read_csv(file, sep='\t')
        df['ID'] = df.apply(lambda row: f"{os.path.basename(file).replace('.txt', '')}_{row['Peak ID']}", axis=1)
        all_dfs.append(df)
    
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Keep only one header row
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    print(f"Initial merged rows: {len(merged_df)}")

    # Option 1: Remove isotopes
    if remove_isotopes:
        merged_df = merged_df[merged_df['Isotope'] == 0]
        print(f"After filtering Isotope != 0: {len(merged_df)}")

    # Retain the user-defined S/N and Height thresholds. Use different S/N thresholds based on the different Heights.
    def custom_filter(row):
        if row['Height'] < height_min_threshold:
            return False # If the height is less than the minimum height threshold, it will not be retained.
        elif row['Height'] < height_threshold:
            return row['S/N'] >= sn_threshold_high  # When the height is less than the set height_threshold, use the high S/N threshold
        else:
            return row['S/N'] >= sn_threshold_low  # When the height is greater than or equal to the set height_threshold, use the low S/N threshold

    merged_df = merged_df[merged_df.apply(custom_filter, axis=1)]
    print(f"After filtering with custom thresholds: {len(merged_df)}")

    # Option 2: Eliminate different loading methods
    if remove_different_adducts:
        merged_df['Precursor m/z'] = pd.to_numeric(merged_df['Precursor m/z'], errors='coerce')
        
        # Calculate the MW column based on the Adduct column and the Precursor m/z value
        def calculate_mw(row, mode):
            precursor_mz = row['Precursor m/z']
            adduct = row['Adduct']
            if mode == 'Pos_Mode':
                if adduct == '[M+H]+':
                    return precursor_mz - 1.0072766
                elif adduct == '[M+H -H2O]+':
                    return precursor_mz + 17.0021912
                elif adduct == '[M+NH4]+':
                    return precursor_mz - 18.0338257
                elif adduct == '[M+Na]+':
                    return precursor_mz - 22.9892213
                elif adduct == '[M+K]+':
                    return precursor_mz - 38.9631585
                elif adduct == '[M]+':
                    return precursor_mz
                else:
                    return 0
            elif mode == 'Neg_Mode':
                if adduct == '[M-2H]2-':
                    return precursor_mz * 2 + 2.0145532
                elif adduct == '[M-H]-':
                    return precursor_mz + 1.0072766
                elif adduct == '[M-H2O-H]-':
                    return precursor_mz + 19.0178413
                else:
                    return 0

        # Calculate the MW column
        merged_df['MW'] = merged_df.apply(lambda row: calculate_mw(row, mode), axis=1)
        
        # Remove rows in the MW column where the value is 0
        merged_df = merged_df[merged_df['MW'] != 0]
        print(f"After filtering MW == 0: {len(merged_df)}")
        
        # Remove features that are consistent with MW
        to_delete_indices = []
        for idx in range(len(merged_df)):
            if idx in to_delete_indices:
                continue
            target_row = merged_df.iloc[idx]
            rt_min = max(0, target_row['RT (min)'] - rt_window)
            rt_max = target_row['RT (min)'] + rt_window
            same_rt_rows = merged_df[(merged_df['RT (min)'] >= rt_min) & (merged_df['RT (min)'] <= rt_max)]
            
            if not same_rt_rows.empty:
                same_rt_rows = same_rt_rows[abs(same_rt_rows['MW'] - target_row['MW']) <= mw_diff_threshold]
                if len(same_rt_rows) > 1:
                    max_height_idx = same_rt_rows['Height'].idxmax()
                    to_delete_indices.extend(same_rt_rows.index.difference([max_height_idx]))
        
        merged_df = merged_df.drop(index=to_delete_indices)
        print(f"After RT and MW filtering: {len(merged_df)}")
        
    # Option 3: Handling secondary data
    if process_secondary_data:
        def process_msms_spectrum(row):
            if pd.isnull(row['MSMS spectrum']):
                return ''
            spectra = row['MSMS spectrum'].split(';')
            processed_spectra = []
            for s in spectra:
                parts = s.split()
                if len(parts) == 2:
                    try:
                        intensity = int(parts[1])
                        if intensity >= intensity_threshold:
                            processed_spectra.append(s)
                    except ValueError:
                        continue
            
            if not processed_spectra:
                return ''
            
            max_response = max([int(s.split()[1]) for s in processed_spectra])
            threshold = max_response * response_percentage
            final_spectra = [s for s in processed_spectra if int(s.split()[1]) >= threshold]
            
            # Check the quantity of mz-intensity pairs. If it is less than or equal to ion_threshold, then clear the data
            if len(final_spectra) <= ion_threshold:
                return ''
            
            final_spectra = sorted(final_spectra, key=lambda x: int(x.split()[1]), reverse=True)
            return ';'.join(final_spectra)
        
        merged_df['MSMS spectrum'] = merged_df.apply(process_msms_spectrum, axis=1)
    
    # Export the merged table
    output_file = os.path.join(output_dir, 'merged_output.csv')
    merged_df.to_csv(output_file, index=False)
    
    return output_file

# Visual Interface
def main():
    root = tk.Tk()
    root.title("MS-DIAL Data Integration and De-duplication Tool v0.1.240910")

    global selected_files, mode
    selected_files = []
    mode = tk.StringVar(value='Pos_Mode')

    remove_isotopes = tk.BooleanVar(value=True)
    remove_different_adducts = tk.BooleanVar(value=True)
    process_secondary_data = tk.BooleanVar(value=True)

    # Add Input Box
    sn_threshold_high_var = tk.DoubleVar(value=10.0) # High Threshold of S/N
    sn_threshold_low_var = tk.DoubleVar(value=5.0)    # Low Threshold of S/N
    height_threshold_var = tk.IntVar(value=1000)      # Height Setting Value
    height_min_threshold_var = tk.IntVar(value=500)   # Minimum Height Threshold
    intensity_threshold_var = tk.IntVar(value=200)    # Intensity Threshold
    response_percentage_var = tk.DoubleVar(value=0.05) # Maximum Response Percentage
    ion_threshold_var = tk.IntVar(value=1)            # Minimum Ion Quantity
    rt_window_var = tk.DoubleVar(value=0.05)          # RT Window Size
    mw_diff_threshold_var = tk.DoubleVar(value=0.02)  # MW Difference Threshold

    def select_files():
        global selected_files
        selected_files = filedialog.askopenfilenames(filetypes=[("TXT files", "*.txt")])
        if selected_files:
            messagebox.showinfo("File selection", f"{len(selected_files)} files have been selected.")

    def run_process():
        if not selected_files:
            messagebox.showwarning("Warning", "Please select the TXT file first.")
            return
        
        output_dir = os.path.dirname(selected_files[0])
        result_file = process_files(
            selected_files, 
            mode.get(), 
            output_dir, 
            remove_isotopes.get(), 
            remove_different_adducts.get(), 
            process_secondary_data.get(), 
            sn_threshold_high_var.get(), 
            sn_threshold_low_var.get(), 
            height_threshold_var.get(), 
            height_min_threshold_var.get(),
            intensity_threshold_var.get(), 
            response_percentage_var.get(), 
            ion_threshold_var.get(),
            rt_window_var.get(), 
            mw_diff_threshold_var.get()
        )
        messagebox.showinfo("Completed", f"Processing completed. The result has been saved as {result_file}")

    # Create interface input boxes and buttons
    import_button = tk.Button(root, text="Import TXT file", command=select_files)
    import_button.pack(pady=20)

    mode_label = tk.Label(root, text="Selection Mode:")
    mode_label.pack()

    mode_option_menu = tk.OptionMenu(root, mode, 'Pos_Mode', 'Neg_Mode')
    mode_option_menu.pack(pady=10)

    remove_isotopes_check = tk.Checkbutton(root, text="Remove isotopes", variable=remove_isotopes)
    remove_isotopes_check.pack(pady=5)

    remove_different_adducts_check = tk.Checkbutton(root, text="Remove different loading methods", variable=remove_different_adducts)
    remove_different_adducts_check.pack(pady=5)

    process_secondary_data_check = tk.Checkbutton(root, text="Process secondary data", variable=process_secondary_data)
    process_secondary_data_check.pack(pady=5)

    # High S/N Threshold Input Box
    sn_high_label = tk.Label(root, text="S/N High Threshold (Height is less than the set value):")
    sn_high_label.pack()
    sn_high_entry = tk.Entry(root, textvariable=sn_threshold_high_var)
    sn_high_entry.pack()

    # Low S/N Threshold Input Box
    sn_low_label = tk.Label(root, text="S/N Low Threshold (Height is greater than or equal to the set value):")
    sn_low_label.pack()
    sn_low_entry = tk.Entry(root, textvariable=sn_threshold_low_var)
    sn_low_entry.pack()

    # Height Threshold Input Box
    height_label = tk.Label(root, text="Height setting value:")
    height_label.pack()
    height_entry = tk.Entry(root, textvariable=height_threshold_var)
    height_entry.pack()

    # Minimum Height Threshold Input Box
    height_min_label = tk.Label(root, text="Minimum Height Threshold:")
    height_min_label.pack()
    height_min_entry = tk.Entry(root, textvariable=height_min_threshold_var)
    height_min_entry.pack()

    # Intensity Threshold Input Box
    intensity_label = tk.Label(root, text="Secondary Intensity Threshold:")
    intensity_label.pack()
    intensity_entry = tk.Entry(root, textvariable=intensity_threshold_var)
    intensity_entry.pack()

    # Response Percentage Threshold Input Box
    response_percentage_label = tk.Label(root, text="Secondary maximum response percentage threshold:")
    response_percentage_label.pack()
    response_percentage_entry = tk.Entry(root, textvariable=response_percentage_var)
    response_percentage_entry.pack()

    # Ion Threshold Input Box
    ion_threshold_label = tk.Label(root, text="Minimum number of secondary ions:")
    ion_threshold_label.pack()
    ion_threshold_entry = tk.Entry(root, textvariable=ion_threshold_var)
    ion_threshold_entry.pack()

    # RT Window Threshold Input Box
    rt_window_label = tk.Label(root, text="Load method weight removal RT window size (min):")
    rt_window_label.pack()
    rt_window_entry = tk.Entry(root, textvariable=rt_window_var)
    rt_window_entry.pack()

    # MW Difference Threshold Input Box
    mw_diff_threshold_label = tk.Label(root, text="Load application method weight difference threshold for MW:")
    mw_diff_threshold_label.pack()
    mw_diff_threshold_entry = tk.Entry(root, textvariable=mw_diff_threshold_var)
    mw_diff_threshold_entry.pack()

    # Run Button
    run_button = tk.Button(root, text="Operation", command=run_process)
    run_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()