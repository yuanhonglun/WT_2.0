import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Toplevel, Listbox, MULTIPLE
import pandas as pd
import numpy as np

def calculate_mw(row, mode):
    precursor_mz = row['Average Mz']
    adduct = row['Adduct type']
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

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("MS-DIAL batch data integration and deduplicationv0.1.240918")

        # 初始化
        self.debug_mode = tk.BooleanVar(value=False)
        self.response_files = []
        self.snr_files = []
        self.merged_df = None
        self.blank_group_columns = None
        self.group_columns = None

        # Debug模式选项
        self.debug_checkbox = tk.Checkbutton(root, text="Debug mode", variable=self.debug_mode)
        self.debug_checkbox.pack(pady=5)

        # 导入响应表格
        self.btn_import_response = tk.Button(root, text="Import intensity form", command=self.import_response)
        self.btn_import_response.pack(pady=5)

        # 导入SNR表格
        self.btn_import_snr = tk.Button(root, text="Import the SNR table", command=self.import_snr)
        self.btn_import_snr.pack(pady=5)

        # 拼接表格按钮
        self.btn_concat = tk.Button(root, text="Merge tables", command=self.concat_tables)
        self.btn_concat.pack(pady=5)

        # 选择空白组
        self.btn_select_blank = tk.Button(root, text="Select the blank group", command=self.select_blank_groups)
        self.btn_select_blank.pack(pady=5)

        # 分组数量输入对话框
        self.btn_group_selection = tk.Button(root, text="Number of input groups", command=self.group_selection)
        self.btn_group_selection.pack(pady=5)

        # 空白倍数阈值
        self.blank_threshold_label = tk.Label(root, text="Blank factor threshold (default: 5):")
        self.blank_threshold_label.pack(pady=5)
        self.blank_threshold_entry = tk.Entry(root)
        self.blank_threshold_entry.insert(0, "5")
        self.blank_threshold_entry.pack(pady=5)

        # 最小响应阈值
        self.intensity_threshold_label = tk.Label(root, text="Minimum response threshold (default: 1000):")
        self.intensity_threshold_label.pack(pady=5)
        self.intensity_threshold_entry = tk.Entry(root)
        self.intensity_threshold_entry.insert(0, "1000")
        self.intensity_threshold_entry.pack(pady=5)

        # 最小SNR阈值
        self.snr_threshold_label = tk.Label(root, text="Minimum SNR threshold (default: 5):")
        self.snr_threshold_label.pack(pady=5)
        self.snr_threshold_entry = tk.Entry(root)
        self.snr_threshold_entry.insert(0, "5")
        self.snr_threshold_entry.pack(pady=5)

        # 正负离子模式选项卡
        self.mode_var = tk.StringVar(value="Pos_Mode")
        self.mode_label = tk.Label(root, text="Select ion mode:")
        self.mode_label.pack(pady=5)
        self.mode_pos = tk.Radiobutton(root, text="Positive ion mode", variable=self.mode_var, value="Pos_Mode")
        self.mode_pos.pack(pady=5)
        self.mode_neg = tk.Radiobutton(root, text="Negative ion mode", variable=self.mode_var, value="Neg_Mode")
        self.mode_neg.pack(pady=5)

        # MW去重参数
        self.mw_rt_window_label = tk.Label(root, text="MW duplicate removal RT window ± (min) (default: 0.05):")
        self.mw_rt_window_label.pack(pady=5)
        self.mw_rt_window_entry = tk.Entry(root)
        self.mw_rt_window_entry.insert(0, "0.05")
        self.mw_rt_window_entry.pack(pady=5)

        self.mw_diff_threshold_label = tk.Label(root, text="MW duplicate removal difference threshold (default: 0.02):")
        self.mw_diff_threshold_label.pack(pady=5)
        self.mw_diff_threshold_entry = tk.Entry(root)
        self.mw_diff_threshold_entry.insert(0, "0.02")
        self.mw_diff_threshold_entry.pack(pady=5)

        # MS2处理参数
        self.ms2_intensity_threshold_label = tk.Label(root, text="MS2 Absolute Response Threshold (default: 200):")
        self.ms2_intensity_threshold_label.pack(pady=5)
        self.ms2_intensity_threshold_entry = tk.Entry(root)
        self.ms2_intensity_threshold_entry.insert(0, "200")
        self.ms2_intensity_threshold_entry.pack(pady=5)

        self.ms2_response_percentage_label = tk.Label(root, text="MS2 Maximum Response Percentage Threshold (%) (Default: 5):")
        self.ms2_response_percentage_label.pack(pady=5)
        self.ms2_response_percentage_entry = tk.Entry(root)
        self.ms2_response_percentage_entry.insert(0, "5")
        self.ms2_response_percentage_entry.pack(pady=5)

        self.ms2_ion_threshold_label = tk.Label(root, text="MS2 Minimum Ion Count (default: 2):")
        self.ms2_ion_threshold_label.pack(pady=5)
        self.ms2_ion_threshold_entry = tk.Entry(root)
        self.ms2_ion_threshold_entry.insert(0, "2")
        self.ms2_ion_threshold_entry.pack(pady=5)
        
        self.process_data_button = tk.Button(self.root, text="Process data", command=self.process_data)
        self.process_data_button.pack(pady=10)
        
    def import_response(self):
        self.response_files = filedialog.askopenfilenames(title="Select the intensity form", filetypes=[("CSV Files", "*.csv")])
        messagebox.showinfo("Success", f"Successfully imported {len(self.response_files)} response files")

    def import_snr(self):
        self.snr_files = filedialog.askopenfilenames(title="Select the SNR table", filetypes=[("CSV Files", "*.csv")])
        if len(self.snr_files) != len(self.response_files):
            messagebox.showerror("Error", "The number of response forms and SNR forms do not match!")
        else:
            messagebox.showinfo("Success", f"Successfully imported {len(self.snr_files)} SNR forms")

    def concat_tables(self):
        # Check if the response form and SNR form have been imported
        if not self.response_files or not self.snr_files:
            messagebox.showerror("Error", "Please import the corresponding table and the SNR table first!")
            return
    
        merged_tables = []  # Used to store each concatenated table
    
        # Traverse the response table and the SNR table
        for idx, (resp_file, snr_file) in enumerate(zip(self.response_files, self.snr_files)):
            resp_df = pd.read_csv(resp_file, skiprows=4, on_bad_lines='skip')
            snr_df = pd.read_csv(snr_file, skiprows=4, on_bad_lines='skip')
    
            # Merging Logic: Matching based on Alignment ID
            for snr_idx, snr_row in snr_df.iterrows():
                alignment_id = snr_row['Alignment ID']
    
                # Locate the corresponding row in the response form and ensure that the types are consistent
                try:
                    resp_match_idx = resp_df[resp_df['Alignment ID'] == alignment_id].index
                except KeyError:
                    print(f"The column 'Alignment ID' does not exist. Please check if the column name is correct.")
                    continue
    
                if not resp_match_idx.empty:
                    # Obtain all columns after the MS/MS spectrum in the SNR table
                    snr_cols_to_merge = snr_row[snr_row.index.get_loc('MS/MS spectrum') + 1:]
                    # Add the prefix "SNR_" to the column names
                    snr_cols_to_merge.index = ["SNR_" + col for col in snr_cols_to_merge.index]
    
                    # Assign the values of these columns to the corresponding rows
                    for col, value in snr_cols_to_merge.items():
                        resp_df.loc[resp_match_idx, col] = value
    
            # Reset the index to prevent duplicate index conflicts
            resp_df.reset_index(drop=True, inplace=True)
    
            # Prompt for successful concatenation of print response table and SNR table
            print(f"The response table and the SNR table {idx+1} have been successfully concatenated.")
    
            # If the current mode is debug, export each of the concatenated response tables
            if self.debug_mode.get():
                resp_df.to_csv(f"intensity_SNR_{idx+1}.csv", index=False)
    
            merged_tables.append(resp_df)
    
        # Obtain the column names of the first table
        first_table_columns = merged_tables[0].columns
    
        # Concatenate all the tables vertically to ensure that the column names of other tables are consistent with that of the first table.
        merged_df = merged_tables[0]
        for df in merged_tables[1:]:
            # Change the column names of the current table to those of the first table
            df.columns = first_table_columns
            df.reset_index(drop=True, inplace=True)  # Reset the index to prevent conflicts
            merged_df = pd.concat([merged_df, df], ignore_index=True)
    
        # Notification of Completion of Printing of the Overall Table Assembly
        print("The master table has been successfully joined together.")
    
        # Sort by the "Average Rt(min)" column in ascending order
        merged_df.sort_values(by="Average Rt(min)", inplace=True)
    
        # If the current mode is debug, export the final total response table
        if self.debug_mode.get():
            merged_df.to_csv("merge_df.csv", index=False)
    
        # Update self.merged_df for subsequent use
        self.merged_df = merged_df
    
        messagebox.showinfo("Success", "Successful table stitching")

    def select_blank_groups(self):
        # Assume that the table has already been stitched together
        if self.merged_df is None:
            messagebox.showerror("Error", "Please concatenate the table first!")
            return
    
        columns = list(self.merged_df.columns)
    
        # Create a new window
        blank_window = Toplevel(self.root)
        blank_window.title("Select the blank group")
    
        # Tag
        lbl_response = tk.Label(blank_window, text="Blank group response column")
        lbl_response.grid(row=0, column=0, padx=10, pady=10)
        lbl_snr = tk.Label(blank_window, text="Blank group SNR column")
        lbl_snr.grid(row=0, column=1, padx=10, pady=10)
    
        # Response Selection Box
        self.blank_response_listbox = Listbox(blank_window, selectmode=MULTIPLE, height=10, exportselection=False)
        self.blank_response_listbox.grid(row=1, column=0, padx=10, pady=10)
    
        # SNR Column Selection Box
        self.blank_snr_listbox = Listbox(blank_window, selectmode=MULTIPLE, height=10, exportselection=False)
        self.blank_snr_listbox.grid(row=1, column=1, padx=10, pady=10)
    
        # Fill column names into two list boxes
        for col in columns:
            self.blank_response_listbox.insert(tk.END, col)
            self.blank_snr_listbox.insert(tk.END, col)
    
        # Submit Button
        submit_btn = tk.Button(blank_window, text="Confirm the selection", command=self.save_blank_group_selection)
        submit_btn.grid(row=2, column=0, columnspan=2, pady=10)

    def save_blank_group_selection(self):
        # Obtain the blank group response column and SNR column selected by the user
        blank_response_indices = self.blank_response_listbox.curselection()
        blank_snr_indices = self.blank_snr_listbox.curselection()
    
        blank_response_columns = [self.blank_response_listbox.get(i) for i in blank_response_indices]
        blank_snr_columns = [self.blank_snr_listbox.get(i) for i in blank_snr_indices]
    
        if not blank_response_columns and not blank_snr_columns:
            messagebox.showerror("Error", "Please select at least one response column or SNR column.")
            return
    
        # Print selection results
        print(f"Blank group response column: {blank_response_columns}")
        print(f"Blank group SNR column: {blank_snr_columns}")
    
        # Save Selection
        self.blank_group_columns = {
            "response": blank_response_columns,
            "snr": blank_snr_columns
        }
    
        # Close Window
        messagebox.showinfo("Success", "Blank group selects success")

    def group_selection(self):
        # Pop up an input box to allow users to enter the number of groups
        group_count = simpledialog.askinteger("Number of input groups", "Please enter the number of groups:")
        if group_count is None or group_count <= 0:
            messagebox.showerror("Error", "Please enter a valid number of groups!")
            return
    
        # Create Group Selection Window
        group_window = Toplevel(self.root)
        group_window.title(f"Select {group_count} groups of columns")
    
        self.group_response_listboxes = []
        self.group_snr_listboxes = []
    
        for i in range(group_count):
            # Tag
            lbl_response = tk.Label(group_window, text=f"The {i + 1}th group of response columns")
            lbl_response.grid(row=0, column=2*i, padx=10, pady=10)
            lbl_snr = tk.Label(group_window, text=f"The {i + 1}th group of SNR columns")
            lbl_snr.grid(row=0, column=2*i+1, padx=10, pady=10)
    
            # Response Selection Box
            response_listbox = Listbox(group_window, selectmode=MULTIPLE, height=10, exportselection=False)
            response_listbox.grid(row=1, column=2*i, padx=10, pady=10)
            self.group_response_listboxes.append(response_listbox)
    
            # SNR Column Selection Box
            snr_listbox = Listbox(group_window, selectmode=MULTIPLE, height=10, exportselection=False)
            snr_listbox.grid(row=1, column=2*i+1, padx=10, pady=10)
            self.group_snr_listboxes.append(snr_listbox)
    
            # Fill column names into the list box
            for col in self.merged_df.columns:
                response_listbox.insert(tk.END, col)
                snr_listbox.insert(tk.END, col)
    
        # Submit Button
        submit_btn = tk.Button(group_window, text="Confirm the selection", command=self.save_group_selection)
        submit_btn.grid(row=2, column=0, columnspan=2*group_count, pady=10)

    def save_group_selection(self):
        group_responses = []
        group_snrs = []
    
        # Obtain the response column and SNR column for each group
        for response_listbox, snr_listbox in zip(self.group_response_listboxes, self.group_snr_listboxes):
            response_indices = response_listbox.curselection()
            snr_indices = snr_listbox.curselection()
    
            response_columns = [response_listbox.get(i) for i in response_indices]
            snr_columns = [snr_listbox.get(i) for i in snr_indices]
    
            if response_columns and snr_columns:
                group_responses.append(response_columns)
                group_snrs.append(snr_columns)
            else:
                messagebox.showerror("Error", "Please select at least one response column and SNR column for each group.")
                return
    
        # Print selection results
        print(f"Group Response Column: {group_responses}")
        print(f"Grouped SNR column: {group_snrs}")
    
        # Save Selection
        self.group_columns = {
            "response": group_responses,
            "snr": group_snrs
        }
    
        # Close Window
        messagebox.showinfo("Success", "Group selection successful")

    def get_thresholds(self):
        # Obtain the blanking factor threshold, the minimum response threshold, and the minimum SNR threshold
        blank_threshold = float(self.blank_threshold_entry.get())
        intensity_threshold = float(self.intensity_threshold_entry.get())
        snr_threshold = float(self.snr_threshold_entry.get())
    
        # Obtain MW deduplication parameters
        rt_window = float(self.mw_rt_window_entry.get())
        mw_diff_threshold = float(self.mw_diff_threshold_entry.get())
    
        # Obtain MS2 processing parameters
        ms2_intensity_threshold = float(self.ms2_intensity_threshold_entry.get())
        ms2_response_percentage = float(self.ms2_response_percentage_entry.get()) / 100
        ms2_ion_threshold = int(self.ms2_ion_threshold_entry.get())
    
        return {
            "blank_threshold": blank_threshold,
            "intensity_threshold": intensity_threshold,
            "snr_threshold": snr_threshold,
            "rt_window": rt_window,
            "mw_diff_threshold": mw_diff_threshold,
            "ms2_intensity_threshold": ms2_intensity_threshold,
            "ms2_response_percentage": ms2_response_percentage,
            "ms2_ion_threshold": ms2_ion_threshold
        }

    def process_msms_spectrum(self, row, ms2_intensity_threshold, ms2_response_percentage, ms2_ion_threshold):
        if pd.isnull(row['MS/MS spectrum']):
            return ''
        
        spectra = row['MS/MS spectrum'].split(' ')
        if self.debug_mode.get():
            print("spectra = ", spectra)
            
        processed_spectra = []
        
        for s in spectra:
            parts = s.split(':')
            if self.debug_mode.get():
                print("parts = ", parts)
                print("len(parts) = ", len(parts))
            if len(parts) == 2:
                try:
                    intensity = int(parts[1])
                    if intensity >= ms2_intensity_threshold:
                        processed_spectra.append(s)
                except ValueError:
                    continue
    
        if not processed_spectra:
            return ''
    
        max_response = max([int(s.split(':')[1]) for s in processed_spectra])
        threshold = max_response * ms2_response_percentage
        final_spectra = [s for s in processed_spectra if int(s.split(':')[1]) >= threshold]
        
        if self.debug_mode.get():
            print("len(final_spectra) = ", len(final_spectra))
        
        if len(final_spectra) <= ms2_ion_threshold:
            return ''
        
        final_spectra = sorted(final_spectra, key=lambda x: int(x.split(':')[1]), reverse=True)
        return ';'.join(final_spectra)

    def process_data(self):
        # Extract all the input thresholds
        thresholds = self.get_thresholds()
        mode = self.mode_var.get() # Obtain positive and negative ion mode
        
        print(f"初始表格 {len(self.merged_df)} 行")
        
        # If there are blank groups, calculate the average of the responses and SNR for the blank groups
        if self.blank_group_columns:
            self.merged_df['blank_ave_intensity'] = self.merged_df[self.blank_group_columns['response']].mean(axis=1)
            self.merged_df['blank_ave_SNR'] = self.merged_df[self.blank_group_columns['snr']].mean(axis=1)
    
        # Calculate the average response and SNR values for each group
        for i, (response_cols, snr_cols) in enumerate(zip(self.group_columns['response'], self.group_columns['snr']), start=1):
            self.merged_df[f'Group_{i}_ave_intensity'] = self.merged_df[response_cols].mean(axis=1)
            self.merged_df[f'Group_{i}_ave_SNR'] = self.merged_df[snr_cols].mean(axis=1)
    
        # Filter out the rows that do not meet the blank ratio threshold
        if 'blank_ave_intensity' in self.merged_df.columns:
            def check_blank_threshold(row):
                for i in range(1, len(self.group_columns['response']) + 1):
                    if row['blank_ave_intensity'] <= 0:
                        row['blank_ave_intensity'] = 0.1
                    factor = row[f'Group_{i}_ave_intensity'] / row['blank_ave_intensity']
                    if factor >= thresholds['blank_threshold']:
                        return False  # If a group meets the threshold, keep it.
                return True  # All groups are below the threshold, delete
            
            self.merged_df = self.merged_df[~self.merged_df.apply(check_blank_threshold, axis=1)]
            print(f"After the blank multiple filtering, {len(self.merged_df)} rows remain.")
    
        # Eliminate the rows that do not meet the minimum response threshold
        def check_intensity_threshold(row):
            for i in range(1, len(self.group_columns['response']) + 1):
                if row[f'Group_{i}_ave_intensity'] >= thresholds['intensity_threshold']:
                    return False  # If a group meets the threshold, keep it.
            return True  # All groups are below the threshold, delete
    
        self.merged_df = self.merged_df[~self.merged_df.apply(check_intensity_threshold, axis=1)]
        print(f"After filtering based on the minimum response value,  {len(self.merged_df)} rows remain.")
    
        # Filter out the rows that do not meet the minimum SNR threshold
        def check_snr_threshold(row):
            for i in range(1, len(self.group_columns['snr']) + 1):
                if row[f'Group_{i}_ave_SNR'] >= thresholds['snr_threshold']:
                    return False # If a group meets the threshold, keep it.
            return True # All groups are below the threshold, delete
    
        self.merged_df = self.merged_df[~self.merged_df.apply(check_snr_threshold, axis=1)]
        print(f"After the minimum SNR value screening, {len(self.merged_df)} rows remain.")
    
        # Delete the rows in the "Isotope tracking weight number" column that do not have a value of 0
        self.merged_df = self.merged_df[self.merged_df["Isotope tracking weight number"] == 0]
        print(f"After deleting the rows where 'Isotope tracking weight number' is not equal to 0, the remaining number of rows is {len(self.merged_df)}.")
    
        # Calculate MW and filter out duplicate rows
        self.merged_df['MW'] = self.merged_df.apply(lambda row: calculate_mw(row, mode), axis=1)
        self.merged_df = self.merged_df[self.merged_df['MW'] != 0]
        print(f"After calculating the MW and deleting the rows where MW is 0, the remaining number of rows is  {len(self.merged_df)}.")
    
        # Remove duplicate rows, based on MW and RT columns
        rt_window = thresholds['rt_window']
        mw_diff_threshold = thresholds['mw_diff_threshold']
    
        to_delete_indices = []
        for idx in range(len(self.merged_df)):
            if idx in to_delete_indices:
                continue
            target_row = self.merged_df.iloc[idx]
            rt_min = max(0, target_row['Average Rt(min)'] - rt_window)
            rt_max = target_row['Average Rt(min)'] + rt_window
            same_rt_rows = self.merged_df[(self.merged_df['Average Rt(min)'] >= rt_min) & (self.merged_df['Average Rt(min)'] <= rt_max)]
            
            same_rt_rows = same_rt_rows[abs(same_rt_rows['MW'] - target_row['MW']) <= mw_diff_threshold]
        
            if len(same_rt_rows) > 1:
                # Calculate the maximum value of the average of each group in each row
                max_avg_values = same_rt_rows.apply(lambda row: max(
                    [row[f'Group_{i}_ave_intensity'] for i in range(1, len(self.group_columns['response']) + 1)]
                ), axis=1)
        
                # Find the index of the row with the maximum average value and the maximum value
                max_avg_idx = max_avg_values.idxmax()
        
                # Keep the rows with the maximum average value and delete the others
                to_delete_indices.extend(same_rt_rows.index.difference([max_avg_idx]))
        
        # Delete the marked rows
        self.merged_df = self.merged_df.drop(index=to_delete_indices)
        print(f"After removing duplicates based on the MW value, there are {len(self.merged_df)} remaining rows.")
    
        # Process MS2 spectra
        self.merged_df['MS/MS spectrum'] = self.merged_df.apply(lambda row: self.process_msms_spectrum(
            row,
            thresholds['ms2_intensity_threshold'],
            thresholds['ms2_response_percentage'],
            thresholds['ms2_ion_threshold']), axis=1)
        print(f"MS2 spectrum processing is complete.")
    
       # Save the final result
        self.merged_df.to_csv("final_processed_data.csv", index=False)
        print("Data processing is complete. The results have been saved as final_processed_data.csv")
        messagebox.showinfo("Success", "Data processing completed. Results saved as final_processed_data.csv")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()