import os
import re
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

csv_file_path = ""
msp_file_paths = []
filter_mode = "all"  # Default mode is all

def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df

def read_msp_files(file_paths):
    msp_contents = {}
    for msp_file in file_paths:
        file_id = os.path.splitext(os.path.basename(msp_file))[0].replace('Msp_', '')
        with open(msp_file, 'r') as file:
            msp_contents[file_id] = file.read()
    return msp_contents

def modify_msp_id(msp_content, file_id, csv_ids):
    lines = msp_content.splitlines()
    modified_lines = []
    for line in lines:
        if line.startswith("COMMENT:") and "PEAKID=" in line:
            match = re.search(r"PEAKID=(\d+)", line)
            if match:
                peak_id = match.group(1)
                new_id = f"{file_id}_{peak_id}" if f"{file_id}_{peak_id}" in csv_ids else peak_id
                line = re.sub(r"PEAKID=\d+", f"PEAKID={new_id}", line)
        modified_lines.append(line)
    return "\n".join(modified_lines)

def filter_and_process_msp_file(msp_content, csv_ids, mode):
    filtered_content = []
    current_block = []
    include_block = False
    keep_block = True

    for line in msp_content.splitlines():
        if line.startswith("NAME:"):
            # Determine whether to save the previous block
            if current_block and include_block and keep_block:
                filtered_content.extend(current_block)
                filtered_content.append("")
            # Start a new block
            current_block = []
            include_block = False
            keep_block = True

            # Pattern Judgment
            if mode == "annotated" and "Unknown" in line:
                keep_block = False
            elif mode == "unknowns" and "Unknown" not in line:
                keep_block = False

        current_block.append(line)

        # Determine the value of PRECURSORMZ
        if line.startswith("PRECURSORMZ:"):
            try:
                mz_value = float(line.split(":")[1].strip())
                if mz_value >= 1500:
                    keep_block = False
            except ValueError:
                continue

        # Determine whether PEAKID exists in the CSV file
        if line.startswith("COMMENT:") and "PEAKID=" in line:
            match = re.search(r"PEAKID=(.*?)(\||$)", line)
            if match:
                peak_id = match.group(1)
                if peak_id in csv_ids:
                    include_block = True

    # Save the last block
    if current_block and include_block and keep_block:
        filtered_content.extend(current_block)

    return "\n".join(filtered_content).strip()

def save_filtered_msp_files(msp_contents, filtered_contents, output_dir):
    merged_content = ""
    for file_id, filtered_content in filtered_contents.items():
        new_file_name = f"new_{file_id}.msp"
        new_file_path = os.path.join(output_dir, new_file_name)
        with open(new_file_path, 'w') as file:
            file.write(filtered_content)
        merged_content += filtered_content + "\n\n"
        print(f"{new_file_name} saved.")
    
    # Save the merged content
    merged_file_path = os.path.join(output_dir, "merged_filtered.msp")
    with open(merged_file_path, 'w') as merged_file:
        merged_file.write(merged_content)
    print(f"Merged filtered MSP file saved as {merged_file_path}")

def select_csv_file():
    global csv_file_path
    csv_file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
    if csv_file_path:
        messagebox.showinfo("Success", f"Successfully selected CSV file: {os.path.basename(csv_file_path)}")

def select_msp_files():
    global msp_file_paths
    msp_file_paths = filedialog.askopenfilenames(title="Select MSP Files", filetypes=[("MSP Files", "*.msp")])
    if msp_file_paths:
        messagebox.showinfo("Success", f"Successfully selected {len(msp_file_paths)} MSP files")

def select_filter_mode(selected_mode):
    global filter_mode
    filter_mode = selected_mode

def run_process():
    if not csv_file_path or not msp_file_paths:
        messagebox.showerror("Error", "Please select both CSV and MSP files.")
        return
    
    # Reading CSV files and MSP files
    csv_df = read_csv_file(csv_file_path)
    csv_ids = set(csv_df['ID'])
    msp_contents = read_msp_files(msp_file_paths)
    
    # Modify the ID in the MSP file
    modified_msp_contents = {}
    for file_id, content in msp_contents.items():
        modified_content = modify_msp_id(content, file_id, csv_ids)
        modified_msp_contents[file_id] = modified_content
    
    # Filter and process each MSP file
    filtered_msp_contents = {}
    for file_id, content in modified_msp_contents.items():
        filtered_content = filter_and_process_msp_file(content, csv_ids, filter_mode)
        filtered_msp_contents[file_id] = filtered_content
    
    # Save the filtered MSP file and the integrated MSP file
    output_dir = os.path.dirname(csv_file_path)
    save_filtered_msp_files(msp_contents, filtered_msp_contents, output_dir)
    
    messagebox.showinfo("Success", f"Filtered MSP files have been saved in {output_dir}")

# Create a simple visual operation interface
root = tk.Tk()
root.title("MSP File Processor")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

csv_button = tk.Button(frame, text="Select CSV File", command=select_csv_file)
csv_button.grid(row=0, column=0, pady=5)

msp_button = tk.Button(frame, text="Select MSP Files", command=select_msp_files)
msp_button.grid(row=1, column=0, pady=5)

filter_label = tk.Label(frame, text="Select Filter Mode:")
filter_label.grid(row=2, column=0, pady=5)

filter_var = tk.StringVar(value="all")
filter_all = tk.Radiobutton(frame, text="All", variable=filter_var, value="all", command=lambda: select_filter_mode("all"))
filter_annotated = tk.Radiobutton(frame, text="Annotated", variable=filter_var, value="annotated", command=lambda: select_filter_mode("annotated"))
filter_unknowns = tk.Radiobutton(frame, text="Unknowns", variable=filter_var, value="unknowns", command=lambda: select_filter_mode("unknowns"))

filter_all.grid(row=3, column=0, pady=5)
filter_annotated.grid(row=4, column=0, pady=5)
filter_unknowns.grid(row=5, column=0, pady=5)

run_button = tk.Button(frame, text="Run", command=run_process)
run_button.grid(row=6, column=0, columnspan=2, pady=10)

root.mainloop()
