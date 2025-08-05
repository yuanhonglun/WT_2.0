import os
import re
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

csv_file_path = ""
msp_file_paths = []
mode = "single sample"  # The default mode is "single sample"


def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    if mode == "batch samples":
        # Add new ID column and file_ID column
        df['file_ID'] = df['Spectrum reference file name'].apply(
            lambda x: re.search(r"(P_\d+-\d+|N_\d+-\d+)", x).group()
        )
        df['ID'] = df.apply(
            lambda row: f"{row['file_ID']}_{row['Alignment ID']}", axis=1
        )
    return df


def read_msp_files(file_paths):
    msp_contents = {}
    for msp_file in file_paths:
        file_id = os.path.splitext(os.path.basename(msp_file))[0]
        with open(msp_file, 'r') as file:
            msp_contents[file_id] = file.read()
    return msp_contents


def modify_msp_id(msp_content, file_id):
    lines = msp_content.splitlines()
    modified_lines = []
    for line in lines:
        if line.startswith("COMMENT:") and "PEAKID=" in line:
            match = re.search(r"PEAKID=(\d+)", line)
            if match:
                peak_id = match.group(1)
                line = re.sub(r"PEAKID=\d+", f"PEAKID={file_id}_{peak_id}", line)
        modified_lines.append(line)
    return "\n".join(modified_lines)


def filter_and_process_msp_file(msp_content, csv_ids):
    filtered_content = []
    current_block = []
    include_block = False
    for line in msp_content.splitlines():
        if line.startswith("NAME:"):
            if current_block and include_block:
                filtered_content.extend(current_block)
                filtered_content.append("")
            current_block = []
            include_block = False
        current_block.append(line)
        if line.startswith("COMMENT:") and "PEAKID=" in line:
            match = re.search(r"PEAKID=(.*?)(\||$)", line)
            if match:
                peak_id = match.group(1)
                if peak_id in csv_ids:
                    include_block = True
    if current_block and include_block:
        filtered_content.extend(current_block)
    return "\n".join(filtered_content).strip()


def save_filtered_msp(filtered_content, output_dir, output_name="filtered_output.msp"):
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, 'w') as file:
        file.write(filtered_content)
    print(f"Filtered MSP saved at {output_path}")


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


def select_mode(selected_mode):
    global mode
    mode = selected_mode


def run_process():
    if not csv_file_path or not msp_file_paths:
        messagebox.showerror("Error", "Please select both CSV and MSP files.")
        return

    # Read CSV file
    csv_df = read_csv_file(csv_file_path)
    csv_ids = set(csv_df['ID'])

    # Read MSP file
    msp_contents = read_msp_files(msp_file_paths)

    # Modify the ID in the MSP file
    modified_msp_contents = {}
    for file_id, content in msp_contents.items():
        modified_content = modify_msp_id(content, file_id)
        modified_msp_contents[file_id] = modified_content

    # Processing Logic
    if mode == "single sample":
        filtered_msp_contents = {}
        for file_id, content in modified_msp_contents.items():
            filtered_content = filter_and_process_msp_file(content, csv_ids)
            filtered_msp_contents[file_id] = filtered_content
        # Merge all filtered contents
        merged_content = "\n\n".join(filtered_msp_contents.values())
        save_filtered_msp(merged_content, os.path.dirname(csv_file_path), "merged_filtered.msp")
    elif mode == "batch samples":
        batch_filtered_content = []
        for _, row in csv_df.iterrows():
            file_id = row['file_ID']
            if file_id not in msp_contents:
                print(f"Error: MSP file not found for file_ID={file_id}")
                continue
            filtered_content = filter_and_process_msp_file(modified_msp_contents[file_id], {row['ID']})
            if filtered_content:
                batch_filtered_content.append(filtered_content)
        # Save Batch Results
        save_filtered_msp("\n\n".join(batch_filtered_content), os.path.dirname(csv_file_path), "batch_filtered.msp")

    messagebox.showinfo("Success", "MSP processing completed.")


# Create a simple visual operation interface
root = tk.Tk()
root.title("MSP File Processor")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

csv_button = tk.Button(frame, text="Select CSV File", command=select_csv_file)
csv_button.grid(row=0, column=0, pady=5)

msp_button = tk.Button(frame, text="Select MSP Files", command=select_msp_files)
msp_button.grid(row=1, column=0, pady=5)

mode_label = tk.Label(frame, text="Select Mode:")
mode_label.grid(row=2, column=0, pady=5)

mode_var = tk.StringVar(value="single sample")
mode_single = tk.Radiobutton(frame, text="Single Sample", variable=mode_var, value="single sample",
                             command=lambda: select_mode("single sample"))
mode_batch = tk.Radiobutton(frame, text="Batch Samples", variable=mode_var, value="batch samples",
                            command=lambda: select_mode("batch samples"))

mode_single.grid(row=3, column=0, pady=5)
mode_batch.grid(row=4, column=0, pady=5)

run_button = tk.Button(frame, text="Run", command=run_process)
run_button.grid(row=5, column=0, columnspan=2, pady=10)

root.mainloop()
