import os
import tkinter as tk
from tkinter import filedialog, messagebox

def merge_msp_files(file_paths):
    merged_data = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            merged_data.append(content.strip())
    return "\n\n".join(merged_data)

def filter_compounds(data, mode):
    valid_modes = {
        "positive": ["[M+H]+", "[M+NH4]+", "[M+H-H2O]+", "[M+Na]+", "[M+K]+", "M+NH4", "M+H", "M+Na", "M+K", "M+", "[M]+", "[M-H2O+H]+", "[M+H]", "[M+]", "[M+H]+", "[M-OH]+", "[M+NH4]+", "[M+K]+", "[M+Na]+", "[M]+", "[M-H2O+H]+", "M+H", "[M+H-H2O]+", "M+Na", "M+", "M+NH4", "M+K", "[M+H]+", "[M+Na]+"],
        "negative": ["[M-H]-", "M-H", "[M-H2O-H]", "[M-H]", "[M-H20-H]-", "[M-H]1-", "[M-H2O-H]-", "[M-H]-", "[M-H2O-H]-", "[M-H]1-", "[M-H-H2O]-"]
    }
    valid_modes = list(set(valid_modes[mode]))  # 去重

    compounds = data.split("\n\n")
    filtered_compounds = []
    for compound in compounds:
        lines = compound.split("\n")
        for line in lines:
            if line.startswith("Precursor_type:") or line.startswith("PRECURSORTYPE:"):
                precursor_type = line.split(":")[1].strip()
                if precursor_type in valid_modes:
                    filtered_compounds.append(compound)
                    break
    return "\n\n".join(filtered_compounds)

def save_filtered_msp(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(data)

class MSPFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MSP Filter")
        self.mode = "positive"
        self.msp_files = []
        
        self.import_button = tk.Button(root, text="Import MSP file", command=self.import_msp_files)
        self.import_button.pack(pady=10)
        
        self.mode_button = tk.Button(root, text="Switch to negative ion mode", command=self.toggle_mode)
        self.mode_button.pack(pady=10)
        
        self.run_button = tk.Button(root, text="Operation", command=self.run)
        self.run_button.pack(pady=10)
        
    def import_msp_files(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("MSP files", "*.msp")])
        if file_paths:
            self.msp_files = file_paths
            messagebox.showinfo("Success", "Successful Import")
    
    def toggle_mode(self):
        if self.mode == "positive":
            self.mode = "negative"
            self.mode_button.config(text="Switch to the positive ion mode")
        else:
            self.mode = "positive"
            self.mode_button.config(text="Switch to negative ion mode")
    
    def run(self):
        if not self.msp_files:
            messagebox.showwarning("Warning", "Please import the MSP file first")
            return
        
        merged_data = merge_msp_files(self.msp_files)
        filtered_data = filter_compounds(merged_data, self.mode)
        
        output_path = os.path.join(os.path.dirname(self.msp_files[0]), "filtered_output.msp")
        save_filtered_msp(filtered_data, output_path)
        
        num_compounds_before = len(merged_data.split("\n\n"))
        num_compounds_after = len(filtered_data.split("\n\n"))
        
        print(f"The total number of combined compounds: {num_compounds_before}")
        print(f"The number of filtered compounds: {num_compounds_after}")
        
        messagebox.showinfo("Success", f"Export successful!\nNumber of combined compounds: {num_compounds_before}\nNumber of compounds after filtering: {num_compounds_after}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MSPFilterApp(root)
    root.mainloop()
