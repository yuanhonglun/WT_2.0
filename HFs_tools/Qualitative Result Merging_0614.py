import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

def process_compound_identifications(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df['new_id'] = df['id'].apply(lambda x: '_'.join(x.split('_')[2:]))
    output_path = os.path.join(os.path.dirname(file_path), 'compound_identifications_new_id.tsv')
    df.to_csv(output_path, sep='\t', index=False)
    return df

def merge_results(msdial_file, sirius_df):
    msdial_df = pd.read_csv(msdial_file)
    msdial_df['SIRIUS_result'] = ''
    msdial_df['SIRIUS_formula'] = ''
    
    for _, row in sirius_df.iterrows():
        msdial_df.loc[msdial_df['ID'] == row['new_id'], 'SIRIUS_result'] = row['name']
        msdial_df.loc[msdial_df['ID'] == row['new_id'], 'SIRIUS_formula'] = row['molecularFormula']
    
    output_path = os.path.join(os.path.dirname(msdial_file), 'combined_result.csv')
    msdial_df.to_csv(output_path, index=False)
    
    unknown_sirius_count = msdial_df[(msdial_df['Name'] == 'Unknown') & (msdial_df['SIRIUS_result'] != '')].shape[0]
    return output_path, unknown_sirius_count

def import_msdial_result():
    msdial_file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if msdial_file:
        global msdial_file_path
        msdial_file_path = msdial_file
        messagebox.showinfo("Success", "MSDIAL result imported successfully!")

def import_sirius_result():
    sirius_file = filedialog.askopenfilename(filetypes=[("TSV files", "*.tsv")])
    if sirius_file:
        global sirius_file_path
        sirius_file_path = sirius_file
        messagebox.showinfo("Success", "SIRIUS result imported successfully!")
        
def run():
    if not msdial_file_path or not sirius_file_path:
        messagebox.showerror("Error", "Please import both MSDIAL and SIRIUS results!")
        return
    
    sirius_df = process_compound_identifications(sirius_file_path)
    output_path, unknown_sirius_count = merge_results(msdial_file_path, sirius_df)
    
    messagebox.showinfo("Success", f"Results combined successfully!\nOutput file: {output_path}\nNumber of unknowns with SIRIUS result: {unknown_sirius_count}")
    
app = tk.Tk()
app.title("MSDIAL and SIRIUS Results Merger")

msdial_file_path = None
sirius_file_path = None

btn_import_msdial = tk.Button(app, text="Import MSDIAL result", command=import_msdial_result)
btn_import_msdial.pack(pady=10)

btn_import_sirius = tk.Button(app, text="Import SIRIUS result", command=import_sirius_result)
btn_import_sirius.pack(pady=10)

btn_run = tk.Button(app, text="Run", command=run)
btn_run.pack(pady=10)

app.mainloop()
