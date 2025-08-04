import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from WT_2.database_builder import DatabaseBuilder
from WT_2.peak_processor import PeakProcessor
from WT_2.remove_duplicate import ResultFormatter


class MultiprocessingManager:
    def __init__(self, outer_max_workers, inner_max_workers, mgf_folder, out_dir, RT_start=30, RT_end=720, fp_wid=6, fp_sigma=2, fp_min_noise=200, group_wid=6, group_sigma=0.5):
        """
        Initialize the multi-process management class

        :param outer_max_workers: Maximum number of working processes in the outer process pool (for processing multiple MGF files)
        :param inner_max_workers: The maximum number of working processes in the inner process pool (for processing multiple pepmasses within a folder)
        :param mgf_folder: The path of a single sample folder, with multiple MGF files contained within the folder.
        :param product_ion_db: Product ion database path
        :param out_dir: Output path
        """
        self.outer_max_workers = outer_max_workers  # Maximum number of working processes in the outer process pool
        self.inner_max_workers = inner_max_workers  # Maximum number of working processes in the inner process pool
        self.mgf_folder = mgf_folder  # Folder Path
        self.out_dir = out_dir  # Output Path

        # Peak Identification Parameters
        self.RT_start = RT_start
        self.RT_end = RT_end
        self.fp_wid = fp_wid
        self.fp_sigma = fp_sigma
        self.fp_min_noise = fp_min_noise
        self.group_wid = group_wid
        self.group_sigma = group_sigma


    def process_mgf_files(self):
        """
        Outer multi-process: Handles multiple MGF files in the folder, and each file is processed in parallel by the inner process pool for pepmass.
        """
        # Obtain all MGF files in the folder
        mgf_files = [os.path.join(self.mgf_folder, file) for file in os.listdir(self.mgf_folder) if
                     file.endswith(".mgf")]

        with ProcessPoolExecutor(max_workers=self.outer_max_workers) as executor:
            futures = []
            for mgf_file in mgf_files:
                futures.append(executor.submit(self.process_mgf_file, mgf_file))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f'Error processing MGF file: {exc}')

    def process_mgf_file(self, mgf_file):
        """
        Process a single MGF file, perform operations such as database construction, peak extraction, and peak grouping.
        """
        print(f"Processing MGF file: {mgf_file}")
        
        # Create the database construction class and build the database
        os.makedirs(os.path.join(self.out_dir, 'db'), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'result'), exist_ok=True)



        product_ion_db = os.path.join(self.out_dir, 'db', f'{os.path.basename(mgf_file)}_product_ion.db')

        self.db_builder = DatabaseBuilder(mgf_file, product_ion_db, noise_threshold=200)
        if not os.path.exists(product_ion_db):
            self.db_builder.build_db()

        # Suppose the required DataFrames (intensity_df, mz_df, rt_df) have been extracted from the database.
        # Now, we are processing multiple pepmasses in each MGF file.
        total_peak_df, total_peak_group_df = self.process_pepmass_in_file(mgf_file, product_ion_db)
        total_peak_df["height/width"] = total_peak_df["apex_raw_intensity"] / total_peak_df["peak_width"]

        # Generate the mass spectra for each peak group
        total_peak_df = total_peak_df[total_peak_df['apex_raw_intensity'] != 0]
        fr = ResultFormatter(total_peak_df)
        total_peak_group_df = fr.format_results()

        total_peak_df.to_csv(os.path.join(self.out_dir, 'result', f'{os.path.basename(os.path.splitext(mgf_file)[0])}_product_ion_peak_df.csv'))
        total_peak_group_df.to_csv(os.path.join(self.out_dir, 'result', f'{os.path.basename(os.path.splitext(mgf_file)[0])}_peak_group_df.csv'))

    def process_pepmass_in_file(self, mgf_file, product_ion_db):
        """
        Inner multi-process: Handles multiple pepmasses within each MGF file.
        """
        print(f"Processing pepmass in MGF file: {mgf_file}")

        # The pepmass list was extracted from the database
        pepmass_list = self.db_builder.get_pepmass_list(product_ion_db)

        total_peak_df = pd.DataFrame(
            columns=["apex_index", "apex_raw_intensity", "apex_guass_intensity", "apex_raw_mz", "apex_RT",
                     "accurate_mz", "left_rt", "right_rt", "SV", "pepmass", "group", "noise", "SNR", "peak_points",
                     "peak_width", "prominences"])
        total_peak_group_df = pd.DataFrame(columns=["pepmass", "index", "RT"])

        # Use the inner ProcessPoolExecutor to process pepmass_list in parallel
        with ProcessPoolExecutor(max_workers=self.inner_max_workers) as executor:
            future_to_pepmass = {executor.submit(self.process_pepmass, pepmass, product_ion_db): pepmass for pepmass in pepmass_list}
            for future in as_completed(future_to_pepmass):
                pepmass = future_to_pepmass[future]
                try:
                    peak_df, peak_group_df = future.result()
                    total_peak_df = pd.concat([total_peak_df, peak_df], ignore_index=True)
                    total_peak_group_df = pd.concat([total_peak_group_df, peak_group_df], ignore_index=True)
                except Exception as exc:
                    print(f'An exception occurred while processing pepmass {pepmass}: {exc}')

        return total_peak_df, total_peak_group_df

    def process_pepmass(self, pepmass, product_ion_db):
        """
        Process the data of each pepmass, perform peak extraction, deduplication, calculation and other operations.
        """
        print(f"Processing pepmass {pepmass}...")

        intensity_df, mz_df, rt_df = self.db_builder.merge_data(product_ion_db, pepmass)
        intensity_df, mz_df, rt_df = self.getRTwindow(intensity_df, mz_df, rt_df, self.RT_start, self.RT_end)
        peak_processor = PeakProcessor(intensity_df, mz_df, rt_df, wid=self.fp_wid, sigma=self.fp_sigma, min_noise=self.fp_min_noise, pepmass=pepmass)
        peak_df = peak_processor.find_peak()
        peak_df, peak_group_df = peak_processor.group_peak(peak_df, rt_df, self.group_wid, self.group_sigma, pepmass)

        print(f"Completed processing pepmass {pepmass}")
        # print(peak_df)
        # print(peak_df.shape)
        # print(peak_group_df)
        # print(peak_group_df.shape)
        return peak_df, peak_group_df

    def getRTwindow(self, intensity_df, mz_df, rt_df, min_RT, max_RT):
        condition = (rt_df.iloc[0] > min_RT) & (rt_df.iloc[0] < max_RT)
        selected_columns = rt_df.columns[condition].tolist()
        intensity_df = intensity_df[selected_columns]
        mz_df = mz_df[selected_columns]
        rt_df = rt_df[selected_columns]

        return intensity_df, mz_df, rt_df