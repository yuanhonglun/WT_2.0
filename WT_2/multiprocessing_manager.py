import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from WT_2.database_builder import DatabaseBuilder
from WT_2.peak_processor import PeakProcessor
from WT_2.remove_duplicate import ResultFormatter


class MultiprocessingManager:
    def __init__(self, outer_max_workers, inner_max_workers, mgf_folder, out_dir, RT_start=30, RT_end=720, fp_wid=6, fp_sigma=2, fp_min_noise=200, group_wid=6, group_sigma=0.5):
        """
        初始化多进程管理类

        :param outer_max_workers: 外层进程池最大工作进程数（处理多个MGF文件）
        :param inner_max_workers: 内层进程池最大工作进程数（处理文件夹内多个pepmass）
        :param mgf_folder: 单个样本文件夹路径，文件夹下包括多个MGF文件
        :param product_ion_db: 产品离子数据库路径
        :param out_dir: 输出路径
        """
        self.outer_max_workers = outer_max_workers  # 外层进程池最大工作进程数
        self.inner_max_workers = inner_max_workers  # 内层进程池最大工作进程数
        self.mgf_folder = mgf_folder  # 文件夹路径
        self.out_dir = out_dir  # 输出路径

        #峰识别参数
        self.RT_start = RT_start
        self.RT_end = RT_end
        self.fp_wid = fp_wid
        self.fp_sigma = fp_sigma
        self.fp_min_noise = fp_min_noise
        self.group_wid = group_wid
        self.group_sigma = group_sigma


    def process_mgf_files(self):
        """
        外层多进程：处理文件夹下的多个 MGF 文件，每个文件通过内层进程池并行处理 pepmass。
        """
        # 获取文件夹中的所有 MGF 文件
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
        处理单个 MGF 文件，进行数据库构建、峰值提取、峰值分组等操作。
        """
        print(f"Processing MGF file: {mgf_file}")
        
        # 创建数据库构建类并构建数据库
        os.makedirs(os.path.join(self.out_dir, 'db'), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'result'), exist_ok=True)



        product_ion_db = os.path.join(self.out_dir, 'db', f'{os.path.basename(mgf_file)}_product_ion.db')

        self.db_builder = DatabaseBuilder(mgf_file, product_ion_db, noise_threshold=200)
        if not os.path.exists(product_ion_db):
            self.db_builder.build_db()

        # 假设从数据库中提取了需要的 DataFrame（intensity_df, mz_df, rt_df）
        # 现在处理每个 MGF 文件中的多个 pepmass
        total_peak_df, total_peak_group_df = self.process_pepmass_in_file(mgf_file, product_ion_db)
        total_peak_df["height/width"] = total_peak_df["apex_raw_intensity"] / total_peak_df["peak_width"]

        # 生成每个peak group的质谱图
        total_peak_df = total_peak_df[total_peak_df['apex_raw_intensity'] != 0]
        fr = ResultFormatter(total_peak_df)
        total_peak_group_df = fr.format_results()

        total_peak_df.to_csv(os.path.join(self.out_dir, 'result', f'{os.path.basename(os.path.splitext(mgf_file)[0])}_product_ion_peak_df.csv'))
        total_peak_group_df.to_csv(os.path.join(self.out_dir, 'result', f'{os.path.basename(os.path.splitext(mgf_file)[0])}_peak_group_df.csv'))

    def process_pepmass_in_file(self, mgf_file, product_ion_db):
        """
        内层多进程：处理每个 MGF 文件中的多个 pepmass。
        """
        print(f"Processing pepmass in MGF file: {mgf_file}")

        # 从数据库中提取了 pepmass 列表
        pepmass_list = self.db_builder.get_pepmass_list(product_ion_db)

        total_peak_df = pd.DataFrame(
            columns=["apex_index", "apex_raw_intensity", "apex_guass_intensity", "apex_raw_mz", "apex_RT",
                     "accurate_mz", "left_rt", "right_rt", "SV", "pepmass", "group", "noise", "SNR", "peak_points",
                     "peak_width", "prominences"])
        total_peak_group_df = pd.DataFrame(columns=["pepmass", "index", "RT"])

        # 使用内层 ProcessPoolExecutor 并行处理 pepmass_list
        with ProcessPoolExecutor(max_workers=self.inner_max_workers) as executor:
            future_to_pepmass = {executor.submit(self.process_pepmass, pepmass, product_ion_db): pepmass for pepmass in pepmass_list}
            for future in as_completed(future_to_pepmass):
                pepmass = future_to_pepmass[future]
                try:
                    peak_df, peak_group_df = future.result()
                    total_peak_df = pd.concat([total_peak_df, peak_df], ignore_index=True)
                    total_peak_group_df = pd.concat([total_peak_group_df, peak_group_df], ignore_index=True)
                except Exception as exc:
                    print(f'处理 pepmass {pepmass} 时发生异常: {exc}')

        return total_peak_df, total_peak_group_df

    def process_pepmass(self, pepmass, product_ion_db):
        """
        处理每个 pepmass 的数据，进行峰值提取、去重、计算等操作。
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