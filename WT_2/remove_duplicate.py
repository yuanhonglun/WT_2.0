import os

import numpy as np
import pandas as pd
from WT_2.utils import weighted_dot_product_distance, split_mz_intensity_row, filter_intensity_neg, filter_unique_mz
from WT_2.hrms1 import HrMs1Predictor
from WT_2.msp_processor import MspGenerator
import logging

class Deduplicator:
    def __init__(self, peak_result_dir, msdial_out_path, sample_name, useHrMs1=False, HrMs1model_path=None):
        """
        :param peak_result_dir: feature提取完成后的result目录
        :param msdial_out_path: MSDIAL导出文件

        """
        try:
            self.peak_result_dir = peak_result_dir
            self.msdial_out_path = msdial_out_path
            self.useHrMs1 = useHrMs1
            self.sample_name = sample_name
            self.HrMs1model_path = HrMs1model_path

        except Exception as e:
            logging.error(
                f"Error initializing Deduplicator with peak_result_dir={peak_result_dir} and msdial_out_path={msdial_out_path}: {e}")
            raise



    def remove_msdial_duplicate(self):

        """
            去除MSDIAL中P1P2的重复，主接口
        """

        try:

            peak_group_files = [os.path.join(self.peak_result_dir, file)for file in os.listdir(self.peak_result_dir) if file.endswith('peak_group_df.csv')]
            self.all_clean_p3 = pd.DataFrame()
            self.all_ms2_peak = pd.DataFrame()

            for file in peak_group_files:
                file_name = os.path.basename(file).split('_peak_group_df')[0]
                print("processing", file_name)
                product_ion_peak = os.path.join(self.peak_result_dir, f'{file_name}_product_ion_peak_df.csv')
                clean_p3 = self.all_remove_type1(self.msdial_out_path, file, product_ion_peak, file_name)
                self.all_clean_p3 = pd.concat([self.all_clean_p3, clean_p3])
                #print(file_name, "一二级匹配完成")

                ms2_peak = pd.read_csv(product_ion_peak, sep=",", index_col=0)
                self.all_ms2_peak = pd.concat([self.all_ms2_peak, ms2_peak])

            self.all_clean_p3.to_csv(os.path.join(self.peak_result_dir, "all_p3.csv"))
            self.all_ms2_peak.to_csv(os.path.join(self.peak_result_dir, "all_ms2_peak.csv"))
            #print("全部一二级匹配完成")
        except Exception as e:
            logging.error(f"Error while removing MSDIAL duplicates: {e}")
            raise  # 重新抛出异常

    def filter_p3_group(self):

        '''
        功能：得到所有的p3group后，只保留peak_df中有效的碎片信息。然后筛选出所有的大于3碎片的group
        '''


        try:

            valid_groups = set(self.all_clean_p3.groupby(['pepmass', 'group']).groups.keys())
            all_real_p3_peaks = self.all_ms2_peak.groupby(['pepmass', 'group']).filter(
                lambda x: (x.name in valid_groups))
            good_real_p3_peaks = all_real_p3_peaks.groupby(['pepmass', 'group']).filter(lambda x: len(x) >= 3)
            num_groups_filtered = good_real_p3_peaks.groupby(['pepmass', 'group']).ngroups

            p3_peak_outpath = os.path.join(self.peak_result_dir, self.sample_name + "_good_real_p3_peaks.csv")
            good_real_p3_peaks.to_csv(p3_peak_outpath)
            # fr = ResultFormatter(good_real_p3_peaks, expended=False)
            # new_format_good_real_p3_group = fr.format_results()
            # new_format_good_real_p3_group.to_csv(os.path.join(self.peak_result_dir, "new_format_good_real_p3_group.csv"))

            if self.useHrMs1:
                new_format_good_real_p3_group = self.all_clean_p3[self.all_clean_p3['count'] >= 3][
                    ["pepmass", "group", "apex_RT_mean", "count", "formatted_info", "HR_pepmass"]]
            else:
                new_format_good_real_p3_group = self.all_clean_p3[self.all_clean_p3['count'] >= 3][
                    ["pepmass", "group", "apex_RT_mean", "count", "formatted_info"]]

            p3_group_outpath = os.path.join(self.peak_result_dir, self.sample_name + "_new_format_good_real_p3_group.csv")
            new_format_good_real_p3_group.to_csv(p3_group_outpath)


            #print(f'{self.peak_result_dir} p3 ion 大于3的feature 数量为{num_groups_filtered}')
            return p3_peak_outpath, p3_group_outpath

        except Exception as e:
            logging.error(
                f"Error while filter p3 group: {e}")
            raise



    def all_remove_type1(self, Q1_file, raw_peak_group_file, ms2_peak_df, file_name):
        """
            先去同位素再匹配MSDIAL一级，留下P3
        """
        try:
            ms1_peak = pd.read_csv(Q1_file, sep=",", index_col=0)
            ms1_peak.dropna(subset=['MSMS spectrum'], inplace=True)
            ms2_group = pd.read_csv(raw_peak_group_file, sep=",", index_col=0)
            ms2_peak = pd.read_csv(ms2_peak_df, index_col=0)

            if self.useHrMs1:
                mg = MspGenerator(ms2_group, os.path.join(self.peak_result_dir, f"{file_name}_all_tmp.msp"))
                predictions_list = HrMs1Predictor(os.path.join(self.peak_result_dir, f"{file_name}_all_tmp.msp"), self.HrMs1model_path)
                ms2_group["HR_pepmass"] = predictions_list

                try:
                    os.remove(os.path.join(self.peak_result_dir, f"{file_name}_all_tmp.msp"))
                except Exception as e:
                    pass

            ms2_group = self.new_isotope_removal(ms2_group, ms2_peak)
            ms2_peak_group = self.match_ms1_ms2(ms2_group, ms1_peak, ms2_peak)

            # 删除RT为0的行
            indexdel = ms2_peak_group[ms2_peak_group["RT (min)"] != 0].index
            ms2_peak_group.drop(indexdel, inplace=True)
            ms2_peak_group.dropna(subset=['pepmass'], inplace=True)

            ms2_peak_group.to_csv(os.path.join(self.peak_result_dir, f"{file_name}_p3.csv"))
            return ms2_peak_group

        except Exception as e:
            logging.error(f"Error while processing type1 removal for {file_name}: {e}")
            raise  # 重新抛出异常

    def new_isotope_removal(self, ms2_peak_group, ms2_peak_df):
        '''
        功能：去同位素

        找X的潜在同位素group时，以mz±6，RT±0.05为切片，找到潜在同位素group，如果没有group则continue
        切片中删掉完整离子数量大于X完整离子数量的group，删后无group则continue
        X先取最大响应20%以上的离子，如果离子数量大于10则只取前10个
        这些离子去剩余group中算占比得分，计算时，X有100，Y有100或101，都算匹配上，Y的离子不删减，为完整离子
        占比得分=匹配上离子数/参与计算的X的离子数
        得到匹配得分后，判断参与计算离子数是否小于等于5，若是，则阈值为0.6，否则阈值为0.5
        大于等于阈值，认为该group为X的同位素

        '''

        try:
            if self.useHrMs1:
                col_name = "HR_pepmass"
            else:
                col_name = "pepmass"

            del_list = []

            for index, row in ms2_peak_group.iterrows():
                if index not in del_list:
                    query_peak_RT = row["apex_RT_mean"]
                    query_peak_index = row["group"]
                    query_peak_pepmass = row[col_name]

                    query_peak_df = ms2_peak_df[
                        (ms2_peak_df["pepmass"] == query_peak_pepmass) &
                        (ms2_peak_df["group"] == query_peak_index)
                        ]

                    if query_peak_df.empty:
                        continue  # Skip if there are no peaks matching the query

                    query_peak_product_ions = self.get_more_max20_ions(query_peak_df)[:10]  # Get up to 10 product ions
                    condition = (
                            (abs(ms2_peak_group['apex_RT_mean'] - query_peak_RT) <= 0.05) &
                            (abs(ms2_peak_group[col_name] - query_peak_pepmass) <= 6)
                    )
                    son_df = ms2_peak_group[condition]

                    for i, target_row in son_df.iterrows():
                        target_peak1_pepmass = target_row["pepmass"]
                        target_peak1_index = target_row["group"]

                        target_peak1_product_ions = list(
                            ms2_peak_df[(ms2_peak_df["pepmass"] == target_peak1_pepmass) &
                                        (ms2_peak_df["group"] == target_peak1_index)]["accurate_mz"]
                        )

                        if query_peak_df.shape[0] > len(target_peak1_product_ions):
                            corresponding_ions_ratio = self.get_corresponding_ions(query_peak_product_ions,
                                                                                   target_peak1_product_ions)

                            threshold = 0.6 if len(query_peak_product_ions) <= 5 else 0.5
                            if corresponding_ions_ratio > threshold:
                                del_list.append(i)

            return ms2_peak_group.drop(del_list)

        except Exception as e:
            logging.error(f"Error in isotope removal process: {e}")
            raise  # 重新抛出异常

    def match_ms1_ms2(self, ms2_peak_group, ms1_peak, ms2_peak):

        '''
        功能: 匹配WT2 group与MSDIAL group

        匹配的时候，如果WT2 group与MSDIAL group的交集ion数量/WT2 group数量 > 0.5，或交集ion数量/MSDIAL group ion数量 > 0.5
        离子数量小于等于5，阈值提高为0.6
        才认为是对上

        与P1匹配的时候，先根据mz和RT切片，得到待比较的group（如果无group，直接作为P3），计算上述的相似性，取得分最高的group卡上，如果多个group得分相同，算反向余弦相似性（谁ion少以谁为基准），取得分最高的group卡上
        卡的时候阈值是上述阈值
        '''

        try:
            match_result = []

            for line in ms2_peak_group.itertuples():
                pepmass, RT, group = line[1], line[3], line[2]
                ms1_peak_filtered = ms1_peak[
                    (abs(ms1_peak['RT (min)'] * 60 - RT) <= 6) &
                    (abs(ms1_peak['Precursor m/z'] - pepmass) <= 0.5)
                    ]
                ms2_peaks = ms2_peak[
                    (ms2_peak["pepmass"] == pepmass) &
                    (ms2_peak["group"] == group)
                    ]
                ms2_mz = np.array(ms2_peaks["accurate_mz"])

                if ms1_peak_filtered.shape[0] != 0 and ms2_peaks.shape[0] != 0:
                    max_score = -1
                    scores = []

                    for index, row in ms1_peak_filtered.iterrows():
                        ms1_peaks_df = split_mz_intensity_row(row)
                        ms1_mz = np.array(ms1_peaks_df["mz"])

                        _, _, common_mz_num, _ = filter_unique_mz(ms2_mz, ms1_mz)

                        score1 = common_mz_num / len(ms1_mz) if len(ms1_mz) > 0 else 0
                        score2 = common_mz_num / len(ms2_mz) if len(ms2_mz) > 0 else 0

                        if len(ms1_mz) <= 5 or len(ms2_mz) <= 5:
                            threshold = 0.6
                        else:
                            threshold = 0.5

                        if score1 >= threshold or score2 >= threshold:
                            if max(score1, score2) > max_score:
                                max_score = max(score1, score2)
                                scores = [(max_score, row, ms1_mz, np.array(ms1_peaks_df["intensity"]))]

                    if scores:
                        if len(scores) == 1:
                            match_result.append(list(scores[0][1].values))  # 直接加入唯一匹配行
                        else:
                            highest_cosine_score = -1
                            best_row = None
                            ms2_intensity = np.array(ms2_peaks["apex_raw_intensity"])

                            for max_score, row, ms1_mz, ms1_intensity in scores:
                                if len(ms1_mz) <= len(ms2_mz):
                                    ms2_intensity_new = filter_intensity_neg(ms2_mz, ms2_intensity, ms1_mz,
                                                                             ms1_intensity)
                                    cosine = weighted_dot_product_distance(ms1_intensity, ms2_intensity_new, ms1_mz)
                                else:
                                    ms1_intensity_new = filter_intensity_neg(ms1_mz, ms1_intensity, ms2_mz,
                                                                             ms2_intensity)
                                    cosine = weighted_dot_product_distance(ms1_intensity_new, ms2_intensity, ms2_mz)

                                if cosine > highest_cosine_score:
                                    highest_cosine_score = cosine
                                    best_row = row

                            if best_row is not None:
                                match_result.append(list(row.values))  # 将最佳行添加到匹配结果
                    else:
                        match_result.append([0] * ms1_peak.shape[1])  # 如果没有匹配，填充零行
                else:
                    match_result.append([0] * ms1_peak.shape[1])  # 没有匹配行，填充零行

            ms2_peak_group = pd.concat([ms2_peak_group, pd.DataFrame(match_result)], axis=1)
            if self.useHrMs1:
                ms2_peak_group.columns = ["pepmass", "group", "apex_RT_mean", "count", "formatted_info", "HR_pepmass"] + list(ms1_peak.columns)
            else:
                ms2_peak_group.columns = ["pepmass", "group", "apex_RT_mean", "count", "formatted_info"] + list(ms1_peak.columns)

            return ms2_peak_group

        except Exception as e:
            logging.error(f"Error in matching MS1 and MS2 peaks: {e}")
            raise  # 重新抛出异常

    def get_more_max20_ions(self, ms2_peak_df):
        max_value = ms2_peak_df['apex_raw_intensity'].max()
        threshold = max_value * 0.2
        indexes = ms2_peak_df[ms2_peak_df['apex_raw_intensity'] >= threshold].index
        result = ms2_peak_df.loc[indexes, 'accurate_mz']

        return list(result)

    def get_corresponding_ions(self, list1, list2):
        count = 0
        for num1 in list1:
            c1 = False
            c2 = False
            for num2 in list2:
                if abs(num1 - num2) <= 0.02:
                    c1 = True
                elif abs(num1 + 1.0033 - num2) <= 0.02:
                    c2 = True
                elif c1 & c2:
                    count += 1
                    break
        return count / len(list1)



class ResultFormatter:
    def __init__(self, df, expended=False):
        """
        设定结果文件格式

        :param df: 来自Deduplicator去重后的df
        """
        try:
            self.df = df
            self.expended = expended
        except Exception as e:
            logging.error(
                f"Error initializing ResultFormatter with df={df}: {e}")
            raise

    def format_results(self):
        """
        整理分组结果，生成新的DataFrame，包含格式化的准确m/z和强度信息。
        """
        grouped = self.df.groupby(['pepmass', 'group']).agg(
            apex_RT_mean=('apex_RT', 'mean'),
            accurate_mz_list=('accurate_mz', lambda x: list(x)),
            apex_raw_intensity_list=('apex_raw_intensity', lambda x: list(x)),
            count=('apex_RT', 'size')
        ).reset_index()

        result_rows = []
        for _, row in grouped.iterrows():
            accurate_mz = row['accurate_mz_list']
            apex_raw_intensity = row['apex_raw_intensity_list']

            new_row = {
                'pepmass': row['pepmass'],
                'group': row['group'],
                'apex_RT_mean': row['apex_RT_mean'],
                'count': row['count']
            }
            new_row['formatted_info'] = self._generate_formatted_info(accurate_mz,
                                                                apex_raw_intensity)  # 为了生成Q3:Q3_intensity Q4:Q4_intensity一列

            if self.expended:
                new_row = self._generate_ourself_info(new_row, accurate_mz,
                                                apex_raw_intensity)  # 为了生成Q3\tQ3_intensity\tQ4\tQ4_intensity多列

            result_rows.append(new_row)

        result_df = pd.DataFrame(result_rows)
        return result_df

    def _generate_ourself_info(self, new_row, accurate_mz, apex_raw_intensity):
        '''
                功能：对p3 group生成类似Q3,Q3_intensity,Q4,Q4_intensity......格式表格
        '''

        sorted_indices = sorted(range(len(apex_raw_intensity)), key=lambda i: apex_raw_intensity[i], reverse=True)

        for index in range(len(sorted_indices)):
            i = sorted_indices[index]
            new_row[f'Q{index + 3}'] = accurate_mz[i]
            new_row[f'Q{index + 3}_intensity'] = apex_raw_intensity[i]

        return new_row

    def _generate_formatted_info(self, accurate_mz, apex_raw_intensity):
        '''
            功能：对p3 group生成类似MSDIAL格式的表格
        '''

        formatted_info = []
        sorted_indices = sorted(range(len(apex_raw_intensity)), key=lambda i: apex_raw_intensity[i], reverse=True)

        for index in range(len(sorted_indices)):
            i = sorted_indices[index]
            formatted_info.append(f"{accurate_mz[i]} {apex_raw_intensity[i]}")

        return ';'.join(formatted_info)


# a = Deduplicator(r"D:\work\WT2.0\WT_2_test\test_data\sample1\result", r"D:\work\WT2.0\WT_2_test\test_data\sample1\CRL_SIF_1_Q1_peak_df.csv", "sample1", useHrMs1=True, HrMs1model_path=r'D:\work\WT2.0\WT_2_test\test_data\model\HrMs1.pth')
# a.remove_msdial_duplicate()
# peak_outpath, group_outpath = a.filter_p3_group()

# df = pd.read_csv(r"D:\work\WT2.0\WT_2\test_data\result\CRL_SIF_P_075-174_1_product_ion_peak_df.csv")
# a = ResultFormatter(df)
# df1 = a.format_results()
# df1.to_csv(r"D:\work\WT2.0\WT_2\test_data\result\t.csv")