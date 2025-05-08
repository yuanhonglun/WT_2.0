import math

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings("ignore")

class PeakProcessor:
    def __init__(self, intensity_df, mz_df, rt_df, wid, sigma, min_noise, pepmass):
        """
        初始化峰值处理类。

        :param intensity_df: 强度数据 DataFrame
        :param mz_df: mz 数据 DataFrame
        :param rt_df: RT 数据 DataFrame
        :param wid: 高斯滤波窗口大小
        :param sigma: 高斯滤波的标准差
        :param min_noise: 最小噪音
        :param pepmass: 离子质量
        """
        self.intensity_df = intensity_df
        self.mz_df = mz_df
        self.rt_df = rt_df
        self.wid = wid
        self.sigma = sigma
        self.min_noise = min_noise
        self.pepmass = pepmass

    def find_peak(self):
        """
        寻找峰值并进行处理。
        """
        sg_list = []
        for n in range(0, self.wid * 2 + 1):
            sg_list.append((1 - (pow(((-self.wid + n) / self.sigma), 2))) * (np.exp(-0.5 * pow(((-self.wid + n) / self.sigma), 2))))

        peak_df = pd.DataFrame(columns=["apex_index", "apex_raw_intensity", "apex_guass_intensity", "apex_raw_mz", "apex_RT", "accurate_mz", "left_rt", "right_rt", "SV", "pepmass", "group", "noise", "SNR", "peak_points", "peak_width", "prominences"])

        for index, row in self.intensity_df.iterrows():
            if row.sum() < 1000:
                continue
            n = 0
            for value in row:
                if value != 0:
                    n = n + 1
                    if n == 3:
                        break
                else:
                    n = 0

            if n < 3:
                continue
            else:
                temp_df = pd.DataFrame(columns=self.intensity_df.columns)
                temp_df.loc['intensity'] = row
                temp_df.loc['intensity'] = temp_df.loc['intensity'].mask(
                    (temp_df.loc['intensity'].shift(1) == 0) & (temp_df.loc['intensity'].shift(-1) == 0), 0)
                temp_df.loc['gauss_intensity'] = temp_df.loc['intensity'].rolling(window=self.wid * 2 + 1,
                                                                                 min_periods=self.wid * 2 + 1,
                                                                                 center=True).apply(
                    lambda x: sum(np.multiply(np.asarray(x), np.asarray(sg_list)))).tolist()
                temp_df.loc["mz"] = self.mz_df.loc[index]
                temp_df.loc["RT"] = self.rt_df.iloc[0]
                noise = self._new_calculate_noise(temp_df)
                noise = noise * 2
                height = max(self.min_noise, noise * 5)
                peaks_apex_index, properties = find_peaks(temp_df.loc['gauss_intensity'], height=height, width=3, rel_height=1)
                # print("peaks = ", peaks_apex_index)
                # print("properties = ", properties)
                # print("*" * 20)
                temp_df = temp_df.T
                temp_df.reset_index(drop=True, inplace=True)

                peaks_left_index = [round(num) for num in properties["left_ips"]]
                peaks_right_index = [round(num) for num in properties["right_ips"]]
                peak_points_list = list(np.array(peaks_right_index) - np.array(peaks_left_index) + 1)
                peak_width_list = list(np.array(properties["right_ips"]) - np.array(properties["left_ips"]))
                prominences = properties["prominences"]
                SV_list = []

                for i in range(len(properties['peak_heights'])):
                    SV = properties['peak_heights'][i] / properties['widths'][i]
                    SV_list.append(SV)

                peak_info_list = []
                for left, apex, right, SV, peak_point, peak_width, p in zip(peaks_left_index, peaks_apex_index, peaks_right_index, SV_list, peak_points_list, peak_width_list, prominences):
                    left_to_right_df = temp_df.iloc[left:right + 1, :]
                    accurate_mz = self._calculate_mz(left_to_right_df)
                    left_rt, right_rt = left_to_right_df.loc[left, "RT"], left_to_right_df.loc[right, "RT"]
                    apex_info = list(left_to_right_df.loc[apex, :])
                    SNR = apex_info[1] / noise if noise != 0 else "NA"
                    peak_info_list.append(
                        [apex] + apex_info + [accurate_mz, left_rt, right_rt, SV, self.pepmass, "NA", noise, SNR, peak_point, peak_width, p])
                peak_info_df = pd.DataFrame(peak_info_list, columns=peak_df.columns)
                peak_df = pd.concat([peak_df, peak_info_df], ignore_index=True)

        peak_df = peak_df[~peak_df.duplicated(keep="first")]
        peak_df = self._remove_duplicated_peaks(peak_df)
        return peak_df

    def _remove_duplicated_peaks(self, df):
        # 先根据 accurate_mz 列的值从小到大排序
        df.sort_values(by="accurate_mz", inplace=True)
        # 初始化要删除的行索引列表
        rows_to_drop = []

        # 遍历 DataFrame
        for index, row in df.iterrows():
            # print("row.name = ", row.name)
            # print("rows_to_drop = ", rows_to_drop)
            if row.name in rows_to_drop:
                continue

            # 找到与本行 accurate_mz 值之差绝对值小于等于 0.02 的行
            similar_rows = df[(abs(df["accurate_mz"] - row["accurate_mz"]) <= 0.02)]

            # 如果只有一行，则继续下一轮循环
            if len(similar_rows) == 1:
                continue
            similar_rows = similar_rows.drop(index)
            # 遍历相似的行，进行后续判断
            for _, similar_row in similar_rows.iterrows():
                # print("len = ", len(similar_rows))
                # print("similar_row = ", similar_row)
                # 判断 RT 之差的绝对值是否小于等于 2
                if abs(row["apex_RT"] - similar_row["apex_RT"]) <= 2:
                    # 判断 apex_raw_intensity 谁大
                    if row["apex_raw_intensity"] >= similar_row["apex_raw_intensity"]:
                        if similar_row.name not in rows_to_drop:
                            rows_to_drop.append(similar_row.name)
                    else:
                        if row.name not in rows_to_drop:
                            rows_to_drop.append(row.name)

        df_cleaned = df.drop(rows_to_drop)

        return df_cleaned


    def _new_calculate_noise(self, df):
        """
        计算噪音值
        """
        df = df.T
        df = df[df['intensity'] != 0]
        noise = df['intensity'].quantile(0.25)
        return noise

    def _calculate_mz(self, df):
        """
        计算精确的 mz 值
        """
        return sum(df["mz"] * df["intensity"]) / sum(df["intensity"])

    def group_peak(self, peak_df, rt_df, wid, sigma, pepmass):
        SV_df = pd.DataFrame(index=range(len(rt_df.columns)), columns=["SV", "gauss_SV"])
        SV_df.fillna(0, inplace=True)
        sg_list = []
        for n in range(0, wid * 2 + 1):
            sg_list.append((1 - (pow(((-wid + n) / sigma), 2))) * (math.exp(-0.5 * pow(((-wid + n) / sigma), 2))))
        for index, row in peak_df.iterrows():
            apex_index = row['apex_index']
            SV = row['SV']
            SV_df.iloc[apex_index, 0] = SV + SV_df.iloc[apex_index, 0]
        SV_df["gauss_SV"] = SV_df["SV"].rolling(window=wid * 2 + 1, min_periods=wid * 2 + 1,
                                                center=True).apply(
            lambda x: sum(np.multiply(np.asarray(x), np.asarray(sg_list)))).to_list()

        peaks_apex_index, properties = find_peaks(SV_df["gauss_SV"], height=0, width=0)
        # SV_df = SV_df.T
        # SV_df.to_csv(f"{pepmass}_SV_df.csv")
        max_index = len(rt_df.columns)
        rt_df = rt_df.T
        peak_group_df = pd.DataFrame(columns=["pepmass", "index", "RT"])
        for index in peaks_apex_index:
            rt_value = rt_df.iloc[index, 0]
            temp_df = pd.DataFrame({"pepmass": [pepmass], "index": [index], "RT": [rt_value]})
            peak_group_df = pd.concat([peak_group_df, temp_df], ignore_index=True)
            pre_index = max(0, index - 1)
            next_index = min(max_index, index + 1)
            temp_index_list = [pre_index, index, next_index]
            for idx, row in peak_df.iterrows():
                if row["apex_index"] in temp_index_list:
                    peak_df.loc[idx, "group"] = index


        return peak_df, peak_group_df

