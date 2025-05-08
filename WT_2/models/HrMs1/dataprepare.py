import random

import numpy as np
import pickle
import scipy.sparse as sp
from matchms.importing import load_from_msp

def process_spectra_to_pickle(msp_file, pickle_file, mz_min=50, mz_max=1076, bin_width=0.02, batch_size=10000, normaliz='NA', distrub=False, calculate=True):
    # 创建 bins
    bins = np.round(np.arange(mz_min, mz_max + bin_width, bin_width), 2)

    # 初始化临时列表
    batch_data = []
    batch_labels = []

    # 创建存储数据的字典
    all_data = {
        'intensities': None,  # 初始化为 None，后续会进行合并
        'labels': [],
        'mz_bins': np.append(bins[:-1], [0, 1])  # 只存 bin 下界
    }

    # 逐个光谱处理
    for i, spectrum in enumerate(load_from_msp(msp_file)):
        mz_values = spectrum.mz
        raw_intensities = spectrum.intensities
        max_value = max(raw_intensities)

        if max_value == 0:
            print("质谱图最大响应为0")
            continue
        intensities = [x / max_value for x in raw_intensities]
        precursor_mz = spectrum.metadata.get('precursor_mz', None)

        if precursor_mz is None:
            continue  # 跳过没有 precursor_mz 的光谱

        Q1 = round(precursor_mz, 1)
        if len(str(mz_values[0]).split('.')[-1]) >= 1:

            # 初始化稀疏数据用于当前光谱
            sparse_data = np.zeros(len(bins) + 1)

            # 使用 digitize 函数将 mz 值分类到 bins
            indices = np.digitize(mz_values, bins)

            # 累加当前光谱的 intensities
            for index, intensity in zip(indices, intensities):
                if 0 < index < len(bins):
                    sparse_data[index - 1] += intensity

            if normaliz == 'log2':
                sparse_data[:-1] = np.where(sparse_data[:-1] > 0, np.log2(sparse_data[:-1]), 0)  # 只对 intensities 部分进行标准化

            # 将 precursor_mz 加入最后一维
            sparse_data[-2] = Q1 - 0.5  # 将 低分辨 Q1-0.5 存放

            if distrub:
                disturbance = random.uniform(-0.5, 0.5)
                sparse_data[-1] = round(Q1 + 0.5 + disturbance) # 将 低分辨 Q1+0.5 存放在最后一维
            elif calculate:
                sparse_data[-1] = round(round(Q1, 0)*1.0006 + 0.0099, 1) + 0.5

            else:
                sparse_data[-1] = Q1 + 0.5  # 将 低分辨 Q1+0.5 存放在最后一维


            # 将累积的结果加入批量数据
            batch_data.append(sparse_data)
            batch_labels.append(precursor_mz)

            # 一定数量的光谱后处理数据
            if (i + 1) % batch_size == 0:
                # 将当前批量的稀疏矩阵添加到 intensities 数据
                batch_sparse_matrix = sp.csr_matrix(batch_data)
                if all_data['intensities'] is None:
                    all_data['intensities'] = batch_sparse_matrix
                else:
                    all_data['intensities'] = sp.vstack([all_data['intensities'], batch_sparse_matrix])

                all_data['labels'].extend(batch_labels)

                batch_data = []  # 清空批量数据以准备下一个批次
                batch_labels = []  # 清空批量标签以准备下一个批次

    # 处理剩余未写入的数据
    if batch_data:
        batch_sparse_matrix = sp.csr_matrix(batch_data)
        if all_data['intensities'] is None:
            all_data['intensities'] = batch_sparse_matrix
        else:
            all_data['intensities'] = sp.vstack([all_data['intensities'], batch_sparse_matrix])
        all_data['labels'].extend(batch_labels)

    # 保存为 pickle 格式
    with open(pickle_file, 'wb') as f:
        pickle.dump(all_data, f)

    #print(f'数据已保存到 {pickle_file}')



#
# process_spectra_to_pickle(r"D:\work\WT2.0\WT_2\test_data\result\test.msp", r"D:\work\WT2.0\WT_2\test_data\result\test.pkl", distrub=False, calculate=True)
#
