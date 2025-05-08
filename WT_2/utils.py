import numpy as np
import pandas as pd

def dot_product_distance(p, q):
    # 计算点积距离
    if np.sum(p) == 0 or np.sum(q) == 0:
        score = 0
    else:
        score = np.power(np.sum(q * p), 2) / (np.sum(np.power(q, 2)) * np.sum(np.power(p, 2)))
    return score


def weighted_dot_product_distance(i_q, i_r, m_q):
    # 加权点积距离
    k = 0.5
    l = 1
    w_q = np.power(i_q, k) * np.power(m_q, l)
    w_r = np.power(i_r, k) * np.power(m_q, l)
    score = dot_product_distance(w_q, w_r)
    return score


def split_mz_intensity_row(row):
    #功能：拆分MSDIAL的P1结果的MSMS spectrum，返回df，三列mz、intenstiy、RT

    # 拆分指定列
    ms_values = row['MSMS spectrum'].split(';')  # 拆分 MSMS spectrum 列

    rt_value = float(row['RT (min)'])  # 获取 RT 值并转换为 float

    # 使用 NumPy 创建一个二维数组
    split_values = np.array([value.split() for value in ms_values])

    # 提取 mz 和 intensity，并转换为 float
    mz_values = split_values[:, 0].astype(float)
    intensity_values = split_values[:, 1].astype(float)

    # 创建新的 DataFrame
    new_df = pd.DataFrame({
        'mz': mz_values,
        'intensity': intensity_values,
        'RT': rt_value
    })

    return new_df


def filter_unique_mz(query_mz, library_mz, threshold=0.02):
    #功能：返回query_mz特有的mz, library_mz特有的mz，两者共有的mz，阈值为0.02认为是一个mz

    # 初始化集合
    query_unique_mz = set(query_mz)  # 初始化为query的所有m/z
    library_unique_mz = set(library_mz)  # 初始化为library的所有m/z
    common_mz = set()  # 初始化交集集合

    # 遍历库中的m/z并与query中的m/z进行比较
    for m in library_mz:
        idx = np.argmin(np.abs(query_mz - m))
        if np.abs(query_mz[idx] - m) <= threshold:
            # 如果匹配成功，加入交集，并从特异集合中删除
            common_mz.add(m)
            query_unique_mz.discard(query_mz[idx])  # 从query的特异集合中移除
            library_unique_mz.discard(m)  # 从library的特异集合中移除

    return len(query_unique_mz), len(library_unique_mz), len(common_mz), list(common_mz)

def filter_intensity_neg(query_mz, query_intensity, library_mz, library_intensity, threshold=0.02):
    #功能：以库的mz为基准，返回new_query_intensity1（反向余弦相似度）

    new_query_intensity1 = np.zeros_like(library_intensity)
    for i, m in enumerate(library_mz):
        idx = np.argmin(np.abs(query_mz - m))
        if np.abs(query_mz[idx] - m) <= threshold:
            new_query_intensity1[i] = query_intensity[idx]
    return new_query_intensity1


# 解析 mz 和强度
def parse_mz_intensity(s):
    """将字符串格式的 mz intensity 解析为两个列表"""
    mz_values = []
    intensity_values = []

    pairs = s.split(';')

    for pair in pairs:
        mz, intensity = pair.split()
        mz_values.append(float(mz))
        intensity_values.append(float(intensity))

    return np.array(mz_values), np.array(intensity_values)



def filter_ms2_spectra(data: str, percent=10, absolute_intensity=200):
    '''
    根据Max%和绝对响应值过滤一下MS2 spectra
    '''
    mz_values, intensity_values = zip(*[map(float, pair.split(' ')) for pair in data.split(";")])
    mz_values = list(mz_values)
    intensity_values = list(intensity_values)

    max_intensity = max(intensity_values)
    threshold = 0.01 * percent * max_intensity
    threshold = max(threshold, absolute_intensity)

    filtered_mz = [mz for mz, intensity in zip(mz_values, intensity_values) if intensity >= threshold]

    return filtered_mz