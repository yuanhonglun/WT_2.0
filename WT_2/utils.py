import numpy as np
import pandas as pd

def dot_product_distance(p, q):
    # Calculate the dot product distance
    if np.sum(p) == 0 or np.sum(q) == 0:
        score = 0
    else:
        score = np.power(np.sum(q * p), 2) / (np.sum(np.power(q, 2)) * np.sum(np.power(p, 2)))
    return score


def weighted_dot_product_distance(i_q, i_r, m_q):
    # Weighted Dot Product Distance
    k = 0.5
    l = 1
    w_q = np.power(i_q, k) * np.power(m_q, l)
    w_r = np.power(i_r, k) * np.power(m_q, l)
    score = dot_product_distance(w_q, w_r)
    return score


def split_mz_intensity_row(row):
    # Function: Split the MSMS spectrum of the P1 result from MSDIAL, and return a df with three columns: mz, intensity, and RT.

    # Split Specified Column
    ms_values = row['MSMS spectrum'].split(';')  # 拆分 MSMS spectrum 列

    rt_value = float(row['RT (min)'])


    split_values = np.array([value.split() for value in ms_values])


    mz_values = split_values[:, 0].astype(float)
    intensity_values = split_values[:, 1].astype(float)


    new_df = pd.DataFrame({
        'mz': mz_values,
        'intensity': intensity_values,
        'RT': rt_value
    })

    return new_df


def filter_unique_mz(query_mz, library_mz, threshold=0.02):
    # Function: Return the unique mz values of query_mz, the unique mz values of library_mz, and the common mz values of both,
    # considering two mz values as the same if their difference is within 0.02.


    query_unique_mz = set(query_mz)
    library_unique_mz = set(library_mz)
    common_mz = set()


    for m in library_mz:
        idx = np.argmin(np.abs(query_mz - m))
        if np.abs(query_mz[idx] - m) <= threshold:

            common_mz.add(m)
            query_unique_mz.discard(query_mz[idx])
            library_unique_mz.discard(m)

    return len(query_unique_mz), len(library_unique_mz), len(common_mz), list(common_mz)

def filter_intensity_neg(query_mz, query_intensity, library_mz, library_intensity, threshold=0.02):
    # Function: Based on the mz value of the library, return new_query_intensity1 (reverse cosine similarity)

    new_query_intensity1 = np.zeros_like(library_intensity)
    for i, m in enumerate(library_mz):
        idx = np.argmin(np.abs(query_mz - m))
        if np.abs(query_mz[idx] - m) <= threshold:
            new_query_intensity1[i] = query_intensity[idx]
    return new_query_intensity1


# 解析 mz 和强度
def parse_mz_intensity(s):
    """Parse the string-formatted mz intensity data into two lists"""
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
    Filter the MS2 spectra based on Max% and absolute response values
    '''
    mz_values, intensity_values = zip(*[map(float, pair.split(' ')) for pair in data.split(";")])
    mz_values = list(mz_values)
    intensity_values = list(intensity_values)

    max_intensity = max(intensity_values)
    threshold = 0.01 * percent * max_intensity
    threshold = max(threshold, absolute_intensity)

    filtered_mz = [mz for mz, intensity in zip(mz_values, intensity_values) if intensity >= threshold]

    return filtered_mz