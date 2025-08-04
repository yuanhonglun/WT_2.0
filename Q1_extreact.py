import os
import re
import sqlite3
import pandas as pd
import numpy as np
import time
import math
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} Execution time: {end_time - start_time} 秒")
        return result

    return wrapper

def getMS1Rt(file_path):
    with open(file_path) as f:
        file = f.read()

    RT_list = [float(x.group().split("\t")[2]) for x in re.finditer("I\tRTime.*$", file, re.MULTILINE)]

    return RT_list



def find_interval(number, interval_size):
    interval_index = int(number / interval_size)
    return interval_index * interval_size


def creatDBTalbe(RT_list, db_path, table_name, file_path):

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA synchronous = OFF;")
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("BEGIN TRANSACTION;")

    df = pd.DataFrame(0, columns=RT_list, index=np.arange(74, 178, 0.02).round(2))
    df.to_sql(table_name, conn, if_exists='replace')

    with open(file_path) as f:
        now_rt = 0

        for line in f:
            if line[0].isalpha():
                if line.split("\t")[1] == "RTime":
                    now_rt = float(line.strip().split("\t")[2])
            elif line[0].isdigit():
                tmp_list = line.strip().split(" ")
                now_Q1 = float(tmp_list[0])
                now_intensity = float(tmp_list[1])
                belong_qujian = round(find_interval(now_Q1, 0.02), 2)

                query_sql = 'SELECT {} FROM Q1_table WHERE "index" = ?'.format([now_rt])
                cursor.execute(query_sql, (belong_qujian,))
                results = cursor.fetchone()
                if results:
                    if results[0] < now_intensity:
                        update_sql = 'UPDATE Q1_table SET {} = ? WHERE "index" = ?'.format([now_rt])
                        cursor.execute(update_sql, (now_intensity, belong_qujian,))

    conn.commit()
    conn.close()


def query_ion_eic(db_path, table_name, ion, RT_list):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query_sql = f'SELECT * FROM {table_name} WHERE "index" = ?'
    cursor.execute(query_sql, (ion,))
    results = cursor.fetchone()
    eic_df = pd.DataFrame({"RT":RT_list, "intensity":list(results[1:])})
    conn.commit()
    conn.close()
    return eic_df


def plot_mz_picture(eic_df, file_name, path, gauss:bool):

    x = eic_df["RT"]
    if gauss:
        y = eic_df["gauss_intensity"]
    else:
        y = eic_df["intensity"]
    plt.figure(figsize=(200, 10), dpi=100)
    plt.plot(x, y, c='red')
    plt.scatter(x, y, c='red')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("RT", fontdict={'size': 16})
    plt.ylabel("Intensity", fontdict={'size': 16})
    plt.title(file_name, fontdict={'size': 20})

    plt.savefig(os.path.join(path, file_name))
    plt.close()


def secondorder_Gaussian_Filter(eic_df, wid, sigma):
    # WTV2.0 find peak
    sg_list = []
    for n in range(0, wid * 2 + 1):
        sg_list.append((1 - (pow(((-wid + n) / sigma), 2))) * (math.exp(-0.5 * pow(((-wid + n) / sigma), 2))))
    
    # print("sg_list = ", sg_list)
    
    eic_df["gauss_intensity"] = eic_df["intensity"].rolling(window=wid * 2 + 1, min_periods=wid * 2 + 1,
                                                    center=True).apply(
        lambda x: sum(np.multiply(np.asarray(x), np.asarray(sg_list)))).to_list()
    
    # eic_df.to_csv("eic_df.csv")
                                                        
    return eic_df

    # scipy find peak
    # from scipy.ndimage import gaussian_filter
    # eic_df["gauss_intensity"] = gaussian_filter(eic_df["intensity"], sigma=1)
    # return eic_df

def find_peak(eic_df, noise):
    gauss_intensity = eic_df["gauss_intensity"]
    peaks, properties = find_peaks(gauss_intensity, height = noise*10, width=3)
    # print("peaks = ", peaks)
    # print("properties = ", properties)
    
    return peaks, properties

def find_peak_plot(peaks, properties, eic_df, file_name, path):
    x = eic_df["gauss_intensity"]
    plt.figure(figsize=(200, 10), dpi=100)
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.vlines(x=peaks, ymin=x[peaks] - properties["prominences"],
               ymax=x[peaks], color="C1")
    plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
              xmax=properties["right_ips"], color="C1")
    plt.savefig(os.path.join(path, file_name))
    plt.close()

def calculate_noise(df, min_noise):
    median_list = []
    for i in range(0, len(df), 13):
        subset = df.iloc[i:i+13]  # Take a subset of 13 rows.
        
        # Check if there is a 0 value in row 13.
        if 0 in subset['intensity'].values:
            continue  # 若有0值则跳过这13行
        
        # If there are no 0 values in row 13, then calculate the average value and the number of rows that are greater than the average value.
        avg_intensity = subset['intensity'].mean()
        above_avg_count = subset[subset['intensity'] > avg_intensity]['intensity'].count()
        
        # If the number of rows that are greater than the average value is equal to or greater than 7, then calculate the median.
        if above_avg_count >= 7:
            median_list.append(subset['intensity'].median())
    
    # If no median is obtained, then output 1000.
    if not median_list:
        
        return min_noise

    return np.median(median_list)    

# ms1_path = r"Gmix_P75-174.ms1"
# db_path = r"test1.db"
#
# RT_list = getMS1Rt(ms1_path)
# # creatDBTalbe(RT_list, db_path, "Q1_table", ms1_path)

#
# picture_path = r"result"
# eic_df = query_ion_eic(db_path, "Q1_table", 75.02, RT_list)
# noise = calculate_noise(eic_df, 1000)
# print("noise = ", noise)
# plot_mz_picture(eic_df, "raw_mz7502", picture_path, False)
# eic_df = secondorder_Gaussian_Filter(eic_df, 5, 2.5)
# plot_mz_picture(eic_df, "gauss_mz7502", picture_path, True)
# peaks, properties = find_peak(eic_df, noise)
# find_peak_plot(peaks, properties, eic_df, "gauss_mz7502_peak", picture_path)