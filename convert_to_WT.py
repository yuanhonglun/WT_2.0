import pandas as pd
import numpy as np

def split_dataframe(df, rows):
    if len(df) >= rows:
        if len(df) - rows <= rows:
            index_list = []
            index = 0
            while len(index_list) < len(df) - rows:
                if index > len(df):
                    break
                else:
                    index_list.append(index)
                    index = index + 2
            index_to_drop = df.index[index_list]
            remaining_data = df.drop(index_to_drop)
            remove_data = df.iloc[index_list]
        else:
            index_list = []
            index = 0
            while len(index_list) < rows:
                if index > len(df):
                    break
                else:
                    index_list.append(index)
                    index = index + 2
            index_to_drop = df.index[index_list]
            remaining_data = df.iloc[index_list]
            remove_data = df.drop(index_to_drop)
            
    else:
        remaining_data, remove_data = df, pd.DataFrame()
        
    return remaining_data, remove_data

# 用户输入参数
RT_window = 40
w = float(RT_window / 60)
f = float(0.086)

area_threshold = 500000
peak_rating_threshold = 5

# 读取原始数据文件
data_df = pd.read_csv('data.csv')

# 添加 RT_window 列到 data_df
data_df['RT_window'] = w + (data_df['RT'] * f)

# 创建空的结果 DataFrame
result_df = pd.DataFrame(columns=['CompoundName', 'RT', 'RT_window', 'Q1', 'Q3'])

# 逐行处理数据
for index, row in data_df.iterrows():
    # 从原始数据中获取相关信息
    compound_name = row['CompoundName']
    rt = row['RT']
    rt_window = row['RT_window']
    area_max = row['Area (Max.)']
    peak_rating = row['peak rating']
    fragments = row[['Fragment 1', 'Fragment 2', 'Fragment 3', 'Fragment 4', 'Fragment 5',
                     'Fragment 6', 'Fragment 7', 'Fragment 8', 'Fragment 9', 'Fragment 10']]
    mz = row['m/z']
    
    # 判断Area (Max.)是否大于等于500000
    if area_max < area_threshold or peak_rating < peak_rating_threshold:
        continue
    else:
        # 判断每个Fragment是否满足条件
        found_mrm = False
        for fragment in fragments:
            if pd.isna(fragment):
                continue
            if abs(mz - fragment) >= 14:
                result_df = pd.concat([result_df, pd.DataFrame({'CompoundName': [compound_name], 'RT': [rt], 'RT_window': [rt_window], 'Q1': [mz], 'Q3': [fragment]})], ignore_index=True)
                found_mrm = True
                break
        
        # 如果所有Fragment都不满足条件，则不添加到 result_df 中
        if not found_mrm:
            continue
        
result_df.drop_duplicates(subset=['CompoundName'], keep='first', inplace=True)
result_df.to_csv("result_df.csv")

cycle_time = float(1.2)
# 计算 c 值
c = cycle_time / 60

# 计算 RT_max
RT_max = data_df['RT'].max() + data_df['RT_window'].max()

# 计算 matrix_df 的列数量
column_names = [str(round(c * i, 5)) for i in range(int(RT_max / c) + 1)]

# 创建 matrix_df
matrix_df = pd.DataFrame(index=result_df['CompoundName'], columns=column_names)
matrix_df['RT'] = result_df.set_index('CompoundName')['RT']  # 添加 RT 列
matrix_df.sort_values(by='RT', inplace=True)
matrix_df.drop(columns=['RT'], inplace=True)

# 计算 matrix_df 中每行的数值区间并填充
for index, row in matrix_df.iterrows():
    # 获取对应行的 RT 和 RT_window 的值
    rt = result_df[result_df['CompoundName'] == index]['RT'].values[0]
    
    rt_window = result_df[result_df['CompoundName'] == index]['RT_window'].values[0]
    
    # 计算 RT_min_temp 和 RT_max_temp
    RT_min_temp = rt - rt_window
    RT_max_temp = rt + rt_window
    
    # 如果 RT_min_temp 小于0，则设为0
    if RT_min_temp < 0:
        RT_min_temp = 0
    
    # 如果 RT_max_temp 大于 RT_max，则设为 RT_max
    if RT_max_temp > RT_max:
        RT_max_temp = RT_max
    
    # 找到对应的列，并将数值区间内的值设为1
    start_col = np.floor(RT_min_temp / c) * c
    end_col = np.ceil(RT_max_temp / c) * c
    for col in matrix_df.columns:
        if float(col) >= start_col and float(col) <= end_col:
            matrix_df.loc[index, col] = 1

min_dwell_time = 3
interval_time = 3
max_ion = round((cycle_time * 1000)/(min_dwell_time + interval_time))
iteration = 1

# 循环直到 remove_matrix_df 每列求和均小于等于 max_ion
while True:
    remove_matrix_df = pd.DataFrame()
    
    # 判断 matrix_df 每列求和是否大于 max_ion
    while matrix_df.sum().max() > max_ion:
        # 找到第一个大于 max_ion 的列
        column_to_split = matrix_df.columns[matrix_df.sum().values > max_ion][0]
        
        # 拆分出去的行放入 remove_matrix_df
        temp_matrix_df = matrix_df[matrix_df[column_to_split] == 1]
        matrix_df = matrix_df[matrix_df[column_to_split] != 1]
        
        # 使用 split_dataframe 函数切割 temp_matrix_df
        remaining_temp_matrix_df, remove_temp_matrix_df = split_dataframe(temp_matrix_df, max_ion)
        
        # 合并剩余部分
        matrix_df = pd.concat([matrix_df, remaining_temp_matrix_df])
        
        # 合并拆分出去的部分
        remove_matrix_df = pd.concat([remove_matrix_df, remove_temp_matrix_df])
        
    temp_result = pd.DataFrame(columns=['Q1', 'Q3', 'RT', 'CompoundName', 'ID', 'Group', 'RT_window', 'P/S', 'TT', 'DW', 'CE'])
    
    # 填充 result_1
    for compound_name in matrix_df.index:
        if not result_df[result_df['CompoundName'] == compound_name].empty:
# 获取与 compound_name 匹配的第一行数据
            compound_row = result_df[result_df['CompoundName'] == compound_name].iloc[0]
            temp_result = pd.concat([
                temp_result,
                pd.DataFrame({
                'Q1': [compound_row['Q1']],
                'Q3': [compound_row['Q3']],
                'RT': [compound_row['RT']],
                'CompoundName': [compound_name],
                'Group': [''],
                'RT_window': [compound_row['RT_window'] * 60],
                'P/S': [1],
                'TT': [''],
                'DW': [1],
                'CE': [30]
                        })
                ], ignore_index=True)

        else:
# 如果不存在匹配的行，可以选择如何处理，比如输出错误信息或者跳过这个 compound_name 的处理
            print(f"No match found for compound name: {compound_name}")
            continue
    
    temp_result.sort_values(by='RT', inplace=True)
    temp_result['ID'] = ['M' + str(i + 1) for i in range(len(temp_result))]
    temp_result.to_csv(f"result_{iteration}.csv")
    
    # 如果 remove_matrix_df 每列求和均小于等于 max_ion，则跳出循环
    if (remove_matrix_df.sum() <= max_ion).all():
        iteration += 1
        temp_result = pd.DataFrame(columns=['Q1', 'Q3', 'RT', 'CompoundName', 'ID', 'Group', 'RT_window', 'P/S', 'TT', 'DW', 'CE'])
        
        for compound_name in remove_matrix_df.index:
            if not result_df[result_df['CompoundName'] == compound_name].empty:
    # 获取与 compound_name 匹配的第一行数据
                compound_row = result_df[result_df['CompoundName'] == compound_name].iloc[0]
                temp_result = pd.concat([
                    temp_result,
                    pd.DataFrame({
                    'Q1': [compound_row['Q1']],
                    'Q3': [compound_row['Q3']],
                    'RT': [compound_row['RT']],
                    'CompoundName': [compound_name],
                    'Group': [''],
                    'RT_window': [compound_row['RT_window'] * 60],
                    'P/S': [1],
                    'TT': [''],
                    'DW': [1],
                    'CE': [30]
                            })
                    ], ignore_index=True)

            else:
    # 如果不存在匹配的行，可以选择如何处理，比如输出错误信息或者跳过这个 compound_name 的处理
                print(f"No match found for compound name: {compound_name}")
                continue

        temp_result.sort_values(by='RT', inplace=True)
        temp_result['ID'] = ['M' + str(i + 1) for i in range(len(temp_result))]
        temp_result.to_csv(f"result_{iteration}.csv")
        
        break
    
    # 将 remove_matrix_df 作为新的 matrix_df，进行下一轮循环
    matrix_df = remove_matrix_df.copy()
    iteration += 1