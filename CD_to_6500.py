import pandas as pd
import numpy as np

# 读取原始数据文件
data_df = pd.read_csv('data.csv')

# 创建空的结果 DataFrame
result_df = pd.DataFrame(columns=['CompoundName', 'RT', 'MRM'])

# 逐行处理数据
for index, row in data_df.iterrows():
    # 从原始数据中获取相关信息
    compound_name = row['CompoundName']
    rt = row['RT']
    area_max = row['Area (Max.)']
    peak_rating = row['peak rating']
    fragments = row[['Fragment 1', 'Fragment 2', 'Fragment 3', 'Fragment 4', 'Fragment 5',
                     'Fragment 6', 'Fragment 7', 'Fragment 8', 'Fragment 9', 'Fragment 10']]
    mz = row['m/z']
    
    # 判断Area (Max.)是否大于等于500000
    if area_max < 500000 or peak_rating < 5:
        result_df = pd.concat([result_df, pd.DataFrame({'CompoundName': [compound_name], 'RT': [rt], 'MRM': ['NA']})], ignore_index=True)
    else:
        # 判断每个Fragment是否满足条件
        found_mrm = False
        for fragment in fragments:
            if pd.isna(fragment):
                continue
            if abs(mz - fragment) >= 14:
                result_df = pd.concat([result_df, pd.DataFrame({'CompoundName': [compound_name], 'RT': [rt], 'MRM': [f"{mz}, {fragment}"]})], ignore_index=True)
                found_mrm = True
                break
        
        # 如果所有Fragment都不满足条件，则MRM列写“NA”
        if not found_mrm:
            result_df = pd.concat([result_df, pd.DataFrame({'CompoundName': [compound_name], 'RT': [rt], 'MRM': ['NA']})], ignore_index=True)

# 打印结果 DataFrame
# print(result_df)

# 将 result_df 写入到 CSV 文件
result_df.to_csv('result.csv', index=False)