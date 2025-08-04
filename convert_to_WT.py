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

# User Input Parameters
RT_window = 40
w = float(RT_window / 60)
f = float(0.086)

area_threshold = 500000
peak_rating_threshold = 5

# Read the original data file
data_df = pd.read_csv('data.csv')

# Add the "RT_window" column to the "data_df"
data_df['RT_window'] = w + (data_df['RT'] * f)

# Create an empty result DataFrame
result_df = pd.DataFrame(columns=['CompoundName', 'RT', 'RT_window', 'Q1', 'Q3'])

# Process data line by line
for index, row in data_df.iterrows():
    # Obtain relevant information from the original data
    compound_name = row['CompoundName']
    rt = row['RT']
    rt_window = row['RT_window']
    area_max = row['Area (Max.)']
    peak_rating = row['peak rating']
    fragments = row[['Fragment 1', 'Fragment 2', 'Fragment 3', 'Fragment 4', 'Fragment 5',
                     'Fragment 6', 'Fragment 7', 'Fragment 8', 'Fragment 9', 'Fragment 10']]
    mz = row['m/z']
    
    # Determine whether the maximum value of Area is greater than or equal to 500,00
    if area_max < area_threshold or peak_rating < peak_rating_threshold:
        continue
    else:
        # Determine whether each Fragment meets the conditions
        found_mrm = False
        for fragment in fragments:
            if pd.isna(fragment):
                continue
            if abs(mz - fragment) >= 14:
                result_df = pd.concat([result_df, pd.DataFrame({'CompoundName': [compound_name], 'RT': [rt], 'RT_window': [rt_window], 'Q1': [mz], 'Q3': [fragment]})], ignore_index=True)
                found_mrm = True
                break
        
        # If none of the Fragments meet the conditions, then they will not be added to the result_df.
        if not found_mrm:
            continue
        
result_df.drop_duplicates(subset=['CompoundName'], keep='first', inplace=True)
result_df.to_csv("result_df.csv")

cycle_time = float(1.2)
# Calculate the value of c
c = cycle_time / 60

# Calculate RT_max
RT_max = data_df['RT'].max() + data_df['RT_window'].max()

# Calculate the number of columns in matrix_df
column_names = [str(round(c * i, 5)) for i in range(int(RT_max / c) + 1)]

# Create matrix_df
matrix_df = pd.DataFrame(index=result_df['CompoundName'], columns=column_names)
matrix_df['RT'] = result_df.set_index('CompoundName')['RT']  # 添加 RT 列
matrix_df.sort_values(by='RT', inplace=True)
matrix_df.drop(columns=['RT'], inplace=True)

# Calculate the numerical range for each row in matrix_df and fill it in
for index, row in matrix_df.iterrows():
    # Obtain the values of RT and RT_window for the corresponding row
    rt = result_df[result_df['CompoundName'] == index]['RT'].values[0]
    
    rt_window = result_df[result_df['CompoundName'] == index]['RT_window'].values[0]
    
    # Calculate RT_min_temp and RT_max_temp
    RT_min_temp = rt - rt_window
    RT_max_temp = rt + rt_window
    
    # If RT_min_temp is less than 0, then set it to 0.
    if RT_min_temp < 0:
        RT_min_temp = 0
    
    # If RT_max_temp is greater than RT_max, then set it to RT_max.
    if RT_max_temp > RT_max:
        RT_max_temp = RT_max
    
    # Find the corresponding column and set the values within the numerical range to 1.
    start_col = np.floor(RT_min_temp / c) * c
    end_col = np.ceil(RT_max_temp / c) * c
    for col in matrix_df.columns:
        if float(col) >= start_col and float(col) <= end_col:
            matrix_df.loc[index, col] = 1

min_dwell_time = 3
interval_time = 3
max_ion = round((cycle_time * 1000)/(min_dwell_time + interval_time))
iteration = 1

# Repeat until the sum of each column in remove_matrix_df is less than or equal to max_ion
while True:
    remove_matrix_df = pd.DataFrame()
    
    # Determine whether the sum of each column in matrix_df is greater than max_ion
    while matrix_df.sum().max() > max_ion:
        # Find the first column that is greater than max_ion
        column_to_split = matrix_df.columns[matrix_df.sum().values > max_ion][0]
        
        # The split-out rows are placed into the remove_matrix_df variable
        temp_matrix_df = matrix_df[matrix_df[column_to_split] == 1]
        matrix_df = matrix_df[matrix_df[column_to_split] != 1]
        
        # Use the split_dataframe function to split temp_matrix_df
        remaining_temp_matrix_df, remove_temp_matrix_df = split_dataframe(temp_matrix_df, max_ion)
        
        # Merge the remaining part
        matrix_df = pd.concat([matrix_df, remaining_temp_matrix_df])
        
        # Merge and split out the separated parts
        remove_matrix_df = pd.concat([remove_matrix_df, remove_temp_matrix_df])
        
    temp_result = pd.DataFrame(columns=['Q1', 'Q3', 'RT', 'CompoundName', 'ID', 'Group', 'RT_window', 'P/S', 'TT', 'DW', 'CE'])
    
    # Fill in result_1
    for compound_name in matrix_df.index:
        if not result_df[result_df['CompoundName'] == compound_name].empty:
# Obtain the first row of data that matches the compound_name
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
# If there is no matching row, you can choose how to handle it, such as outputting an error message or skipping the processing of this compound_name.
            print(f"No match found for compound name: {compound_name}")
            continue
    
    temp_result.sort_values(by='RT', inplace=True)
    temp_result['ID'] = ['M' + str(i + 1) for i in range(len(temp_result))]
    temp_result.to_csv(f"result_{iteration}.csv")
    
    # If the sum of each column in remove_matrix_df is less than or equal to max_ion, then exit the loop
    if (remove_matrix_df.sum() <= max_ion).all():
        iteration += 1
        temp_result = pd.DataFrame(columns=['Q1', 'Q3', 'RT', 'CompoundName', 'ID', 'Group', 'RT_window', 'P/S', 'TT', 'DW', 'CE'])
        
        for compound_name in remove_matrix_df.index:
            if not result_df[result_df['CompoundName'] == compound_name].empty:
    # Obtain the first row of data that matches the compound_name
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
    # If there is no matching row, you can choose how to handle it, such as outputting an error message or skipping the processing of this compound_name.
                print(f"No match found for compound name: {compound_name}")
                continue

        temp_result.sort_values(by='RT', inplace=True)
        temp_result['ID'] = ['M' + str(i + 1) for i in range(len(temp_result))]
        temp_result.to_csv(f"result_{iteration}.csv")
        
        break
    
    # Replace the current matrix_df with remove_matrix_df and proceed to the next round of the loop
    matrix_df = remove_matrix_df.copy()
    iteration += 1