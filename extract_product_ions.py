import sqlite3

def find_cycles(mgf_file):
    max_cycles = 0
    with open(mgf_file, 'r') as f:
        for line in f:
            if 'cycle=' in line:
                cycle_value = int(line.split('cycle=')[1].split()[0])
                max_cycles = max(max_cycles, cycle_value)
                
    return max_cycles

def create_database(db_file, cycles):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    # 创建表格
    c.execute('''CREATE TABLE IF NOT EXISTS intensity_data (
                    pepmass REAL,
                    mz_bin TEXT,
                    {}
                )'''.format(", ".join(["cycle{} REAL".format(i) for i in range(1, cycles+1)])))

    conn.commit()
    conn.close()

def insert_data_to_database(db_file, noise=200):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    with open(mgf_file, 'r') as f:
        pepmass = None
        cycle = None
        data_to_insert = {}
        for line in f:
            line = line.strip()
            if line.startswith('PEPMASS='):
                pepmass = float(line.split('=')[1])
            elif line.startswith('TITLE='):
                cycle = int(line.split('cycle=')[1].split()[0])
            elif line.startswith(('BEGIN IONS', 'END IONS', 'RTINSECONDS=')):
                continue
            else:
                mz, intensity = map(float, line.split())
                if intensity > noise:
                    mz_bin = "{:.2f}-{:.2f}".format((mz // 0.02) * 0.02, (mz // 0.02 + 1) * 0.02)
                    key = (pepmass, mz_bin)
                    if key not in data_to_insert:
                        data_to_insert[key] = [0] * cycles
                    data_to_insert[key][cycle-1] = max(data_to_insert[key][cycle-1], intensity)

    # 插入数据
    for (pepmass, mz_bin), intensities in data_to_insert.items():
        c.execute('''INSERT OR REPLACE INTO intensity_data (pepmass, mz_bin, {})
                     VALUES (?, ?, {})'''.format(", ".join(["cycle{}".format(i) for i in range(1, cycles+1)]), 
                                                ", ".join(["?"] * cycles)), 
                  [pepmass, mz_bin] + intensities)

    conn.commit()
    conn.close()

def find_rt_in_seconds(mgf_file):
    rt_data = {}
    with open(mgf_file, 'r') as f:
        pepmass = None
        cycle = None
        rt_seconds = None
        for line in f:
            line = line.strip()
            if line.startswith('PEPMASS='):
                pepmass = float(line.split('=')[1])
                if pepmass not in rt_data:
                    rt_data[pepmass] = [None] * cycles
            elif line.startswith('TITLE='):
                cycle = int(line.split('cycle=')[1].split()[0])
            elif line.startswith('RTINSECONDS='):
                rt_seconds = float(line.split('=')[1])
            elif line.startswith(('BEGIN IONS', 'END IONS', 'RTINSECONDS=')):
                continue
            else:
                mz, intensity = map(float, line.split())
                if pepmass is not None and cycle is not None and rt_seconds is not None:
                    rt_data[pepmass][cycle-1] = rt_seconds

    return rt_data

def create_rt_table(db_file, cycles):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    # 创建表格
    c.execute('''CREATE TABLE IF NOT EXISTS rt_data (
                    pepmass REAL PRIMARY KEY,
                    {}
                )'''.format(", ".join(["cycle{} REAL".format(i) for i in range(1, cycles+1)])))

    conn.commit()
    conn.close()

def insert_rt_data(db_file, rt_data):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    # 插入数据
    for pepmass, rt_seconds_list in rt_data.items():
        c.execute('''INSERT OR REPLACE INTO rt_data (pepmass, {})
                     VALUES (?, {})'''.format(", ".join(["cycle{}".format(i) for i in range(1, cycles+1)]), 
                                              ", ".join(["?" for _ in range(cycles)])), 
                  [pepmass] + rt_seconds_list)

    conn.commit()
    conn.close()

def fill_null_values(rt_data):
    for pepmass, rt_seconds_list in rt_data.items():
        for i in range(len(rt_seconds_list)):
            if rt_seconds_list[i] is None:
                # Find nearest two non-null values
                left_index = i - 1
                right_index = i + 1
                while left_index >= 0 and rt_seconds_list[left_index] is None:
                    left_index -= 1
                while right_index < len(rt_seconds_list) and rt_seconds_list[right_index] is None:
                    right_index += 1
                
                # Calculate coefficient for linear interpolation
                left_value = rt_seconds_list[left_index]
                right_value = rt_seconds_list[right_index]
                num_of_nulls = right_index - left_index - 1
                coefficient = (right_value - left_value) / (num_of_nulls + 1)
                
                # Fill null value with linear interpolation
                rt_seconds_list[i] = min(left_value, right_value) + coefficient * (i - left_index)

mgf_file = 'Gmix_P75-174.mgf'

cycles = find_cycles(mgf_file)
rt_data = find_rt_in_seconds(mgf_file)
fill_null_values(rt_data)

create_rt_table('rt_data.db', cycles)
insert_rt_data('rt_data.db', rt_data)

create_database('product_ion.db', cycles)
insert_data_to_database('product_ion.db')