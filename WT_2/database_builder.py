import sqlite3
import pandas as pd
import logging

class DatabaseBuilder:
    def __init__(self, mgf_file, product_ion_db, noise_threshold=200):
        """
        初始化数据库构建类。

        :param mgf_file: MGF 文件路径
        :param product_ion_db: 产品离子数据库路径
        :param noise_threshold: 噪音阈值
        """
        self.mgf_file = mgf_file
        self.product_ion_db = product_ion_db
        self.noise_threshold = noise_threshold


    def find_cycles(self):
        """
        根据 MGF 文件动态生成 Q3_cycles。
        """
        max_cycles = 0
        try:
            with open(self.mgf_file, 'r') as f:
                for line in f:
                    if 'cycle=' in line:
                        cycle_value = int(line.split('cycle=')[1].split()[0])
                        max_cycles = max(max_cycles, cycle_value)
        except FileNotFoundError:
            logging.error(f"MGF file {self.mgf_file} not found.")
            raise FileNotFoundError(f"MGF file {self.mgf_file} not found.")
        except Exception as e:
            logging.error(f"Error reading MGF file {self.mgf_file}: {e}")
            raise

        return max_cycles

    def create_tables(self, Q3_cycles):
        """
        创建数据库中的必要表格，包括 `intensity_data`、`mz_data` 和 `rt_data`。
        """
        try:
            conn = sqlite3.connect(self.product_ion_db)
            c = conn.cursor()

            # 创建 intensity_data 表
            c.execute('''CREATE TABLE IF NOT EXISTS intensity_data (
                            pepmass REAL,
                            mz_bin TEXT,
                            {}
                        )'''.format(", ".join([f"cycle{i} REAL" for i in range(1, Q3_cycles + 1)])))

            # 创建 mz_data 表
            c.execute('''CREATE TABLE IF NOT EXISTS mz_data (
                            pepmass REAL,
                            mz_bin TEXT,
                            {}
                        )'''.format(", ".join([f"cycle{i} REAL" for i in range(1, Q3_cycles + 1)])))

            # 创建 rt_data 表
            c.execute('''CREATE TABLE IF NOT EXISTS rt_data (
                            pepmass REAL PRIMARY KEY,
                            {}
                        )'''.format(", ".join([f"cycle{i} REAL" for i in range(1, Q3_cycles + 1)])))

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logging.error(f"Database error while creating tables: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error while creating tables: {e}")
            raise

    def insert_mz_data_to_database(self, Q3_cycles):
        try:
            conn = sqlite3.connect(self.product_ion_db)
            c = conn.cursor()

            def read_mgf_lines(file_path):
                with open(file_path, 'r') as f:
                    for line in f:
                        yield line.strip()

            mgf_lines = read_mgf_lines(self.mgf_file)
            pepmass = None
            data_to_insert = {}
            intensity_values = {}

            for line in mgf_lines:
                line = line.strip()
                if line.startswith('PEPMASS='):
                    pepmass = float(line.split('=')[1])
                elif line.startswith('TITLE='):
                    cycle = int(line.split('cycle=')[1].split()[0])
                elif line.startswith(('BEGIN IONS', 'END IONS', 'RTINSECONDS=')):
                    continue
                else:
                    mz, intensity = map(float, line.split())
                    if intensity > self.noise_threshold:
                        mz_bin = "{:.2f}-{:.2f}".format((mz // 0.02) * 0.02, (mz // 0.02 + 1) * 0.02)
                        key = (pepmass, mz_bin)
                        if key not in intensity_values or intensity > intensity_values[key][cycle - 1]:
                            if key not in intensity_values:
                                intensity_values[key] = [0] * Q3_cycles
                                data_to_insert[key] = [0] * Q3_cycles
                            intensity_values[key][cycle - 1] = intensity
                            data_to_insert[key][cycle - 1] = mz

            for (pepmass, mz_bin), intensities in intensity_values.items():
                c.execute('''INSERT OR REPLACE INTO intensity_data (pepmass, mz_bin, {} )
                             VALUES (?, ?, {})'''.format(", ".join([f"cycle{i}" for i in range(1, Q3_cycles + 1)]),
                                                          ", ".join(["?"] * Q3_cycles)),
                             [pepmass, mz_bin] + intensities)

            for (pepmass, mz_bin), mz_values in data_to_insert.items():
                c.execute('''INSERT OR REPLACE INTO mz_data (pepmass, mz_bin, {} )
                             VALUES (?, ?, {})'''.format(", ".join([f"cycle{i}" for i in range(1, Q3_cycles + 1)]),
                                                          ", ".join(["?"] * Q3_cycles)),
                             [pepmass, mz_bin] + mz_values)

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logging.error(f"Database error while parsing unknown attributes in mgf file {self.mgf_file}: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error while parsing unknown attributes in mgf file {self.mgf_file}: {e}")
            raise

    def find_rt_in_seconds(self, Q3_cycles):
        """
        提取 RT 数据并返回。
        """
        rt_data = {}
        pepmass = None
        cycle = None
        rt_seconds = None
        try:
            with open(self.mgf_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('PEPMASS='):
                        pepmass = float(line.split('=')[1])
                        if pepmass not in rt_data:
                            rt_data[pepmass] = [None] * Q3_cycles
                    elif line.startswith('TITLE='):
                        cycle = int(line.split('cycle=')[1].split()[0])
                    elif line.startswith('RTINSECONDS='):
                        rt_seconds = float(line.split('=')[1])
                    elif line.startswith(('BEGIN IONS', 'END IONS', 'RTINSECONDS=')):
                        continue
                    else:
                        if pepmass is not None and cycle is not None and rt_seconds is not None:
                            rt_data[pepmass][cycle - 1] = rt_seconds
        except Exception as e:
            logging.error(f"Error reading RT data from {self.mgf_file}: {e}")
            raise

        return rt_data

    def fill_missing_values(self, rt_data):
        """
        填补缺失的 RT 数据。
        """
        for pepmass, rt_seconds_list in rt_data.items():
            start_index = 0
            while start_index < len(rt_seconds_list) and rt_seconds_list[start_index] is None:
                start_index += 1
            diff = None
            for i in range(start_index + 1, len(rt_seconds_list)):
                if rt_seconds_list[i] is not None:
                    diff = (rt_seconds_list[i] - rt_seconds_list[start_index]) / (i - start_index)
                    break

            for i in range(len(rt_seconds_list)):
                if i < start_index and rt_seconds_list[i] is None:
                    rt_seconds_list[i] = rt_seconds_list[start_index] - (diff * (start_index - i))
                elif i > start_index and rt_seconds_list[i] is None:
                    rt_seconds_list[i] = rt_seconds_list[i - 1] + diff

            rt_data[pepmass] = rt_seconds_list

        return rt_data

    def insert_rt_data(self, rt_data, Q3_cycles):
        """
        将 RT 数据插入到 rt_data 表。
        """
        try:
            conn = sqlite3.connect(self.product_ion_db)
            cursor = conn.cursor()
            for pepmass, rt_values in rt_data.items():
                cursor.execute('''INSERT OR REPLACE INTO rt_data (pepmass, {} )
                                 VALUES (?, {})'''.format(", ".join([f"cycle{i}" for i in range(1, Q3_cycles + 1)]),
                                                          ", ".join(["?"] * Q3_cycles)),
                                 [pepmass] + rt_values)

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logging.error(f"Database error while inserting RT data: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error while inserting RT data: {e}")
            raise

    def build_db(self):
        """
        构建数据库，创建表格并插入数据。
        """
        try:
            Q3_cycles = self.find_cycles()  # 根据 MGF 文件内容生成 Q3_cycles
            self.create_tables(Q3_cycles)
            self.insert_mz_data_to_database(Q3_cycles)
            # 处理 RT 数据并插入
            rt_data = self.find_rt_in_seconds(Q3_cycles)
            rt_data = self.fill_missing_values(rt_data)
            self.insert_rt_data(rt_data, Q3_cycles)
            print(f"{self.mgf_file} database build finished.")
        except Exception as e:
            logging.error(f"Error while building database: {e}")
            raise



    #---------------------------------------------db search--------------------------------#

    def get_pepmass_list(self, db):
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute("SELECT pepmass FROM rt_data")
        results = cursor.fetchall()

        conn.close()

        pepmass_list = [result[0] for result in results]
        # print(pepmass_list)

        return pepmass_list

    def getMaxIntensity1(self, df):
        # 0123列分别为intensity1, intensity2, mz1, mz2
        # 获取0、1列的最大值
        df['intensity'] = df[[0, 1]].max(axis=1)
        # 根据0、1列的最大值确定要合并的列
        df['mz'] = df.apply(lambda row: row[2] if row['intensity'] == row[0] else row[3], axis=1)

        df.drop(columns=[0, 1, 2, 3], inplace=True)
        df = df.reset_index(drop=True)
        return df
    def merge_data(self, product_ion_db, pepmass):
        conn_product_ion = sqlite3.connect(product_ion_db)


        query = '''SELECT * FROM intensity_data WHERE pepmass={} ORDER BY mz_bin'''.format(pepmass)
        intensity_data = pd.read_sql_query(query, conn_product_ion)

        query = '''SELECT * FROM mz_data WHERE pepmass={} ORDER BY mz_bin'''.format(pepmass)
        mz_data = pd.read_sql_query(query, conn_product_ion)

        query = '''SELECT * FROM rt_data WHERE pepmass={}'''.format(pepmass)
        rt_data = pd.read_sql_query(query, conn_product_ion)

        conn_product_ion.close()


        cycle_names = rt_data.columns[1:]

        rt_df = pd.DataFrame(columns=cycle_names)
        rt_df.loc[pepmass, :] = rt_data.loc[rt_data["pepmass"] == pepmass, rt_data.columns[1:]].values

        intensity_dfs = []
        mz_dfs = []

        is_merge = False
        for i in range(intensity_data.shape[0]):
            son_intensity_data = intensity_data.loc[i:i + 1, :]
            son_mz_data = mz_data.loc[i:i + 1, :]
            combined_df = pd.concat([son_intensity_data, son_mz_data], axis=0, ignore_index=True).T
            if son_intensity_data.shape[0] == 2:
                start = son_intensity_data.iloc[0, 1].split("-")[0]
                end = son_intensity_data.iloc[0, 1].split("-")[1]
                next_start = son_intensity_data.iloc[1, 1].split("-")[0]
                next_end = son_intensity_data.iloc[1, 1].split("-")[1]
                if end == next_start:
                    max_df = self.getMaxIntensity1(combined_df)
                    max_df.loc[1, :] = [f'{start}-{next_end}'] * 2
                    intensity = list(max_df["intensity"])
                    mz = list(max_df["mz"])
                    intensity_dfs.append(intensity)
                    mz_dfs.append(mz)
                    is_merge = True
                else:
                    if is_merge != True:
                        intensity_dfs.append(list(son_intensity_data.iloc[0, :]))
                        mz_dfs.append(list(son_mz_data.iloc[0, :]))
                    is_merge = False
            else:
                intensity_dfs.append(list(son_intensity_data.iloc[0, :]))
                mz_dfs.append(list(son_mz_data.iloc[0, :]))
        intensity_df, mz_df = pd.DataFrame(intensity_dfs), pd.DataFrame(mz_dfs)
        intensity_df = intensity_df.drop(intensity_df.columns[0], axis=1)
        intensity_df = intensity_df.set_index(intensity_df.columns[0])
        intensity_df.columns = cycle_names
        mz_df = mz_df.drop(mz_df.columns[0], axis=1)
        mz_df = mz_df.set_index(mz_df.columns[0])
        mz_df.columns = cycle_names

        return intensity_df, mz_df, rt_df



#
# a = DatabaseBuilder(mgf_file='../test_data/CRL_SIF_P_1/CRL_SIF_P_075-174_1.mgf', product_ion_db=r'D:\work\WT2.0\WT_2\test_data\db\CRL_SIF_P_075-174_1.db', noise_threshold=200)
# a.build_db()