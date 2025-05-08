import numpy as np
from matchms.importing import load_from_msp
from WT_2.utils import parse_mz_intensity, weighted_dot_product_distance
import logging

class MspGenerator:
    def __init__(self, p3_peak_group_df, msp_file_path, useHrMs1=False):
        """
        导出MSP文件，单样本或多样本对齐后均可

        :param p3_peak_group_df: 去重后的p3 group结果, 包括pepmass group apex_RT_mean formatted_info(Q3:Q3_intensity Q4:Q4_intensity)列
        :param msp_file_path: MSP导出文件

        """

        try:
            self.p3_peak_group_df = p3_peak_group_df
            self.msp_file_path = msp_file_path
            self.useHrMs1 = useHrMs1

            self._write_msp()
        except Exception as e:
            logging.error(
                f"Error initializing MspGenerator with p3_peak_group_df={p3_peak_group_df} and msp_file_path={msp_file_path}: {e}")
            raise

    def _write_msp(self):

        try:
            msp_contents = []

            i = 0
            for row in self.p3_peak_group_df.itertuples():

                pepmass = row.pepmass
                group = row.group
                ms2 = row.formatted_info
                mz_list, intensity_list = parse_mz_intensity(ms2)

                name_line = f"NAME: {i}|pepmass={pepmass}|group={str(group)}"
                msp_contents.append(name_line)

                # 写入PRECURSORMZ和RETENTIONTIME行
                if self.useHrMs1:
                    msp_contents.append(f"PRECURSORMZ: {row.HR_pepmass}")
                else:
                    msp_contents.append(f"PRECURSORMZ: {pepmass}")
                msp_contents.append(f"RETENTIONTIME: {row.apex_RT_mean}")
                msp_contents.append("PRECURSORTYPE: null")
                msp_contents.append("FORMULA: null")
                msp_contents.append("ONTOLOGY: null")
                msp_contents.append("INCHIKEY: null")
                msp_contents.append("SMILES: null")
                msp_contents.append("COMMENT: null")
                # 写入Num Peaks行
                msp_contents.append(f"Num Peaks: {len(mz_list)}")

                # 写入每行的accurate_mz和apex_raw_intensity
                for mz, i in zip(mz_list, intensity_list):
                    mz_intensity_line = f"{mz}\t{i}"
                    msp_contents.append(mz_intensity_line)

                # 添加一个空行分隔不同的条目
                msp_contents.append('')
                i += 1
            # 将所有内容一次性写入MSP文件
            with open(self.msp_file_path, 'w') as f:
                f.write('\n'.join(msp_contents))
        except Exception as e:
            logging.error(
                f"Error _write_msp of MspGenerator: {e}")
            raise




class MspFileLibraryMatcher:

    def __init__(self, query_msp_path, library_msp_path, out_path, num=3):


        """
        MSP格式比库

        :param query_msp_path: 需比库文件路径
        :param library_msp_path: 库文件路径

        """

        try:
            self.query_msp_path = query_msp_path
            self.library_msp_path = library_msp_path
            self.out_path = out_path
            self.num = num

        except Exception as e:
            logging.error(
                f"Error initializing MspFileLibraryMatcher with query_msp_path={query_msp_path} and library_msp_path={library_msp_path}: {e}")
            raise

    def calculateCosineBoth(self):

        query_spectrums = self.load_spectrums(self.query_msp_path)
        library_spectrums = self.load_spectrums(self.library_msp_path)
        library_spectrums, l_Q1_list = self.filter_spectrums(library_spectrums)

        with open(self.out_path, "w", encoding='utf-8') as f:
            f.write('query name,library name,positive score,reverse score,average score\n')
            count_flag = 0
            for q in query_spectrums:
                count_flag += 1

                #print(f'{count_flag}个结束')
                if 'precursor_mz' in q.metadata:
                    q_mz, q_i, q_Q1 = q.mz, q.intensities, q.metadata['precursor_mz']
                    idx_list = self.find_indices_within_range(l_Q1_list, q_Q1)
                    scores = []  # 存储所有的 ave_score 和对应的名称

                    if idx_list:
                        for idx in idx_list:
                            l = library_spectrums[idx]
                            l_mz, l_i = l.mz, l.intensities
                            new_q_i = self.filter_intensity_neg(q_mz, q_i, l_mz, l_i)  # 以库为基准
                            new_q_i_pos, new_l_i_pos, combine_mz = self.filter_intensity_pos(q_mz, q_i, l_mz, l_i)
                            q_name = q.metadata["compound_name"]
                            l_name = l.metadata["compound_name"]
                            if np.any(new_q_i) and np.any(l_i):
                                cosine_similarity_neg = weighted_dot_product_distance(new_q_i, l_i, l_mz)
                            else:
                                cosine_similarity_neg = 0

                            if np.any(new_q_i_pos) and np.any(new_l_i_pos):
                                cosine_similarity_pos = weighted_dot_product_distance(new_q_i_pos, new_l_i_pos,
                                                                                      combine_mz)
                            else:
                                cosine_similarity_pos = 0

                            ave_score = (cosine_similarity_neg + cosine_similarity_pos) / 2

                            scores.append((ave_score, q_name, l_name, cosine_similarity_pos, cosine_similarity_neg))

                    scores = sorted(scores, key=lambda x: x[0], reverse=True)[:self.num]
                    for score in scores:
                        f.write(f'{score[1]},{score[2]},{score[3]},{score[4]},{score[0]}\n')



    def filter_intensity_neg(self, query_mz, query_intensity, library_mz, library_intensity, threshold=0.02):
        # 以库为基准
        new_query_intensity1 = np.zeros_like(library_intensity)
        for i, m in enumerate(library_mz):
            idx = np.argmin(np.abs(query_mz - m))
            if np.abs(query_mz[idx] - m) <= threshold:
                new_query_intensity1[i] = query_intensity[idx]
        return new_query_intensity1

    def filter_intensity_pos(self, query_mz, query_intensity, library_mz, library_intensity, threshold=0.02):
        # 两者取并集
        # Step 1: Compute the union of query_mz and library_mz
        combined_mz = np.union1d(query_mz, library_mz)

        # Step 2: Initialize new intensity arrays with zeros
        new_query_intensity = np.zeros_like(combined_mz)
        new_library_intensity = np.zeros_like(combined_mz)

        # Step 3: Map the original intensities to the new combined m/z array
        for i, mz in enumerate(combined_mz):
            # Find the closest match in query_mz within the threshold
            idx_query = np.argmin(np.abs(query_mz - mz))
            if np.abs(query_mz[idx_query] - mz) <= threshold:
                new_query_intensity[i] = query_intensity[idx_query]

            # Find the closest match in library_mz within the threshold
            idx_library = np.argmin(np.abs(library_mz - mz))
            if np.abs(library_mz[idx_library] - mz) <= threshold:
                new_library_intensity[i] = library_intensity[idx_library]

        return new_query_intensity, new_library_intensity, combined_mz


    def load_spectrums(self, file_path):
        spectrums = []
        for spectrum in load_from_msp(file_path):
            spectrums.append(spectrum)
        return spectrums

    def filter_spectrums(self, spectrums):

        new_spectrums = []
        Q1_list = []
        for i in spectrums:
            # if i.metadata['adduct'] == '[M+H]+':
            #     new_spectrums.append(i)
            #     Q1_list.append(i.metadata['precursor_mz'])
            new_spectrums.append(i)
            Q1_list.append(i.metadata['precursor_mz'])
        return new_spectrums, Q1_list

    def find_indices_within_range(self, lst, target):
        indices = [i for i, x in enumerate(lst) if abs(x - target) < 0.5]
        return indices


