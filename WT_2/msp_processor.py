import numpy as np
from matchms.importing import load_from_msp
from WT_2.utils import parse_mz_intensity, weighted_dot_product_distance
import logging

class MspGenerator:
    def __init__(self, p3_peak_group_df, msp_file_path, useHrMs1=False):
        """
        Export the MSP file. It can be done for either a single sample or multiple samples after alignment.
        :param p3_peak_group_df: De-duplicated p3 group results, including columns such as 'pepmass group apex_RT_mean', 'formatted_info(Q3:Q3_intensity)', 'Q4:Q4_intensity'
        :param msp_file_path: MSP export file

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

                msp_contents.append(f"Num Peaks: {len(mz_list)}")


                for mz, i in zip(mz_list, intensity_list):
                    mz_intensity_line = f"{mz}\t{i}"
                    msp_contents.append(mz_intensity_line)


                msp_contents.append('')
                i += 1

            with open(self.msp_file_path, 'w') as f:
                f.write('\n'.join(msp_contents))
        except Exception as e:
            logging.error(
                f"Error _write_msp of MspGenerator: {e}")
            raise




class MspFileLibraryMatcher:

    def __init__(self, query_msp_path, library_msp_path, out_path, num=3):


        """
        MSP format versus library
        :param query_msp_path: The path of the query library file
        :param library_msp_path: The path of the library file
        :param out_path: The path of the output txt file
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
            f.write('query name\tlibrary name\tpositive score\treverse score\taverage score\n')
            count_flag = 0
            for q in query_spectrums:
                count_flag += 1


                if 'precursor_mz' in q.metadata:
                    q_mz, q_i, q_Q1 = q.mz, q.intensities, q.metadata['precursor_mz']
                    idx_list = self.find_indices_within_range(l_Q1_list, q_Q1)
                    scores = []

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
                        f.write(f'{score[1]}\t{score[2]}\t{score[3]}\t{score[4]}\t{score[0]}\n')



    def filter_intensity_neg(self, query_mz, query_intensity, library_mz, library_intensity, threshold=0.02):
        # Based on the library
        new_query_intensity1 = np.zeros_like(library_intensity)
        for i, m in enumerate(library_mz):
            idx = np.argmin(np.abs(query_mz - m))
            if np.abs(query_mz[idx] - m) <= threshold:
                new_query_intensity1[i] = query_intensity[idx]
        return new_query_intensity1

    def filter_intensity_pos(self, query_mz, query_intensity, library_mz, library_intensity, threshold=0.02):

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

            precursor_mz = i.metadata.get('precursor_mz', None)
            if precursor_mz is not None:
                Q1_list.append(precursor_mz)
                new_spectrums.append(i)

            # else:
            #     new_spectrums.append(i)
        return new_spectrums, Q1_list

    def find_indices_within_range(self, lst, target):
        indices = [i for i, x in enumerate(lst) if abs(x - target) < 0.5]
        return indices


