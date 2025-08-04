import os
import numpy as np
import pandas as pd
import logging
from WT_2.utils import filter_ms2_spectra, filter_unique_mz, parse_mz_intensity
from WT_2.samplealigment import aligment

class SampleQuantity:
    def __init__(self, quantity_folder, quantity_out_path, ref_file=None, useHrMs1=True, uesSampleAligmentmodel=True, SampleAligmentmodel_path=None):
        """
        :param quantity_folder: The P3 result after duplicate removal and consolidation of a single sample, uniformly placed in the "result" directory
        :param quantity_out_path: Output of quantitative results

        """
        try:
            self.quantity_folder = quantity_folder
            self.quantity_out_path = quantity_out_path
            self.ref_file = ref_file
            self.useHrMs1 = useHrMs1
            self.uesSampleAligmentmodel = uesSampleAligmentmodel
            self.SampleAligmentmodel_path = SampleAligmentmodel_path
        except Exception as e:
            logging.error(
                f"Error initializing SampleQuantity with quantity_folder={quantity_folder} and quantity_out_path={quantity_out_path}: {e}")
            raise


    def quantity_processor(self):
        files = os.listdir(self.quantity_folder)

        if self.ref_file is None:
            self.ref_file = files[0]
            other_files = files[1:]
        else:
            other_files = [x for x in files if x != self.ref_file]


        quantity_df = self.peak_alignment(self.ref_file, other_files, threshold=0.4)
        final_df = self.quantitative_ion_select(quantity_df)
        final_df = self._spilit_best_intensity_set(final_df, [self.ref_file] + other_files)

        final_df.to_csv(os.path.join(self.quantity_out_path, "quantity_result.csv"))


    def peak_alignment(self, ref_sample, other_sample_list, threshold, RT_window=24):
        '''

        Peak alignment score = 2a / (2a + b + c). If the peak groups to be aligned within the RT window have scores lower than the threshold for all reference peak groups,
        then this group of peak to be aligned becomes a new reference peak group,
        and is added to the reference peak group. Finally, a df is formed, with sample names as rows and peak group names as columns, and each cell contains MS2 data.
        '''

        reference_sample_groups = pd.read_csv(os.path.join(self.quantity_folder, ref_sample),
                                              usecols=["pepmass", "group", "apex_RT_mean", "count", "formatted_info",
                                                       "HR_pepmass"])


        final_df = reference_sample_groups.copy()
        final_df.columns = [f'{os.path.basename(ref_sample)}_{x}' for x in reference_sample_groups.columns]
        # sample_list = [sample2_groups, sample3_groups]
        # threshold = 0.1

        for f in other_sample_list:
            sample_groups = pd.read_csv(os.path.join(self.quantity_folder, f),
                                               usecols=["pepmass", "group", "apex_RT_mean", "count", "formatted_info",
                                                        "HR_pepmass"])

            sample_df = pd.DataFrame(index=reference_sample_groups.index,
                                     columns=[f'{os.path.basename(f)}_{x}' for x in reference_sample_groups.columns])
            unmatched_groups = []
            co = 0
            for group in sample_groups.itertuples():
                co += 1
                if co <= 10:
                    continue
                pepmass, RT, peaks = group.pepmass, group.apex_RT_mean, group.formatted_info
                mz = filter_ms2_spectra(peaks)


                if self.useHrMs1:
                    try:
                        pepmass = group.HR_pepmass
                        condition = ((abs(reference_sample_groups['apex_RT_mean'] - RT) <= RT_window) & (
                            (abs(reference_sample_groups['HR_pepmass'] - pepmass <= 0.1))))
                    except Exception as e:
                        logging.error(
                            f": There is no high-resolution MS1 column-HR_pepmass {e}")
                        raise

                else:
                    condition = ((abs(reference_sample_groups['apex_RT_mean'] - RT) <= RT_window) & (
                        (reference_sample_groups['pepmass'] == pepmass)))


                son_df = reference_sample_groups[condition]

                max_score = 0
                best_son_group = None

                if self.uesSampleAligmentmodel:
                    tmp_df_to_samplealigment = self._generate_df_to_samplealigment(RT, peaks, son_df)
                    predictions_list = aligment(tmp_df_to_samplealigment, self.SampleAligmentmodel_path)
                    predictions_list = [x[0] for x in predictions_list]
                    tmp_df_to_samplealigment["score"] = predictions_list
                    max_score_row = tmp_df_to_samplealigment.loc[tmp_df_to_samplealigment["score"].idxmax()]
                    score = max_score_row["score"]
                    if score > max_score and score >= threshold:
                        max_score = score

                        best_son_group = next(son_df.loc[[max_score_row['label']]].itertuples())

                else:
                    for son_group in son_df.itertuples():
                        reference_mz = filter_ms2_spectra(son_group.formatted_info)
                        b, c, a, _ = filter_unique_mz(np.array(reference_mz), np.array(mz))

                        score = 2 * a / (2 * a + b + c)

                        # If the score is greater than max_score and also greater than the threshold, update the best_son_group
                        if score > max_score and score >= threshold:
                            max_score = score
                            best_son_group = son_group

                # If a suitable "son_group" that meets the requirements is found
                if best_son_group is not None:
                    sample_df.loc[best_son_group.Index, :] = list(group)[1:]

                else:
                    # If there is no corresponding best_son_group, include the current group in the unmatched_groups.
                    unmatched_groups.append(group)

            if unmatched_groups:
                unmatched_df = pd.DataFrame([list(group)[1:] for group in unmatched_groups], columns=sample_df.columns)
                sample_df = pd.concat([sample_df, unmatched_df], ignore_index=True)
                unmatched_df.columns = reference_sample_groups.columns
                reference_sample_groups = pd.concat([reference_sample_groups, unmatched_df], ignore_index=True)

            final_df = pd.concat([final_df, sample_df], axis=1)

        final_df = pd.concat([reference_sample_groups, final_df], axis=1)

        try:
            os.remove("./tmp.pkl")
        except Exception as e:
            pass

        return final_df

    def quantitative_ion_select(self, df):

        '''
        Take the intersection of the ions contained in each MS2 column, calculate the average response of the intersecting ions, and select the ion with the highest average response as the quantitative ion for all groups in that column.
        For the remaining groups, also use this quantitative ion. Calculate the peak height of this ion. If a certain group has no selected ion, fill in 0 first. After each column is calculated, take the minimum value of that column, and replace the 0 values with 1/10 of the minimum value.
        '''

        best_mz_list = []
        best_intensity_set_list = []
        best_ms2_list = []

        for index, row in df.iterrows():
            formatted_info = row.filter(like='_formatted_info')
            mz_values = []
            intensity_values = []

            for info in formatted_info:


                if pd.notna(info):
                    mz, intensity = parse_mz_intensity(info)
                    mz_values.append(list(mz))
                    intensity_values.append(list(intensity))
                else:
                    mz_values.append([])
                    intensity_values.append([])

            flag_ion = []
            for ion, height in zip(mz_values, intensity_values):

                if flag_ion == []:
                    flag_ion = ion
                else:
                    if ion != []:
                        _, _, _, flag_ion = filter_unique_mz(flag_ion, ion)

            if flag_ion != []:
                max_avg_intensity = 0
                best_mz = None
                best_intensity_set = None
                ms2_t = None
                best_ms2 = None

                for i in flag_ion:
                    flag_close_intensities = 0
                    corresponding_intensities = []
                    for mz, intensity, ms2 in zip(mz_values, intensity_values, list(formatted_info)):
                        close_intensities = [int_value for mz_val, int_value in zip(mz, intensity) if
                                             abs(mz_val - i) <= 0.02]

                        if close_intensities:
                            close_intensities = max(close_intensities)
                            corresponding_intensities.append(close_intensities)

                            if flag_close_intensities < close_intensities:
                                ms2_t = ms2
                                flag_close_intensities = close_intensities

                        else:
                            corresponding_intensities.append(0)  # No matching intensity found

                    average_intensity = np.mean(corresponding_intensities)

                    if average_intensity > max_avg_intensity:
                        max_avg_intensity = average_intensity
                        best_mz = i
                        best_intensity_set = corresponding_intensities
                        best_ms2 = ms2_t

                        # Append the best_mz and best_intensity_set for this row
            best_mz_list.append(best_mz)
            best_ms2_list.append(best_ms2)
            min_i = min(best_intensity_set)
            for i in range(len(best_intensity_set)):
                if best_intensity_set[i] == 0:
                    best_intensity_set[i] = min_i / 10

            best_intensity_set_list.append(';'.join(map(str, best_intensity_set)))

        df['best_ms2'] = best_ms2_list
        df['best_mz'] = best_mz_list
        df['best_intensity_set'] = best_intensity_set_list

        return df
    def _generate_df_to_samplealigment(self, RT, peaks, df):
        '''

        :param RT: RT
        :param peaks: MS2
        :param df: son_df
        :return: The df input for the peak alignment model
        '''
        result_rows = []

        for i, row in df.iterrows():

            new_row = {
                'left_data_MS1': None,
                'left_data_RT': RT,
                'left_data_MS2': peaks,
                'right_data_MS1': None,
                'right_data_RT': row['apex_RT_mean'],
                'right_data_MS2': row['formatted_info'],
                'label': i
            }
            result_rows.append(new_row)

        result_df = pd.DataFrame(result_rows)
        return result_df

    def _spilit_best_intensity_set(self, df, sample_list):
        # Split the "best_intensity_set" column into one column for each sample
        split_data = df['best_intensity_set'].str.split(';', expand=True)
        split_data.columns = [x.split('.csv')[0] for x in sample_list]
        df = pd.concat([df, split_data], axis=1)
        return df



