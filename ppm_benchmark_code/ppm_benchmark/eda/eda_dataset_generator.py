import pm4py
import pandas as pd
from .eda_dataset import EDADataset
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import traceback


class EDADatasetGenerator:

    def __init__(self, save_folder):
        self.save_folder = save_folder

    def _get_time_group_indices(self, el):
        el.sort_values(by=['time:timestamp'], ascending=True, inplace=True)
        first_day = el['time:timestamp'].min()
        el['time_group_index'] = (el['time:timestamp'] - first_day).dt.days

        first_time_group_index = el.groupby('case:concept:name')['time_group_index'].min().reset_index()
        first_time_group_index.columns = ['case:concept:name', 'first_time_group_index']
        el = el.merge(first_time_group_index, on='case:concept:name', how='left')
        return el

    def _classify_attributes(self, el):
        nunique_per_case = el.groupby('case:concept:name').nunique()

        case_attributes = nunique_per_case.columns[(nunique_per_case == 1).all()].tolist()
        event_attributes = nunique_per_case.columns[(nunique_per_case != 1).any()].tolist()
        event_attributes.remove('time:timestamp')

        categorical_case_attributes = [col for col in case_attributes if
                                       pd.api.types.is_string_dtype(el[col]) or isinstance(el[col],
                                                                                           pd.CategoricalDtype)]
        categorical_event_attributes = [col for col in event_attributes if
                                        pd.api.types.is_string_dtype(el[col]) or isinstance(el[col],
                                                                                            pd.CategoricalDtype)]
        numerical_case_attributes = [col for col in case_attributes if pd.api.types.is_numeric_dtype(el[col])]
        numerical_event_attributes = [col for col in event_attributes if pd.api.types.is_numeric_dtype(el[col])]
        return categorical_case_attributes, categorical_event_attributes, numerical_case_attributes, numerical_event_attributes

    def _mean_numeric_attr(self, data):
        weighted_sum = sum(key * value for key, value in data.items())
        total_count = sum(data.values())
        return weighted_sum / total_count if total_count else 0

    def _calc_stats(self, el, dataset, categorical_case_attributes, categorical_event_attributes, numerical_case_attributes, numerical_event_attributes):
        def process_group(day_index, group):
            results = {}
            group_variants = pm4py.stats.get_variants_as_tuples(group)

            results['timestamp_mapping'] = group['str_date'].iloc[0]
            results['num_events'] = len(group)
            results['variant_occurences'] = group_variants
            results['start_activities'] = pm4py.stats.get_start_activities(group)
            results['end_activities'] = pm4py.stats.get_end_activities(group)
            results['num_variants'] = len(group_variants.keys())
            results['min_self_distances'] = pm4py.stats.get_minimum_self_distances(group)
            results['case_arrival_average'] = pm4py.stats.get_case_arrival_average(group)
            results['rework_per_act'] = pm4py.stats.get_rework_cases_per_activity(group)
            results['mean_case_duration'] = np.mean(pm4py.stats.get_all_case_durations(group)) / 60 / 60
            results['activity_position_summaries'] = {activity: pm4py.stats.get_activity_position_summary(group, activity) for activity in activities}

            results['cat_event_attr_values'] = {attr: pm4py.stats.get_event_attribute_values(group, attr) for attr in categorical_event_attributes}
            results['cat_case_attr_values'] = {attr: pm4py.stats.get_event_attribute_values(group, attr) for attr in categorical_case_attributes}
            results['num_event_attr_values'] = {attr: self._mean_numeric_attr(pm4py.stats.get_event_attribute_values(group, attr)) for attr in numerical_event_attributes}
            results['num_case_attr_values'] = {attr: self._mean_numeric_attr(pm4py.stats.get_event_attribute_values(group, attr)) for attr in numerical_case_attributes}
            return day_index, results

        el['str_date'] = el['time:timestamp'].dt.strftime('%d-%m-%Y')
        grouped_el = el.groupby('first_time_group_index')
        activities = el['concept:name'].unique().tolist()

        print(f'Number of days: {el["time_group_index"].nunique()}')
        with ThreadPoolExecutor() as executor:
            future_to_day = {executor.submit(process_group, time_group_index, group) for time_group_index, group in
                              grouped_el}

            for future in tqdm(as_completed(future_to_day), total=len(grouped_el)):
                day_index, results = future.result()
                dataset.data_store[day_index] = results

        dataset.data_store['log_stats'] = dict()
        dataset.data_store['log_stats']['num_cases'] = len(el['case:concept:name'].unique())
        case_durations = pm4py.stats.get_all_case_durations(el)
        dataset.data_store['log_stats']['case_durations'] = [x / 60 / 60 for x in case_durations]

    def generate(self, name, el):
        el['time:timestamp'] = pd.to_datetime(el['time:timestamp'])
        el = self._get_time_group_indices(el)
        categorical_case_attributes, categorical_event_attributes, numerical_case_attributes, numerical_event_attributes \
            = self._classify_attributes(el)

        dataset = EDADataset()
        dataset.create(name + '_' + 'day')
        dataset.init_data_store(el['time_group_index'].unique().tolist())
        print(f'Calculating statistics for {name}')
        self._calc_stats(el, dataset, categorical_case_attributes, categorical_event_attributes,
                         numerical_case_attributes, numerical_event_attributes)

        dataset.save(self.save_folder)
        return


