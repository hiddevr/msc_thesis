import pm4py
import pandas as pd
from .dataset import Dataset
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class DatasetGenerator:

    def __init__(self, save_folder):
        self.save_folder = save_folder

    def _load_el(self, file_path):
        el = pm4py.read_xes(file_path)
        return el

    def _get_time_group_indices(self, el, granularity='week'):
        el.sort_values(by=['time:timestamp'], ascending=True, inplace=True)
        el['year'] = el['time:timestamp'].dt.isocalendar().year
        el['week'] = el['time:timestamp'].dt.isocalendar().week
        if granularity == 'day':
            el['day'] = el['time:timestamp'].dt.isocalendar().day
        start_year = el['year'].iloc[0]
        start_week = el['week'].iloc[0]
        if granularity == 'day':
            start_day = el['day'].iloc[0]

        if granularity == 'week':
            el['time_group_index'] = (el['year'] - start_year) * 52 + (el['week'] - start_week + 1)
            el = el.drop(columns=['year', 'week'])
        elif granularity == 'day':
            el['time_group_index'] = (el['year'] - start_year) * 52 + (el['week'] - start_week + 1) + (el['day'] - start_day + 1)
            el = el.drop(columns=['year', 'week', 'day'])

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
        variants = pm4py.stats.get_variants_as_tuples(el)
        variant_ids = {s: i for i, s in enumerate(variants.keys())}

        def process_group(week_index, group):
            results = {}

            variant_counts = dict()
            group_variants = pm4py.stats.get_variants_as_tuples(group)
            for variant in group_variants.keys():
                variant_id = variant_ids[variant]
                variant_counts[variant_id] = variants[variant]

            results['variant_occurences'] = variant_counts
            results['start_activities'] = pm4py.stats.get_start_activities(group)
            results['end_activities'] = pm4py.stats.get_end_activities(group)
            results['num_variants'] = len(pm4py.stats.get_variants(group))
            results['min_self_distances'] = pm4py.stats.get_minimum_self_distances(group)
            results['case_arrival_average'] = pm4py.stats.get_case_arrival_average(group)
            results['rework_per_act'] = pm4py.stats.get_rework_cases_per_activity(group)
            results['mean_case_duration'] = np.mean(pm4py.stats.get_all_case_durations(group))
            results['activity_position_summaries'] = {activity: pm4py.stats.get_activity_position_summary(group, activity) for activity in activities}

            results['cat_event_attr_values'] = {attr: pm4py.stats.get_event_attribute_values(group, attr) for attr in categorical_event_attributes}
            results['cat_case_attr_values'] = {attr: pm4py.stats.get_event_attribute_values(group, attr) for attr in categorical_case_attributes}
            results['num_event_attr_values'] = {attr: self._mean_numeric_attr(pm4py.stats.get_event_attribute_values(group, attr)) for attr in numerical_event_attributes}
            results['num_case_attr_values'] = {attr: self._mean_numeric_attr(pm4py.stats.get_event_attribute_values(group, attr)) for attr in numerical_case_attributes}
            return week_index, results

        grouped_el = el.groupby('first_time_group_index')
        activities = el['concept:name'].unique().tolist()

        print(f'Number of weeks: {max(el["time_group_index"])}')
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_week = {executor.submit(process_group, week_index, group): week_index for week_index, group in
                              grouped_el}

            for future in tqdm(as_completed(future_to_week)):
                week_index, results = future.result()
                dataset.data_store[week_index] = results

    def generate(self, name, el_path, granularity):
        el = self._load_el(el_path)
        el = self._get_time_group_indices(el, granularity)
        categorical_case_attributes, categorical_event_attributes, numerical_case_attributes, numerical_event_attributes \
            = self._classify_attributes(el)

        dataset = Dataset()
        dataset.create(name + '_' + granularity)
        dataset.init_data_store(el['time_group_index'].unique().tolist())
        print(f'Calculating statistics for {name}')
        self._calc_stats(el, dataset, categorical_case_attributes, categorical_event_attributes,
                         numerical_case_attributes, numerical_event_attributes)

        dataset.save(self.save_folder)
        return


