from abc import ABC, abstractmethod
import pandas as pd
from collections import defaultdict
from fastDamerauLevenshtein import damerauLevenshtein
from functools import lru_cache
from ppm_benchmark.utils.sequence_matcher import SequenceMatcher
from ppm_benchmark.utils.progress_bar import ft


class BaseTaskGenerator(ABC):

    def __init__(self):
        self.train_act_branch_probas = None

    @ft.nested_function_call
    def _classify_attributes(self, normalized_data):
        nunique_per_case = normalized_data.groupby('case:concept:name').nunique()
        num_cases = normalized_data['case:concept:name'].nunique()
        case_attributes = nunique_per_case.columns[(nunique_per_case == 1).all()].tolist()
        event_attributes = nunique_per_case.columns[(nunique_per_case != 1).any()].tolist()
        event_attributes.remove('time:timestamp')

        categorical_case_attributes = [col for col in case_attributes if (pd.api.types.is_string_dtype(normalized_data[col]) or isinstance(normalized_data[col], pd.CategoricalDtype)) and len(normalized_data[col].unique()) < num_cases]
        categorical_event_attributes = [col for col in event_attributes if pd.api.types.is_string_dtype(normalized_data[col]) or isinstance(normalized_data[col], pd.CategoricalDtype)]
        numerical_case_attributes = [col for col in case_attributes if pd.api.types.is_numeric_dtype(normalized_data[col])]
        numerical_event_attributes = [col for col in event_attributes if pd.api.types.is_numeric_dtype(normalized_data[col])]
        return categorical_case_attributes, categorical_event_attributes, numerical_case_attributes, numerical_event_attributes

    @ft.nested_function_call
    def _find_act_branch_probas(self, train):
        sub_acts = self._find_act_subsequences(train, True)
        probability_dict = {}

        for subtuple, next_values_dict in sub_acts.items():
            total_count = sum(next_values_dict.values())
            probability_dict[subtuple] = {k: v / total_count for k, v in next_values_dict.items()}

        self.train_act_branch_probas = probability_dict
        return probability_dict

    @ft.nested_function_call
    def _find_act_subsequences(self, event_log, include_counts=False):
        grouped_el = event_log.groupby('case:concept:name')
        sub_sequence_counts = defaultdict(lambda: defaultdict(int))

        for case_id, group_df in grouped_el:
            activity_sequence = tuple(group_df['concept:name'])
            for i in range(len(activity_sequence)):
                subtuple = tuple(activity_sequence[0:i + 1])
                if i + 1 < len(activity_sequence):
                    next_value = activity_sequence[i + 1]
                    sub_sequence_counts[subtuple][next_value] += 1
                else:
                    next_value = '<END>'
                    sub_sequence_counts[subtuple][next_value] += 1

        if not include_counts:
            return set(sub_sequence_counts.keys())
        else:
            return sub_sequence_counts

    @ft.nested_function_call
    def _find_attribute_drift(self, train, test, attributes):
        attribute_drift_cases = dict()

        for attribute in attributes:
            train_attr_values = set(train[attribute].unique())
            test_attr_values = test[attribute].unique()

            new_attribute_values = set(test_attr_values) - train_attr_values

            filtered_df = test[test[attribute].isin(new_attribute_values)]
            grouped_filtered_df = filtered_df.groupby('case:concept:name')[attribute].first().reset_index()

            for _, row in grouped_filtered_df.iterrows():
                case_id = row['case:concept:name']
                attribute_value = row[attribute]
                if case_id in attribute_drift_cases:
                    attribute_drift_cases[case_id][attribute] = attribute_value
                else:
                    attribute_drift_cases[case_id] = {attribute: attribute_value}

        return attribute_drift_cases

    @ft.nested_function_call
    def _find_closest_train_act_sequence(self, train, test):
        branching_probas = self._find_act_branch_probas(train)
        train_sequences = set(branching_probas.keys())
        test_sequences = self._find_act_subsequences(test)

        sequence_matcher = SequenceMatcher()
        sequence_matcher.fit(train_sequences, test_sequences)

        test_sequence_branches = dict()
        for subsequence in ft.track_loop(test_sequences, "Finding closest train sequences for test data..."):
            if subsequence in train_sequences:
                test_sequence_branches[subsequence] = (branching_probas[subsequence], subsequence, 0)
            else:
                closest_sequence, distance = sequence_matcher.find_match(subsequence)
                test_sequence_branches[subsequence] = (branching_probas[closest_sequence], closest_sequence, distance)

        return test_sequence_branches

    @ft.nested_function_call
    def _find_train_attr_values(self, train, attributes):

        train_attr_values = dict()

        for attribute in ft.track_loop(attributes, "Identifying attribute drift..."):
            train_attr_values[attribute] = set(train[attribute].unique())
        return train_attr_values

    @abstractmethod
    def generate_task(self, train, test, dataset_name):
        pass