import tempfile
import os
import pickle
from collections import defaultdict
from ppm_benchmark.Models.BaseTaskGenerator import BaseTaskGenerator
from log_analysis.event_log_stats.create_train_test import nextEventBenchmark
from fastDamerauLevenshtein import damerauLevenshtein
from itertools import combinations
from ppm_benchmark.utils.progress_bar import ft
from ppm_benchmark.utils.output_suppression import suppress_output
import random
import numpy as np


class AttributeSuffixRegression(BaseTaskGenerator):

    def __init__(self):
        super().__init__()

    @ft.nested_function_call
    def _find_baseline_predictions(self, train):
        target_lengths = train['target'].apply(len)
        length_mode = target_lengths.mode().iloc[0]
        median_values = []
        for i in range(length_mode):
            ith_values = train['target'].apply(lambda x: x[i] if len(x) > i else np.nan)
            median_values.append(ith_values.median())

        baseline = tuple(median_values)
        return baseline

    @ft.nested_function_call
    def generate_task(self, train, test, task_name):
        categorical_case_attributes, categorical_event_attributes, numerical_case_attributes, numerical_event_attributes = self._classify_attributes(train)

        train_attr_values = self._find_train_attr_values(train, categorical_case_attributes + categorical_event_attributes)

        test_sequence_branches = self._find_closest_train_act_sequence(train, test)

        baseline_prediction = self._find_baseline_predictions()

        def process_group(group):
            group = group.reset_index(drop=True)
            activity_sequences = [tuple(group['concept:name'][:i + 1]) for i in range(len(group))]

            eval_cases = []
            for i, sequence in enumerate(activity_sequences):

                evaluation_dict = {
                    'task_name': task_name,
                    'case_id': group['case:concept:name'].iloc[0],
                    'test_index': group.index[i],
                    'activity_sequence': sequence,
                    'train_branches': test_sequence_branches[sequence][0],
                    'closest_train_sequence': test_sequence_branches[sequence][1],
                    'train_sequence_distance': test_sequence_branches[sequence][2],
                    'prediction_target': group['target'].iloc[i],
                    'baseline': baseline_prediction,
                    'fraction_completed': len(sequence) / len(group)
                }

                for attr in categorical_event_attributes + categorical_case_attributes:
                    row_attr_value = group[attr].iloc[i]
                    if row_attr_value not in train_attr_values[attr]:
                        evaluation_dict[f'attr_drift_{attr}'] = row_attr_value

                eval_cases.append(evaluation_dict)

            return eval_cases

        grouped = test.groupby('case:concept:name')
        eval_cases = []
        for _, group in ft.track_loop(grouped, "Creating evaluation data"):
            eval_cases.extend(process_group(group))

        return eval_cases
