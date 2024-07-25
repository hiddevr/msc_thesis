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


class NextAttributeClassification(BaseTaskGenerator):

    def __init__(self):
        super().__init__()

    @ft.nested_function_call
    def _find_naive_baseline_predictions(self):
        all_probas = defaultdict(list)
        for act_sequence in self.train_act_branch_probas.keys():
            activity = act_sequence[-1]
            all_probas[activity].append(self.train_act_branch_probas[act_sequence])

        final_probas = dict()
        for activity in all_probas.keys():
            sum_dict = dict()
            count_dict = dict()
            for d in all_probas[activity]:
                for key, value in d.items():
                    if key in sum_dict:
                        sum_dict[key] += value
                        count_dict[key] += 1
                    else:
                        sum_dict[key] = value
                        count_dict[key] = 1
            final_probas[activity] = {key: sum_dict[key] / count_dict[key] for key in sum_dict}

        return final_probas

    @ft.nested_function_call
    def generate_task(self, train, test, task_name):
        categorical_case_attributes, categorical_event_attributes, numerical_case_attributes, numerical_event_attributes = self._classify_attributes(train)

        train_attr_values = self._find_train_attr_values(train, categorical_case_attributes + categorical_event_attributes)

        test_sequence_branches = self._find_closest_train_act_sequence(train, test)

        naive_baseline_predictions = self._find_naive_baseline_predictions()

        def process_group(group):
            group = group.reset_index(drop=True)
            activity_sequences = [tuple(group['concept:name'][:i + 1]) for i in range(len(group))]

            eval_cases = []
            for i, sequence in enumerate(activity_sequences):
                if sequence[-1] in naive_baseline_predictions.keys():
                    naive_pred = naive_baseline_predictions[sequence[-1]]
                else:
                    naive_pred = {'<UNKNOWN>': 1}

                evaluation_dict = {
                    'task_name': task_name,
                    'case_id': group['case:concept:name'].iloc[0],
                    'test_index': group.index[i],
                    'activity_sequence': sequence,
                    'train_branches': test_sequence_branches[sequence][0],
                    'closest_train_sequence': test_sequence_branches[sequence][1],
                    'train_sequence_distance': test_sequence_branches[sequence][2],
                    'prediction_target': group['target'].iloc[i],
                    'naive_baseline': naive_pred,
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
