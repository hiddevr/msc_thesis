import pandas as pd

from ppm_benchmark.Models.BaseTaskGenerator import BaseTaskGenerator


class SimpleClassificationGenerator(BaseTaskGenerator):

    def __init__(self):
        BaseTaskGenerator.__init__(self, 'simple_classification')

    def generate_task(self, df):
        unique_cases = df['case:concept:name'].unique()
        num_cases = len(unique_cases)
        split_index = int(num_cases * 0.8)
        train_cases = unique_cases[:split_index]
        test_cases = unique_cases[split_index:]

        train_df = df[df['case:concept:name'].isin(train_cases)]
        test_df = df[df['case:concept:name'].isin(test_cases)]

        grouped = test_df.groupby('case:concept:name')
        last_values = grouped['concept:name'].apply(lambda x: x.iloc[-1])
        test_targets = last_values.to_dict()
        return train_df, test_df, test_targets
