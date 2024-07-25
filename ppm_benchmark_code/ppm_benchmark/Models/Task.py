import os
import pandas as pd
import pickle
from ppm_benchmark.utils.progress_bar import ft
import numpy as np
import re
from collections import defaultdict
from ppm_benchmark.utils.hashable_dict import HashableDict


class Task:

    def __init__(self, name, task_generator, save_folder, task_type):
        self.name = name
        self.task_generator = task_generator
        self.save_folder = save_folder
        self.task_type = task_type

    @ft.nested_function_call
    def generate_task(self, train, test):
        evaluation_data = self.task_generator.generate_task(train, test, self.name)

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        train.to_csv(os.path.join(self.save_folder, 'train.csv'), index=None)
        test.to_csv(os.path.join(self.save_folder, 'test.csv'), index=None)

        return evaluation_data

    def get_train_data(self):
        train_df = pd.read_csv(os.path.join(self.save_folder, 'train.csv'))
        return train_df

    def get_test_data(self):
        test_df = pd.read_csv(os.path.join(self.save_folder, 'test.csv'))
        return test_df
