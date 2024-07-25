import pandas as pd
from ppm_benchmark.utils.progress_bar import ft
import sys


class Dataset:

    def __init__(self, name, dataset_normalizer, dataset_loader, data_path, is_remote, data_owner):
        self.name = name
        self.dataset_normalizer = dataset_normalizer
        self.dataset_loader = dataset_loader
        self.data_path = data_path
        self.is_remote = is_remote
        self.data_owner = data_owner

    @ft.nested_function_call
    def load(self):
        if self.is_remote:
            print(f'Downloading {self.name} dataset by {self.data_owner} from {self.data_path}')

        data = self.dataset_loader.load_data(self.data_path)
        return data

    @ft.nested_function_call
    def normalize_and_split(self, task_type, start_date, end_date, max_days, keywords_dict=None, attr_col=None, include_load=True, raw_data=None):
        if include_load:
            raw_data = self.load()
        train, test = self.dataset_normalizer.normalize_and_split(raw_data, task_type, start_date, end_date, max_days, keywords_dict, attr_col)
        return train, test

    def create_task(self):
        pass
