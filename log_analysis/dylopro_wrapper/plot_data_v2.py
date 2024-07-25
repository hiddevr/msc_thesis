import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=PerformanceWarning)
import os
import pickle
import numpy as np


class PlotData:
    def __init__(self, name, time_freq):
        self.name = name
        self.time_freq = time_freq
        self.data_store = {
            'variants': {},
            'cat_case_ftr': {},
            'num_case_ftr': {},
            'cat_event_ftr': {},
            'num_event_ftr': {}
        }

    def add_data(self, plot_type, data, attribute_name=None):
        if not attribute_name:
            self.data_store[plot_type] = data
        else:
            if attribute_name not in self.data_store[plot_type]:
                self.data_store[plot_type][attribute_name] = {}
            self.data_store[plot_type][attribute_name] = data

    def get_data(self, plot_type, attribute_name=None):
        if not attribute_name:
            return self.data_store.get(plot_type, {})
        else:
            return self.data_store.get(plot_type, {}).get(attribute_name, {})

    def save(self, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        filename = f"{self.name}_{self.time_freq}.pkl"
        file_path = os.path.join(save_folder, filename)
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
        print(f'Saved plot data to: {file_path}')

    @staticmethod
    def load_from_file(save_path):
        with open(save_path, 'rb') as file:
            saved_plot_data = pickle.load(file)
        return saved_plot_data

    def _calculate_average_y(self, sub_dict):
        """Recursively calculate the average of 'y' values in the nested dictionary."""
        if 'x' in sub_dict and 'y' in sub_dict:
            return sub_dict['y']
        averages = []
        for key, value in sub_dict.items():
            if isinstance(value, dict):
                avg_y = self._calculate_average_y(value)
                if avg_y is not None:
                    averages.append(avg_y)
        if averages:
            return sum(averages) / len(averages)
        return 0

    def sort_data(self):
        average_y_dict = {key: self._calculate_average_y(value) for key, value in self.data_store.items()}
        average_y_dict = {key: value for key, value in average_y_dict.items() if value is not None}
        sorted_keys = sorted(average_y_dict, key=average_y_dict.get, reverse=True)
        self.data_store = {key: self.data_store[key] for key in sorted_keys}