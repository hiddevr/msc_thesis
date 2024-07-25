import os
import pickle


class Dataset:

    def __init__(self):
        self.name = None
        self.data_store = None

    def create(self, name):
        self.name = name
        self.data_store = dict()

    def init_data_store(self, time_group_indices):
        for time_group_index in time_group_indices:
            self.data_store[time_group_index] = dict()
        return self.data_store

    def save(self, save_folder):
        file_path = os.path.join(save_folder, f"{self.name}.pkl")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
            return obj