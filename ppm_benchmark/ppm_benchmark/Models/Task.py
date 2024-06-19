import os
import pandas as pd
import pickle


class Task:

    def __init__(self, name, metrics, task_generator, save_folder, category):
        self.name = name
        self.metrics = metrics
        self.task_generator = task_generator
        self.save_folder = save_folder
        self.category = category

    def generate_task(self, normalized_data):
        train_df, test_df, test_targets = self.task_generator.generate_task(normalized_data)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        train_df.to_csv(os.path.join(self.save_folder, 'train.csv'), index=None)
        test_df.to_csv(os.path.join(self.save_folder, 'test.csv'), index=None)
        with open(os.path.join(self.save_folder, 'targets.dict'), 'wb') as file:
            pickle.dump(test_targets, file)

    def evaluate(self, predictions):
        with open(os.path.join(self.save_folder, 'targets.dict'), 'rb') as file:
            test_targets = pickle.load(file)

        for metric in self.metrics:
            score = metric.evaluate(predictions, test_targets)
            yield metric.name, score

    def get_train_data(self):
        train_df = pd.read_csv(os.path.join(self.save_folder, 'train.csv'))
        return train_df

    def get_test_data(self):
        test_df = pd.read_csv(os.path.join(self.save_folder, 'test.csv'))
        return test_df
