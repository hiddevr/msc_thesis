import uuid
from datetime import datetime
import tensorflow as tf
import pickle
import os
from collections import defaultdict
from ppm_benchmark.Models.Run import Run


class RunNotFoundError(Exception):
    pass


class TFCallback(tf.keras.callbacks.Callback):
    def __init__(self, experiment, task, run_id):
        self.experiment = experiment
        self.task = task
        self.run_id = run_id
        self.epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        if self.epochs == 0:
            hyperparameters = {
                "learning_rate": float(tf.keras.backend.get_value(self.model.optimizer.lr)),
                "batch_size": self.params['batch_size'],
                "optimizer": type(self.model.optimizer).__name__,
                "layers": []
            }
            for layer in self.model.layers:
                layer_info = {
                    "name": layer.name,
                    "type": type(layer).__name__,
                    "output_shape": layer.output_shape
                }
                hyperparameters["layers"].append(layer_info)
            self.experiment.set_start_time(self.run_id)
        elif self.epochs != 0 and logs is not None:
            self.experiment.log_train_metrics(epoch, logs, self.run_id)
        self.epochs += 1

    def on_train_end(self):
        self.experiment.set_end_time(self.run_id)

    def evaluate(self, predictions):
        for metric, score in self.task.evaluate(predictions):
            self.experiment.log_evaluation(self.run_id, metric, score)
        self.experiment.finish_run(self.run_id, self.task)


class GenericCallback:
    def __init__(self, experiment, task, run_id):
        self.experiment = experiment
        self.task = task
        self.run_id = run_id
        self.epochs = 0

    def set_hyperparams(self, hyperparams):
        self.experiment.set_hyperparams(self.run_id, hyperparams)

    def epoch_end(self, data=None, suppress_warnings=False):
        if self.epochs == 0:
            self.experiment.set_start_time(self.run_id)
        elif self.epochs != 0 and data is not None:
            self.experiment.log_train_metrics(self.epochs, data, self.run_id)
        elif not suppress_warnings:
            print('WARNING: no data supplied for tracking.')
        self.epochs += 1

    def train_end(self):
        self.experiment.set_end_time(self.run_id)

    def evaluate(self, predictions):
        for metric, score in self.task.evaluate(predictions):
            self.experiment.log_evaluation(self.run_id, metric, score)
        self.experiment.finish_run(self.run_id, self.task)


def nested_defaultdict():
    return defaultdict(nested_defaultdict)


class Experiment:

    def __init__(self):
        self.name = None
        self.runs = dict()

    def new_experiment(self, name):
        self.name = name

    def load_experiment(self, save_path):
        with open(save_path, 'rb') as file:
            saved_experiment = pickle.load(file)
        self.name = saved_experiment.name
        self.runs = saved_experiment.runs

    def init_run(self, task, run_id=None, model_type=None):
        if not run_id:
            run_id = uuid.uuid1()
        run = Run(task, run_id)
        self.runs[run_id] = run
        if model_type == 'tensorflow':
            return TFCallback(self, task, run_id), run
        else:
            return GenericCallback(self, task, run_id), run

    def log_train_metrics(self, epoch, metrics, run_id):
        self.runs[run_id].epochs[epoch] = metrics

    def set_hyperparams(self, run_id, hyperparams):
        self.runs[run_id].hyperparams = hyperparams

    def set_start_time(self, run_id):
        self.runs[run_id].start_time = datetime.now()

    def set_end_time(self, run_id):
        self.runs[run_id].end_time = datetime.now()

    def log_evaluation(self, run_id, metric, score):
        start = self.runs[run_id].start_time
        end = self.runs[run_id].end_time
        time_difference = end - start

        self.runs[run_id].training_time = time_difference.total_seconds() / 60
        self.runs[run_id].test_metrics[metric] = score

    def finish_run(self, run_id, task):
        print(f'Run {run_id} for task {task.name} finished.')
        print(f"Completed in {len(self.runs[run_id].epochs.keys())} epochs")
        print(f"Training time: {self.runs[run_id].training_time} minutes")
        print(f"Test set metrics: {self.runs[run_id].test_metrics}")

    def get_runs(self):
        return [run for run in self.runs.values()]

    def get_run_by_id(self, run_id):
        for initialized_run_id in self.runs.keys():
            if initialized_run_id == run_id:
                return self.runs[run_id]
        raise RunNotFoundError(f"Cannot find run with id {run_id}")

    def save(self, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        file_path = os.path.join(save_folder, self.name + '.pkl')
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
        print(f'Saved experiment to: {file_path}')

