import yaml
import os
from ppm_benchmark.Models.Benchmark import Benchmark
from ppm_benchmark.Models.Dataset import Dataset
from ppm_benchmark.Models.Task import Task
import importlib
import pickle


class BenchmarkLoader:

    def __init__(self):
        pass

    def _load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def _get_datasets(self, config):
        normalizer_module = importlib.import_module('ppm_benchmark.DatasetNormalizers')
        loader_module = importlib.import_module('ppm_benchmark.DatasetLoaders')
        datasets = []
        for dataset in config['datasets']:
            normalizer = getattr(normalizer_module, dataset['dataset_normalizer'])
            loader = getattr(loader_module, dataset['dataset_loader'])
            data_path = dataset['data_path']
            is_remote = dataset['is_remote']
            data_owner = dataset['data_owner']
            name = dataset['name']
            dataset_obj = Dataset(name, normalizer(), loader(), data_path, is_remote, data_owner)
            datasets.append(dataset_obj)
        return datasets

    def _get_metrics(self, task):
        module = importlib.import_module('ppm_benchmark.Metrics')
        metrics = []
        for metric in task['metrics']:
            cls = getattr(module, metric['name'])
            metrics.append(cls())
        return metrics

    def _get_task_generator(self, task):
        module = importlib.import_module('ppm_benchmark.TaskGenerators')
        task_generator = task['task_generator']
        cls = getattr(module, task_generator['name'])
        return cls()

    def _get_tasks(self, config):
        tasks = config['benchmark']['tasks']
        final_tasks = []
        for task in tasks:
            metrics = self._get_metrics(task)
            task_generator = self._get_task_generator(task)
            task_obj = Task(task['name'], metrics, task_generator, task['save_folder'], task['category'])
            final_tasks.append(task_obj)
        return final_tasks

    def _init_tasks(self, tasks, datasets, config):
        for dataset in datasets:
            dataset_config = None
            for dataset_in_config in config['datasets']:
                if dataset_in_config['name'] == dataset.name:
                    dataset_config = dataset_in_config
            task_names = [task['name'] for task in dataset_config['tasks']]
            normalized_data = dataset.normalize()
            for task in tasks:
                if task.name in task_names:
                    task.generate_task(normalized_data)
        return

    def load_from_config(self, config_path):
        config = self._load_config(config_path)
        datasets = self._get_datasets(config)
        tasks = self._get_tasks(config)
        self._init_tasks(tasks, datasets, config)
        benchmark = Benchmark(config['benchmark']['name'], tasks)

        save_folder = config['benchmark']['save_folder']
        file_path = os.path.join(save_folder, "benchmark.pkl")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        with open(file_path, 'wb') as file:
            pickle.dump(benchmark, file)

        return benchmark

    def load_from_folder(self, folder):
        file_path = os.path.join(folder, "benchmark.pkl")
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
            return obj


