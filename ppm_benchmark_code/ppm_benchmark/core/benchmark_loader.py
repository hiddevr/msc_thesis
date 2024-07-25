import yaml
import os
from ppm_benchmark.Models.Benchmark import Benchmark
from ppm_benchmark.Models.Dataset import Dataset
from ppm_benchmark.Models.Task import Task
import importlib
import pickle
from ppm_benchmark.utils.progress_bar import ft, pm, lock
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback
from multiprocessing import Manager, Lock
from joblib import Parallel, delayed
from collections import defaultdict


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

    def _get_metrics(self, config):
        module = importlib.import_module('ppm_benchmark.Metrics')
        metrics = []
        for metric in config['benchmark']['metrics']:
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
            task_generator = self._get_task_generator(task)
            task_obj = Task(task['name'], task_generator, task['save_folder'], config['benchmark']['task_type'])
            final_tasks.append(task_obj)
        return final_tasks

    def _get_evaluator(self, config, metrics, evaluation_dicts):
        module = importlib.import_module('ppm_benchmark.evaluators')
        cls = getattr(module, config['benchmark']['evaluator'])
        task_type = config['benchmark']['task_type']
        return cls(evaluation_dicts, metrics, task_type)

    @ft.function_decorator
    def _init_tasks(self, tasks, datasets, config, max_workers):
        normalized_datasets = defaultdict(dict)
        manager = Manager()
        global lock
        lock = manager.Lock()
        global pm
        pm.create_main_pb(len(datasets))
        evaluation_data = []

        for dataset in datasets:
            dataset_config = None
            for dataset_in_config in config['datasets']:
                if dataset_in_config['name'] == dataset.name:
                    dataset_config = dataset_in_config

            task_type = config['benchmark']['task_type']
            start_date = dataset_config['split_details']['start_date']
            end_date = dataset_config['split_details']['end_date']
            max_days = dataset_config['split_details']['max_days']
            attr_col = config['benchmark']['attr_col']
            if config['benchmark']['keywords_dict']:
                keywords_dict = dict(config['benchmark']['keywords_dict'])
            else:
                keywords_dict = None

            train, test = self._normalize_dataset(dataset, task_type, start_date, end_date, max_days, keywords_dict, attr_col)

            normalized_datasets[dataset.name]['train'] = train
            normalized_datasets[dataset.name]['test'] = test

        for task in tasks:
            task_data = self._process_task(task, normalized_datasets, config)
            evaluation_data.extend(task_data)

        return evaluation_data

    def _normalize_dataset(self, dataset, task_type, start_date, end_date, max_days, keywords_dict=None, attr_col=None):
        return dataset.normalize_and_split(task_type, start_date, end_date, max_days, keywords_dict, attr_col)

    def _process_task(self, task, normalized_datasets, config):
        task_data = []
        for normalized_dataset in normalized_datasets.keys():
            dataset_config = None
            for dataset_in_config in config['datasets']:
                if dataset_in_config['name'] == normalized_dataset:
                    dataset_config = dataset_in_config

            dataset_task_names = [task['name'] for task in dataset_config['tasks']]
            if task.name in dataset_task_names:
                train = normalized_datasets[normalized_dataset]['train']
                test = normalized_datasets[normalized_dataset]['test']
                task_data.extend(task.generate_task(train, test))

        return task_data

    def load_from_config(self, config_path, max_workers=1):
        config = self._load_config(config_path)
        datasets = self._get_datasets(config)
        tasks = self._get_tasks(config)
        metrics = self._get_metrics(config)
        evaluation_data = self._init_tasks(tasks, datasets, config, max_workers)
        evaluator = self._get_evaluator(config, metrics, evaluation_data)
        benchmark = Benchmark(config['benchmark']['name'], tasks, evaluator)

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


