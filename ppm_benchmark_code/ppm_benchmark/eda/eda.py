import yaml
import importlib
from ..Models.Dataset import Dataset
import concurrent.futures
from .eda_dataset_generator import EDADatasetGenerator
import pandas as pd
import traceback
import concurrent.futures
import traceback


class EDA:

    def __init__(self, config_path, save_folder):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.dataset_generator = EDADatasetGenerator(save_folder)

    def _process_single_dataset(self, dataset, processed_datasets):
        normalizer_module = importlib.import_module('ppm_benchmark.DatasetNormalizers')
        loader_module = importlib.import_module('ppm_benchmark.DatasetLoaders')

        if isinstance(processed_datasets, list) and dataset['name'] in processed_datasets:
            return

        print(f'Processing {dataset["name"]} dataset...')
        normalizer = getattr(normalizer_module, dataset['dataset_normalizer'])
        loader = getattr(loader_module, dataset['dataset_loader'])
        data_path = dataset['data_path']
        is_remote = dataset['is_remote']
        data_owner = dataset['data_owner']
        name = dataset['name']
        dataset_instance = Dataset(name, normalizer(), loader(), data_path, is_remote, data_owner)
        raw_data = dataset_instance.load()
        classification_data = dataset_instance.normalize('classification', include_load=False, raw_data=raw_data)
        normalized_data = dataset_instance.normalize('next_event', include_load=False, raw_data=raw_data)
        if isinstance(classification_data, pd.DataFrame):
            normalized_data = pd.concat([normalized_data, classification_data]).reset_index(drop=True)

        self.dataset_generator.generate(dataset_instance.name, normalized_data)

    def _process_datasets(self, processed_datasets, max_workers):
        datasets = self.config['datasets']
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_single_dataset, dataset, processed_datasets) for dataset in datasets]
            for future in concurrent.futures.as_completed(futures):
                try:
                    # Get the result to catch exceptions
                    result = future.result()
                    # You can do something with the result here if needed
                except Exception as e:
                    # Handle the exception
                    print(f"An exception occurred: {e}")
                    traceback.print_exc()

    def process_data(self, max_workers=1, processed_datasets=None):
        self._process_datasets(processed_datasets, max_workers)
