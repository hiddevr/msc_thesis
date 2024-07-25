from ppm_benchmark.eda.streamlit_wrapper import StreamlitWrapper
import os

def get_datasets(folder_path):
    datasets = []
    for filename in os.listdir(folder_path):
        datasets.append(filename)
    return datasets

dataset_names = get_datasets('eda_data')
wrapper = StreamlitWrapper('eda_data', dataset_names)
wrapper.start_app()