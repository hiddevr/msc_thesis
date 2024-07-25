from ppm_benchmark.Models.BaseDatasetLoader import BaseDatasetLoader
import tempfile
import os
from ppm_benchmark.utils.output_suppression import suppress_output
from ppm_benchmark.utils.progress_bar import ft
import requests
import pm4py
import pandas as pd


class LocalCSV(BaseDatasetLoader):

    def __init__(self):
        super().__init__()

    def load_data(self, file_path):
        with suppress_output():
            event_log = pd.read_csv(file_path, sep=';')
        return event_log
