from ppm_benchmark.Models.BaseDatasetLoader import BaseDatasetLoader
import tempfile
import os
from ppm_benchmark.utils.output_suppression import suppress_output
from ppm_benchmark.utils.progress_bar import ft
import requests
import pm4py


class LocalXes(BaseDatasetLoader):

    def __init__(self):
        super().__init__()

    def load_data(self, file_path):
        with suppress_output():
            event_log = pm4py.read_xes(file_path)
        return event_log
