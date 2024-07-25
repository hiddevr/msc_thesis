from ppm_benchmark.Models.BaseDatasetLoader import BaseDatasetLoader
import tempfile
import os
from ppm_benchmark.utils.output_suppression import suppress_output
from ppm_benchmark.utils.progress_bar import ft
import requests
import pm4py


class RemoteXes(BaseDatasetLoader):

    def __init__(self):
        super().__init__()

    def load_data(self, url):
        with tempfile.TemporaryDirectory() as temp_dir:
            xes_file = self._download_xes(url, temp_dir)
            with suppress_output():
                event_log = pm4py.read_xes(xes_file)
            df = pm4py.convert_to_dataframe(event_log)
            return df

    @ft.nested_function_call
    def _download_xes(self, url, temp_dir):
        local_filename = os.path.join(temp_dir, 'downloaded.xes')
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        return local_filename
