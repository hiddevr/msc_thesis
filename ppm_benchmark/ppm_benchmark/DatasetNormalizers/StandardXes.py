from ppm_benchmark.Models.BaseDatasetNormalizer import BaseDatasetNormalizer


class StandardXes(BaseDatasetNormalizer):

    def __init__(self):
        super().__init__()

    def normalize(self, df):
        return df
