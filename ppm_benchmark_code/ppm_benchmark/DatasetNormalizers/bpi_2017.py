from ppm_benchmark.Models.BaseDatasetNormalizer import BaseDatasetNormalizer


class BPI2017Normalizer(BaseDatasetNormalizer):

    def __init__(self):
        super().__init__()

    def normalize_next_attribute(self, df):
        return df

    def normalize_outcome(self, df):
        return self.normalize_next_attribute(df)

    def normalize_attribute_suffix(self, df):
        return self.normalize_next_attribute(df)
