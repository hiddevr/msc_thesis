from ppm_benchmark.Models.BaseDatasetNormalizer import BaseDatasetNormalizer


class BPI2012Normalizer(BaseDatasetNormalizer):

    def __init__(self):
        super().__init__()

    def normalize_next_attribute(self, df):
        df['concept:name'] = df["concept:name"] + "_" + df["lifecycle:transition"]
        return df

    def normalize_outcome(self, df):
        return self.normalize_next_attribute(df)

    def normalize_attribute_suffix(self, df):
        return self.normalize_next_attribute(df)
