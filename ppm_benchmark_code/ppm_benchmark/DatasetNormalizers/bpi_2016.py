from ppm_benchmark.Models.BaseDatasetNormalizer import BaseDatasetNormalizer
import pandas as pd

bpi_2016_cols_to_drop = ['ComplaintTheme', 'ComplaintSubtheme', 'ComplaintTopic', 'QuestionTheme', 'QuestionSubtheme',
                'QuestionTopic', 'tip', 'service_detail', 'page_action_detail']

bpi_2016_activity_mapping = {
    'BPI2016_Complaints': 'make_complaint',
    'BPI2016_Questions': 'ask_question',
    'BPI2016_Werkmap_Messages': 'send_message'
}

bpi_2016_timestamp_mapping = {
    'BPI2016_Clicks_Logged_In': 'TIMESTAMP',
    'BPI2016_Complaints': 'ContactDate',
    'BPI2016_Questions': 'ContactDate',
    'BPI2016_Werkmap_Messages': 'EventDateTime'
}


class BPI2016Normalizer(BaseDatasetNormalizer):

    def __init__(self):
        super().__init__()

    def _process_df(self, name, df):
        if name in bpi_2016_activity_mapping.keys():
            df['concept:name'] = bpi_2016_activity_mapping[name]
        else:
            df['concept:name'] = df['URL_FILE']

        df['time:timestamp'] = pd.to_datetime(df[bpi_2016_timestamp_mapping[name]], format='mixed')

        df['org:resource'] = df['Office_U'].astype(str) + '_' + df['Office_W'].astype(str)

        return df

    def normalize_next_attribute(self, dfs):
        processed_dfs = []
        for name, df in dfs.items():
            new_df = self._process_df(name, df)
            processed_dfs.append(new_df)

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.rename(columns={'CustomerID': 'case:concept:name'})
        combined_df = combined_df.sort_values(['case:concept:name', 'time:timestamp'])
        combined_df['case:concept:name'] = combined_df['case:concept:name'].apply(str)
        combined_df = combined_df.drop(bpi_2016_cols_to_drop, axis=1)
        return combined_df

    def normalize_outcome(self, dfs):
        return self.normalize_next_attribute(dfs)

    def normalize_attribute_suffix(self, df):
        return self.normalize_next_attribute(df)
