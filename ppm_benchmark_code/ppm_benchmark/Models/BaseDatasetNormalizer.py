from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseDatasetNormalizer(ABC):

    def __init__(self):
        pass

    def start_from_date(self, dataset, start_date):
        '''
        removes outliers starting before start date from dataset
        Args:
            dataset: pandas DataFrame
            start_date: string "MM-YYYY": dataset starts here after removing outliers

        Returns:
            dataset: pandas Dataframe

        '''
        case_starts_df = pd.DataFrame(dataset.groupby("case:concept:name")["time:timestamp"].min().reset_index())
        case_starts_df['date'] = case_starts_df["time:timestamp"].dt.to_period('M')
        cases_after = case_starts_df[case_starts_df['date'].astype('str') >= start_date]["case:concept:name"].values
        dataset = dataset[dataset["case:concept:name"].isin(cases_after)]
        return dataset

    def end_before_date(self, dataset, end_date):
        '''

        removes outliers ending after end date from dataset
        Args:
            dataset: pandas DataFrame
            end_date: string "MM-YYYY": dataset stops here after removing outliers

        Returns:
            dataset: pandas Dataframe
        '''
        case_stops_df = pd.DataFrame(dataset.groupby("case:concept:name")["time:timestamp"].max().reset_index())
        case_stops_df['date'] = case_stops_df["time:timestamp"].dt.to_period('M')
        cases_before = case_stops_df[case_stops_df['date'].astype('str') <= end_date]["case:concept:name"].values
        dataset = dataset[dataset["case:concept:name"].isin(cases_before)]
        return dataset

    def limited_duration(self, dataset, max_duration):
        '''

        limits dataset to cases shorter than maximal duration and debiases the end of the dataset
        by dropping cases starting after the last timestamp of the dataset - max_duration
        Args:
            dataset: pandas DataFrame
            max_duration: float

        Returns:
            dataset: pandas Dataframe
            latest_start: timeStamp with new end time for the dataset

        '''
        # compute each case's duration
        agg_dict = {"time:timestamp": ['min', 'max']}
        duration_df = pd.DataFrame(dataset.groupby("case:concept:name").agg(agg_dict)).reset_index()
        duration_df["duration"] = (duration_df[("time:timestamp", "max")] - duration_df[
            ("time:timestamp", "min")]).dt.total_seconds() / (24 * 60 * 60)
        # condition 1: cases are shorter than max_duration
        condition_1 = duration_df["duration"] <= max_duration * 1.00000000001
        cases_retained = duration_df[condition_1]["case:concept:name"].values
        dataset = dataset[dataset["case:concept:name"].isin(cases_retained)].reset_index(drop=True)
        # condition 2: drop cases starting after the dataset's last timestamp - the max_duration
        latest_start = dataset["time:timestamp"].max() - pd.Timedelta(max_duration, unit='D')
        condition_2 = duration_df[("time:timestamp", "min")] <= latest_start
        cases_retained = duration_df[condition_2]["case:concept:name"].values
        dataset = dataset[dataset["case:concept:name"].isin(cases_retained)].reset_index(drop=True)
        return dataset, latest_start

    def train_test_split(self, df, test_len, latest_start, targets):
        '''
        splits the dataset in train and test set, applying strict temporal splitting and
        debiasing the test set
        Args:
            df: pandas DataFrame
            test_len: float: share of cases belonging in test set
            latest_start: timeStamp with new end time for the dataset
        Returns:
            df_train: pandas DataFrame
            df_test: pandas DataFrame
        '''
        case_starts_df = df.groupby("case:concept:name")["time:timestamp"].min()
        case_nr_list_start = case_starts_df.sort_values().index.array
        case_stops_df = df.groupby("case:concept:name")["time:timestamp"].max().to_frame()

        first_test_case_nr = int(len(case_nr_list_start) * (1 - test_len))
        first_test_start_time = np.sort(case_starts_df.values)[first_test_case_nr]
        test_case_nrs = case_stops_df[case_stops_df["time:timestamp"].values >= first_test_start_time].index.array
        df_test_all = df[df["case:concept:name"].isin(test_case_nrs)].reset_index(drop=True)

        df_test = df_test_all[df_test_all["time:timestamp"] <= latest_start]

        df_test.loc[df_test["time:timestamp"].values < first_test_start_time, targets] = np.nan

        train_case_nrs = case_stops_df[
            case_stops_df["time:timestamp"].values < first_test_start_time].index.array
        df_train = df[df["case:concept:name"].isin(train_case_nrs)].reset_index(drop=True)

        return df_train, df_test

    def split_next_attribute(self, df, start_date, end_date, max_days, attr_col):
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
        if start_date:
            df = self.start_from_date(df, start_date)
        if end_date:
            df = self.end_before_date(df, end_date)
        df.drop_duplicates(inplace=True)

        df, latest_start = self.limited_duration(df, max_days)

        case_stops_df = df.groupby("case:concept:name")["time:timestamp"].max()

        sorted_timestamps = np.sort(df["time:timestamp"].values)
        separation_time = sorted_timestamps[int(len(sorted_timestamps) * (1 - 0.2))]

        dataset_train = df[df["time:timestamp"] <= pd.to_datetime(separation_time, utc=True)].reset_index(
            drop=True)
        test_case_nrs = case_stops_df[case_stops_df.values > separation_time].index.array
        dataset_test = df[df["case:concept:name"].isin(test_case_nrs)].reset_index(drop=True)

        def calcNextEvent(grp):
            grp["target"] = grp[attr_col].shift(periods=-1)
            return grp

        dataset_train = dataset_train.groupby("case:concept:name").apply(calcNextEvent)
        dataset_train = dataset_train.dropna(subset=["target"]).reset_index(drop=True)
        dataset_test = dataset_test.groupby("case:concept:name").apply(calcNextEvent)
        dataset_test = dataset_test.dropna(subset=["target"]).reset_index(drop=True)
        dataset_test.loc[dataset_test["time:timestamp"].values < separation_time, "target"] = np.nan

        if attr_col == 'time:timestamp':
            dataset_train['target'] = dataset_train['target'].astype('int64').astype(int) / (24 * 60 * 60)
            dataset_test['target'] = dataset_test['target'].astype('int64').astype(int) / (24 * 60 * 60)
        return dataset_train, dataset_test

    def split_outcome(self, df, start_date, end_date, max_days, keywords_dict):
        def tv(group):
            classification_target = 'none'
            for event in group.itertuples(index=False):
                for target, keywords in keywords_dict.items():
                    if event._2 in keywords:
                        classification_target = target
            group['target'] = classification_target
            return group

        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)

        if 'REMAINING_TIME' in keywords_dict.keys():
            df["target"] = df.groupby("case:concept:name")["time:timestamp"].apply(
                lambda x: x.max() - x).values
            df["target"] = df["target"].dt.total_seconds() / (24 * 60 * 60)

        else:
            df = df.groupby("case:concept:name").apply(tv).reset_index(drop=True)

        if start_date:
            df = self.start_from_date(df, start_date)
        if end_date:
            df = self.end_before_date(df, end_date)

        df.drop_duplicates(inplace=True)
        dataset_short, latest_start = self.limited_duration(df, max_days)
        train, test = self.train_test_split(df, 0.2, latest_start, ['target'])
        return train, test

    def split_attribute_suffix(self, df, start_date, end_date, max_days, attr_col):
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
        if start_date:
            df = self.start_from_date(df, start_date)
        if end_date:
            df = self.end_before_date(df, end_date)
        df.drop_duplicates(inplace=True)

        df, latest_start = self.limited_duration(df, max_days)

        case_stops_df = df.groupby("case:concept:name")["time:timestamp"].max()

        sorted_timestamps = np.sort(df["time:timestamp"].values)
        separation_time = sorted_timestamps[int(len(sorted_timestamps) * (1 - 0.2))]

        dataset_train = df[df["time:timestamp"] <= pd.to_datetime(separation_time, utc=True)].reset_index(
            drop=True)
        test_case_nrs = case_stops_df[case_stops_df.values > separation_time].index.array
        dataset_test = df[df["case:concept:name"].isin(test_case_nrs)].reset_index(drop=True)

        def calc_suffix(group):
            activities = group[attr_col].tolist()
            targets = [tuple(activities[i + 1:]) for i in range(len(activities))]
            group['target'] = targets
            return group

        dataset_train = dataset_train.groupby("case:concept:name").apply(calc_suffix)
        dataset_train = dataset_train.dropna(subset=["target"]).reset_index(drop=True)
        dataset_test = dataset_test.groupby("case:concept:name").apply(calc_suffix)
        dataset_test = dataset_test.dropna(subset=["target"]).reset_index(drop=True)
        dataset_test.loc[dataset_test["time:timestamp"].values < separation_time, "target"] = np.nan
        return dataset_train, dataset_test

    @abstractmethod
    def normalize_next_attribute(self, df):
        pass

    @abstractmethod
    def normalize_outcome(self, df):
        pass

    @abstractmethod
    def normalize_attribute_suffix(self, df):
        pass

    def normalize_and_split(self, df, task_type, start_date, end_date, max_days, keywords_dict=None, attr_col=None):
        if task_type == 'next_attribute':
            df = self.normalize_next_attribute(df)
            train, test = self.split_next_attribute(df, start_date, end_date, max_days, attr_col)
        elif task_type == 'outcome':
            df = self.normalize_outcome(df)
            train, test = self.split_outcome(df, start_date, end_date, max_days, keywords_dict)
        elif task_type == 'attribute_suffix':
            df = self.normalize_attribute_suffix(df)
            train, test = self.split_attribute_suffix(df, start_date, end_date, max_days, attr_col)
        else:
            raise ValueError(f'Unknown task type {task_type}')

        return train, test
