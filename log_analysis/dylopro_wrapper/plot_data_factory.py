import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=PerformanceWarning)

from .dylopro_data_retriever import NonInteractivePlot
import pandas as pd
import DyLoPro as dlp
import pm4py
from .plot_data_v2 import PlotData


class PlotDataFactory:

    def __init__(self, time_freq):
        self.plot_method_mappings = {
            'variants': 'topK_variants_evol',
            'cat_case_ftr': 'topK_categorical_caseftr_evol',
            'num_case_ftr': 'num_casefts_evol',
            'cat_event_ftr': 'topK_categorical_eventftr_evol',
            'num_event_ftr': 'num_eventfts_evol'
        }
        self.time_freq = time_freq

    def create(self, name, file_path):
        el = self._load_event_log(file_path)
        (categorical_case_features, numeric_event_features, categorical_event_features,
         numeric_case_features) = self._classify_columns(el)
        num_variants = len(pm4py.get_variants(el))
        plot_obj = self._get_plot_obj(el, categorical_case_features, numeric_case_features,
                                      categorical_event_features, numeric_event_features)

        plot_data = PlotData(name, self.time_freq)

        self._populate_data_store(plot_data, plot_obj, el, num_variants, categorical_case_features, numeric_event_features, categorical_event_features, numeric_case_features)

        return plot_data

    def _populate_data_store(self, plot_data, plot_obj, el, num_variants, categorical_case_features, numeric_event_features, categorical_event_features, numeric_case_features):
        plot_data.add_data('variants', self._get_dylopro_data(plot_obj, 'variants', {'frequency': plot_data.time_freq, 'max_k': num_variants}))

        data = self._get_dylopro_data(plot_obj, 'num_case_ftr', {'frequency': self.time_freq, 'numeric_case_list': numeric_case_features})
        plot_data.add_data('num_case_ftr', data)

        data = self._get_dylopro_data(plot_obj, 'num_event_ftr', {'frequency': self.time_freq, 'numeric_event_list': numeric_event_features})
        plot_data.add_data('num_event_ftr', data)

        for attribute in categorical_case_features:
            data = self._get_dylopro_data(plot_obj, 'cat_case_ftr', {'frequency': self.time_freq, 'max_k': len(el[attribute].unique().tolist()), 'case_feature': attribute})
            plot_data.add_data('cat_case_ftr', data, attribute)

        for attribute in categorical_event_features:
            data = self._get_dylopro_data(plot_obj, 'cat_event_ftr', {'frequency': self.time_freq, 'max_k': len(el[attribute].unique().tolist()), 'event_feature': attribute})
            plot_data.add_data('cat_event_ftr', data, attribute)

    def _get_dylopro_data(self, plot_obj, plot_type, kwargs):
        plot_method = getattr(plot_obj, self.plot_method_mappings[plot_type])
        with NonInteractivePlot() as plot_ctx:
            plot_method(**kwargs)
            plot_data = plot_ctx.get_plot_data()

        if plot_type == 'variants':
            return {y_axis.rsplit(':', 1)[0]: plot_data[y_axis] for y_axis in plot_data.keys() if
                    y_axis not in ['# Cases', 'Fraction of initialized cases']}
        elif plot_type == 'num_case_ftr':
            return {y_axis.rsplit(' ', 1)[1]: plot_data[y_axis] for y_axis in plot_data.keys() if
                    y_axis not in ['# Cases', 'Fraction of initialized cases', f'{self.time_freq} mean case features (Normalized)']}
        elif plot_type == 'num_event_ftr':
            return {y_axis.rsplit(' ', 1)[1]: plot_data[y_axis] for y_axis in plot_data.keys() if
                    y_axis not in ['# Cases', 'Fraction of initialized cases', f'{self.time_freq} mean event features (Normalized)']}
        elif plot_type == 'cat_case_ftr':
            return {y_axis.rsplit(':', 1)[0]: plot_data[y_axis] for y_axis in plot_data.keys() if
                    y_axis not in ['# Cases', 'Fraction of initialized cases']}
        elif plot_type == 'cat_event_ftr':
            return {y_axis.rsplit(':', 1)[0]: plot_data[y_axis] for y_axis in plot_data.keys() if
                    y_axis not in ['# Cases', 'Fraction of initialized cases']}

    @staticmethod
    def _load_event_log(file_path):
        return pm4py.read_xes(file_path)

    @staticmethod
    def _classify_columns(df):
        categorical_case_features = []
        numeric_event_features = []
        categorical_event_features = []
        numeric_case_features = []

        case_columns = [col for col in df.columns if col.startswith('case:')]

        for col in case_columns:
            if col == 'case:concept:name' or df[col].nunique() < 2:
                continue
            if df[col].dtype in ['object', 'string']:
                categorical_case_features.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                numeric_case_features.append(col)

        for col in df.columns:
            if col not in case_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    unique_values = df.groupby('case:concept:name')[col].nunique()
                    if unique_values.max() > 1:
                        numeric_event_features.append(col)
                elif df[col].dtype in ['object', 'string']:
                    unique_values = df.groupby('case:concept:name')[col].nunique()
                    if unique_values.max() > 1:
                        categorical_event_features.append(col)
        print(categorical_case_features, numeric_event_features, categorical_event_features, numeric_case_features)
        return categorical_case_features, numeric_event_features, categorical_event_features, numeric_case_features

    @staticmethod
    def _get_plot_obj(el, categorical_case_features, numeric_case_features, categorical_event_features, numeric_event_features):
        return dlp.DynamicLogPlots(event_log=el,
                                   case_id_key='case:concept:name',
                                   activity_key='concept:name',
                                   timestamp_key='time:timestamp',
                                   categorical_casefeatures=categorical_case_features,
                                   numerical_casefeatures=numeric_case_features,
                                   categorical_eventfeatures=categorical_event_features,
                                   numerical_eventfeatures=numeric_event_features)