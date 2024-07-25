import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from .dylopro_data_retriever import NonInteractivePlot
import pandas as pd
import DyLoPro as dlp
import pm4py
import os
import pickle


class PlotData:

    def __init__(self):
        self.name = None
        self.el = None
        self.categorical_case_features = None
        self.numeric_event_features = None
        self.categorical_event_features = None
        self.numeric_case_features = None
        self.num_variants = None
        self.plot_obj = None
        self.data_store = None
        self.time_freq = None

    def _get_plot_kwargs(self, plot_type, num_plots, attribute_name=None):
        if plot_type == 'variants':
            kwargs = {'frequency': self.time_freq, 'max_k': num_plots}
        elif plot_type == 'cat_case_ftr':
            kwargs = {'frequency': self.time_freq, 'case_feature': attribute_name, 'max_k': num_plots}
        elif plot_type == 'num_case_ftr':
            kwargs = {'frequency': self.time_freq, 'numeric_case_list': [attribute_name]}
        elif plot_type == 'cat_event_ftr':
            kwargs = {'frequency': self.time_freq, 'event_feature': attribute_name, 'max_k': num_plots}
        elif plot_type == 'num_event_ftr':
            kwargs = {'frequency': self.time_freq, 'numeric_event_list': [attribute_name]}
        else:
            return None
        return kwargs

    def _get_dylopro_data(self, plot_type, kwargs):
        plot_method = self.plot_method_mappings[plot_type]
        with NonInteractivePlot() as plot_ctx:
            plot_method(**kwargs)
            plot_data = plot_ctx.get_plot_data()
        return plot_data

    def _update_plot_data(self, plot_type, kwargs, attribute_name=None):
        print(f'Updating {plot_type} data for {self.name} with attribute {attribute_name}')
        data_dict = dict()
        plot_data = self._get_dylopro_data(plot_type, kwargs)
        for y_axis in plot_data.keys():
            if y_axis not in ['# Cases', 'Fraction of initialized cases']:
                data_dict[y_axis.split(':')[0]] = plot_data[y_axis]

        if not attribute_name:
            self.data_store[plot_type] = data_dict
        else:
            if attribute_name not in self.data_store[plot_type].keys():
                self.data_store[plot_type] = dict()
            self.data_store[plot_type][attribute_name] = data_dict
        return

    def get_plot_data(self, plot_type, attribute_name=None):
        if plot_type not in self.data_store.keys() or (attribute_name not in self.data_store[plot_type].keys()):
            num_plots = 0
            if attribute_name:
                num_plots = len(self.el[attribute_name].unique().tolist())
            elif plot_type == 'variants':
                num_plots = self.num_variants
            elif plot_type == 'num_case_ftr':
                num_plots = len(self.numeric_case_features)
            elif plot_type == 'num_event_ftr':
                num_plots = len(self.numeric_case_features)

            kwargs = self._get_plot_kwargs(plot_type, num_plots, attribute_name)
            self._update_plot_data(plot_type, kwargs, attribute_name)

        if not attribute_name:
            return self.data_store[plot_type]
        else:
            return self.data_store[plot_type][attribute_name]

    def get_plot_types(self):
        return list(self.plot_method_mappings.keys())

    def cache_all_data(self):
        self.get_plot_data('variants')
        for attribute in self.numeric_case_features:
            self.get_plot_data('num_case_ftr', attribute)
        for attribute in self.numeric_event_features:
            self.get_plot_data('num_event_ftr', attribute)
        for attribute in self.categorical_case_features:
            self.get_plot_data('cat_case_ftr', attribute)
        for attribute in self.categorical_event_features:
            self.get_plot_data('cat_event_ftr', attribute)

    def save(self, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        filename = f"{self.name}_{self.time_freq}"
        file_path = os.path.join(save_folder, filename + '.pkl')
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
        print(f'Saved plot data to: {file_path}')

    def load_from_file(self, save_path):
        with open(save_path, 'rb') as file:
            saved_plot_data = pickle.load(file)
        self.name = saved_plot_data.name
        self.el = saved_plot_data.el
        self.categorical_case_features = saved_plot_data.categorical_case_features
        self.numeric_event_features = saved_plot_data.numeric_event_features
        self.categorical_event_features = saved_plot_data.categorical_event_features
        self.numeric_case_features = saved_plot_data.numeric_case_features
        self.num_variants = saved_plot_data.num_variants
        self.plot_obj = saved_plot_data.plot_obj
        self.data_store = saved_plot_data.data_store
        self.time_freq = saved_plot_data.time_freq

    def create(self, name, file_path, time_freq):
        self.name = name
        self.el = self._load_event_log(file_path)
        (self.categorical_case_features, self.numeric_event_features, self.categorical_event_features,
         self.numeric_case_features) = self._classify_columns(self.el)
        self.num_variants = len(pm4py.get_variants(self.el))
        self.plot_obj = self._get_plot_obj(self.el)
        self.plot_method_mappings = {'variants': self.plot_obj.topK_variants_evol,
                                     'cat_case_ftr': self.plot_obj.topK_categorical_caseftr_evol,
                                     'num_case_ftr': self.plot_obj.num_casefts_evol,
                                     'cat_event_ftr': self.plot_obj.topK_categorical_eventftr_evol,
                                     'num_event_ftr': self.plot_obj.num_eventfts_evol
                                     }

        self.data_store = dict()
        self.time_freq = time_freq

    def _load_event_log(self, file_path):
        el = pm4py.read_xes(file_path)
        return el

    def _classify_columns(self, df):
        categorical_case_features = []
        numeric_event_features = []
        categorical_event_features = []
        numeric_case_features = []

        case_columns = [col for col in df.columns if col.startswith('case:')]

        for col in case_columns:
            if col == 'case:concept:name' or df[col].nunique() < 2:
                continue
            if df[col].dtype == 'object' or df[col].dtype == 'string':
                categorical_case_features.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                numeric_case_features.append(col)

        for col in df.columns:
            if col not in case_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    unique_values = df.groupby('case:concept:name')[col].nunique()
                    if unique_values.max() > 1:
                        numeric_event_features.append(col)
                elif df[col].dtype == 'object' or df[col].dtype == 'string':
                    unique_values = df.groupby('case:concept:name')[col].nunique()
                    if unique_values.max() > 1:
                        categorical_event_features.append(col)
        print(categorical_case_features, numeric_event_features, categorical_event_features, numeric_case_features)
        return categorical_case_features, numeric_event_features, categorical_event_features, numeric_case_features

    def _get_plot_obj(self, el):
        plot_object = dlp.DynamicLogPlots(event_log=el,
                                          case_id_key='case:concept:name',
                                          activity_key='concept:name',
                                          timestamp_key='time:timestamp',
                                          categorical_casefeatures=self.categorical_case_features,
                                          numerical_casefeatures=self.numeric_case_features,
                                          categorical_eventfeatures=self.categorical_event_features,
                                          numerical_eventfeatures=self.numeric_event_features)
        return plot_object


