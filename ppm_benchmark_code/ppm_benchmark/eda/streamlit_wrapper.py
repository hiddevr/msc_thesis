import os
import pickle
from .chart_generator import ChartGenerator
import streamlit as st
import time
from datetime import datetime


class StreamlitWrapper:

    def __init__(self, datasets_folder, dataset_names):
        self.datasets_folder = datasets_folder
        self.chart_generator = ChartGenerator()
        self.dataset_names = dataset_names
        self.selected_dataset_name = None

    def _load_dataset(self, dataset_name):
        with open(os.path.join(self.datasets_folder, dataset_name), 'rb') as file:
            dataset = pickle.load(file)
        return dataset

    def _convert_day_keys_to_datetime(self, data_store):
        new_data_store = {}
        for key, value in data_store.items():
            if len(value.keys()) > 0:
                date_str = value['timestamp_mapping']
                date_obj = datetime.strptime(date_str, '%d-%m-%Y')
                new_data_store[date_obj] = value
        return new_data_store

    def _clear_cache(self):
        st.cache_data.clear()

    def create_time_tab(self, selected_dataset, filtered_days):
        fig = self.chart_generator.get_num_variants_plot(selected_dataset.data_store, filtered_days)
        st.plotly_chart(fig, use_container_width=True)

        fig = self.chart_generator.get_case_duration_plot(selected_dataset.data_store, filtered_days)
        st.plotly_chart(fig, use_container_width=True)

        fig = self.chart_generator.get_case_arrival_plot(selected_dataset.data_store, filtered_days)
        st.plotly_chart(fig, use_container_width=True)

        fig = self.chart_generator.get_num_cases_plot(selected_dataset.data_store, filtered_days)
        st.plotly_chart(fig, use_container_width=True)

        #fig = self.chart_generator.create_variants_heatmap(filtered_days, selected_dataset.data_store)
        #st.plotly_chart(fig, use_container_width=True)

    def create_activity_tab(self, selected_dataset, filtered_days):
        fig = self.chart_generator.get_rework_line_chart(selected_dataset.data_store, filtered_days)
        st.plotly_chart(fig, use_container_width=True)

        fig = self.chart_generator.get_start_activities_line_chart(selected_dataset.data_store, filtered_days)
        st.plotly_chart(fig, use_container_width=True)

        fig = self.chart_generator.get_end_activities_line_chart(selected_dataset.data_store, filtered_days)
        st.plotly_chart(fig, use_container_width=True)

        fig = self.chart_generator.get_min_self_distances_line_chart(selected_dataset.data_store, filtered_days)
        st.plotly_chart(fig, use_container_width=True)

    def create_cat_attr_tab(self, selected_dataset, filtered_days):
        num_buttons_per_row = 6
        first_day = list(selected_dataset.data_store.keys())[0]
        cat_case_attributes = list(selected_dataset.data_store[first_day]['cat_case_attr_values'].keys())
        cat_event_attributes = list(selected_dataset.data_store[first_day]['cat_event_attr_values'].keys())
        all_attrs = cat_case_attributes + cat_event_attributes
        attr_mappings = {}
        for attr in all_attrs:
            attr_mappings[attr.replace(':', '_')] = attr

        button_placeholder = st.container()
        button_columns = button_placeholder.columns(num_buttons_per_row, gap="small")
        selected_buttons = {}

        for i, button in enumerate(list(attr_mappings.keys())):
            with button_columns[i % num_buttons_per_row]:
                selected_buttons[button] = st.checkbox(button, key=button, value=False)
            if (i + 1) % num_buttons_per_row == 0 and (i + 1) < len(selected_buttons.keys()):
                current_row = button_placeholder.container()
                button_columns = current_row.columns(num_buttons_per_row, gap="small")

        selected_attrs = [name for name, selected in selected_buttons.items() if selected]
        for attr in selected_attrs:
            attr_name = attr_mappings[attr]
            if attr_name in cat_case_attributes:
                fig = self.chart_generator.get_cat_case_attr_line_chart(selected_dataset.data_store, filtered_days, attr_name)
            elif attr_name in cat_event_attributes:
                fig = self.chart_generator.get_cat_event_attr_line_chart(selected_dataset.data_store, filtered_days, attr_name)
            st.plotly_chart(fig, use_container_width=True)

    def create_num_attr_tab(self, selected_dataset, filtered_days):
        num_buttons_per_row = 6
        first_day = list(selected_dataset.data_store.keys())[0]
        num_case_attributes = list(selected_dataset.data_store[first_day]['num_case_attr_values'].keys())
        num_case_attributes = [attr for attr in num_case_attributes if attr not in ['time_group_index', 'first_time_group_index']]
        num_event_attributes = list(selected_dataset.data_store[first_day]['num_event_attr_values'].keys())
        num_event_attributes = [attr for attr in num_event_attributes if attr not in ['time_group_index', 'first_time_group_index']]
        all_attrs = num_case_attributes + num_event_attributes
        attr_mappings = {}
        for attr in all_attrs:
            attr_mappings[attr.replace(':', '_')] = attr

        button_placeholder = st.container()
        button_columns = button_placeholder.columns(num_buttons_per_row, gap="small")
        selected_buttons = {}

        for i, button in enumerate(list(attr_mappings.keys())):
            with button_columns[i % num_buttons_per_row]:
                selected_buttons[button] = st.checkbox(button, key=button, value=True)
            if (i + 1) % num_buttons_per_row == 0 and (i + 1) < len(selected_buttons.keys()):
                current_row = button_placeholder.container()
                button_columns = current_row.columns(num_buttons_per_row, gap="small")

        selected_attrs = [name for name, selected in selected_buttons.items() if selected]
        for attr in selected_attrs:
            attr_name = attr_mappings[attr]
            if attr_name in num_case_attributes:
                fig = self.chart_generator.get_num_case_attr_line_chart(selected_dataset.data_store, filtered_days, attr_name)
            elif attr_name in num_event_attributes:
                fig = self.chart_generator.get_num_event_attr_line_chart(selected_dataset.data_store, filtered_days, attr_name)
            st.plotly_chart(fig, use_container_width=True)

    def create_activity_position_tab(self, selected_dataset, filtered_days):
        first_day = list(selected_dataset.data_store.keys())[0]
        activities = list(selected_dataset.data_store[first_day]['cat_event_attr_values']['concept:name'].keys())
        for activity in activities:
            fig = self.chart_generator.get_activity_position_chart(selected_dataset.data_store, filtered_days, activity)
            st.plotly_chart(fig, use_container_width=True)

    def start_app(self):
        st.set_page_config(layout="wide")

        st.sidebar.title("Filter Options")
        selected_dataset_name = st.sidebar.selectbox("Choose a dataset", self.dataset_names)

        if self.selected_dataset_name != selected_dataset_name:
            self._clear_cache()
            selected_dataset = self._load_dataset(selected_dataset_name)
            selected_dataset.data_store = self._convert_day_keys_to_datetime(selected_dataset.data_store)
            self.selected_dataset_name = selected_dataset_name

        day_numbers = list(selected_dataset.data_store.keys())
        min_day = min(day_numbers)
        max_day = max(day_numbers)
        selected_days = st.sidebar.slider(
            "Select day Range",
            min_value=min_day,
            max_value=max_day,
            value=(min_day, max_day)
        )
        filtered_days = [day for day in day_numbers if selected_days[0] <= day <= selected_days[1]]

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Variants & Duration", "Activities", "Categorical Attributes",  "Numerical Attributes", "Activity Positions"])

        with tab1:
            self.create_time_tab(selected_dataset, filtered_days)

        with tab2:
            self.create_activity_tab(selected_dataset, filtered_days)

        with tab3:
            self.create_cat_attr_tab(selected_dataset, filtered_days)

        with tab4:
            self.create_num_attr_tab(selected_dataset, filtered_days)

        with tab5:
            self.create_activity_position_tab(selected_dataset, filtered_days)
