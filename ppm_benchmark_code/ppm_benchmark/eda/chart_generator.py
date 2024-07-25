import plotly.express as px
import streamlit as st
import plotly.graph_objs as go
import numpy as np


class ChartGenerator:

    def __init__(self):
        pass

    def _create_num_cases_plot(self, data_store, day_numbers, y_title='N.o. cases'):
        y_lst = []
        x_lst = []
        for day in day_numbers:
            if data_store[day]:
                y_lst.append(sum(data_store[day]['variant_occurences'].values()))
                x_lst.append(day)
        data = {
            'day number': x_lst,
            y_title: y_lst
        }
        fig = px.line(data, x='day number', y=y_title, title='N.o. cases by start day')
        return fig

    def _create_single_line_plot(self, day_numbers, data_store, storage_key, y_title, plot_title):
        y_lst = []
        x_lst = []
        for day in day_numbers:
            if data_store[day]:
                y_lst.append(data_store[day][storage_key])
                x_lst.append(day)
        data = {
            'day number': x_lst,
            y_title: y_lst
        }
        fig = px.line(data, x='day number', y=y_title, title=plot_title)
        return fig

    def _create_2_level_single_line_plot(self, day_numbers, data_store, storage_key, y_title, plot_title, attribute):
        y_lst = []
        x_lst = []
        for day in day_numbers:
            if storage_key in data_store[day].keys():
                if attribute in data_store[day][storage_key].keys():
                    y_lst.append(data_store[day][storage_key][attribute])
                    x_lst.append(day)
        data = {
            'day number': x_lst,
            y_title: y_lst
        }
        fig = px.line(data, x='day number', y=y_title, title=plot_title)
        return fig

    def _create_multi_line_chart(self, day_numbers, data_store, storage_key, y_title, plot_title):
        y_lst = []
        x_lst = []
        for day in day_numbers:
            if data_store[day]:
                y_lst.append(data_store[day][storage_key])
                x_lst.append(day)

        lines = set()
        for y_dict in y_lst:
            lines.update(y_dict.keys())

        fig = go.Figure()

        # Add a trace for each line
        for line in lines:
            y_vals = []
            for y_dict in y_lst:
                y_vals.append(y_dict.get(line, None))

            fig.add_trace(go.Scatter(x=x_lst, y=y_vals, mode='lines+markers', name=line))

        # Update layout
        fig.update_layout(
            title=plot_title,
            xaxis_title='day number',
            yaxis_title=y_title,
            template='plotly_white'
        )
        return fig

    def _create_2_level_multi_line_chart(self, day_numbers, data_store, storage_key, y_title, plot_title, attribute):
        y_lst = []
        x_lst = []
        for day in day_numbers:
            if data_store[day]:
                if attribute in data_store[day][storage_key].keys():
                    y_lst.append(data_store[day][storage_key][attribute])
                    x_lst.append(day)

        lines = set()
        for y_dict in y_lst:
            lines.update(y_dict.keys())

        fig = go.Figure()

        # Add a trace for each line
        for line in lines:
            y_vals = []
            for y_dict in y_lst:
                y_vals.append(y_dict.get(line, None))

            fig.add_trace(go.Scatter(x=x_lst, y=y_vals, mode='lines+markers', name=line))

        # Update layout
        fig.update_layout(
            title=plot_title,
            xaxis_title='day number',
            yaxis_title=y_title,
            template='plotly_white'
        )
        return fig

    def _create_activity_occurence_heatmap(self, day_numbers, data_store, storage_key, y_title, plot_title, activity_key):
        dataset = dict()
        for day in day_numbers:
            if day in data_store.keys():
                if storage_key in data_store[day].keys():
                    if activity_key in data_store[day][storage_key].keys():
                        dataset[day] = data_store[day][storage_key][activity_key]

        days = sorted(dataset.keys())
        positions = sorted({key for day in dataset.values() for key in day.keys()})
        heatmap_data = np.zeros((len(positions), len(days)))
        for day_idx, day in enumerate(days):
            for int_idx, integer in enumerate(positions):
                heatmap_data[int_idx, day_idx] = dataset.get(day, {}).get(integer, 0)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=days,
            y=positions,
            colorscale='Viridis'
        ))

        # Update layout
        fig.update_layout(
            title=plot_title,
            xaxis_title='day Number',
            yaxis_title=y_title,
        )
        return fig

    def create_variants_heatmap(self, day_numbers, data_store):
        dataset = dict()
        for day in day_numbers:
            if day in data_store.keys():
                if 'variant_occurences' in data_store[day].keys():
                    top_10_day = sorted(data_store[day]['variant_occurences'].values(), reverse=True)[0:10]
                    dataset[day] = top_10_day

        days = sorted(dataset.keys())
        positions = sorted({key for day in dataset.values() for key in day.keys()})
        heatmap_data = np.zeros((len(positions), len(days)))
        for day_idx, day in enumerate(days):
            for int_idx, integer in enumerate(positions):
                heatmap_data[int_idx, day_idx] = dataset.get(day, {}).get(integer, 0)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=days,
            y=positions,
            colorscale='Viridis'
        ))

        # Update layout
        fig.update_layout(
            title='test',
            xaxis_title='day Number',
            yaxis_title='test',
        )
        return fig



    def _create_2_level_bar_chart(self, day_number, data_store, storage_key, x_title, plot_title):
        first_key = list(data_store.keys())[0]
        if day_number in data_store.keys():
            if len(data_store[day_number].keys()) > 0:
                data = {
                    x_title: list(data_store[day_number][storage_key].keys()),
                    'Amount': list(data_store[day_number][storage_key].values())
                }
            else:
                data = {
                    x_title: list(data_store[first_key][storage_key].keys()),
                    'Amount': [0 for dict_item in data_store[first_key][storage_key].values()]
                }
        else:
            data = {
                x_title: list(data_store[first_key][storage_key].keys()),
                'Amount': [0 for dict_item in data_store[first_key][storage_key].values()]
            }
        fig = px.bar(data, x=x_title, y='Amount', title=plot_title)
        return fig

    def _create_attribute_stacked_bar(self, day_number, data_store, storage_key, x_title, plot_title):
        attr_data = data_store[day_number][storage_key]
        colors = px.colors.qualitative.Plotly

        for attr in attr_data.keys():
            new_dict = dict()
            i = 0
            for attr_value in attr_data[attr].keys():
                if i == 10:
                    break
                new_dict[attr_value] = attr_data[attr][attr_value]
                i += 1
            attr_data[attr] = new_dict

        all_sub_keys = set()
        for sub_dict in attr_data.values():
            all_sub_keys.update(sub_dict.keys())
        categories = list(attr_data.keys())
        traces = []
        for i, sub_key in enumerate(all_sub_keys):
            values = [attr_data[category].get(sub_key, 0) for category in categories]
            trace = go.Bar(
                name=sub_key,
                x=categories,
                y=values,
                marker=dict(color=colors[i % len(colors)])
            )
            traces.append(trace)

        # Create the figure
        fig = go.Figure(data=traces)

        # Update the layout to have stacked bars
        fig.update_layout(barmode='stack', title=plot_title, xaxis_title=x_title, yaxis_title='Occurence amount', showlegend=False)
        return fig


    def get_num_variants_plot(self, data_store, day_numbers):
        fig = self._create_single_line_plot(day_numbers, data_store, 'num_variants', 'N.o. variants', 'Number of variants by start day')
        return fig

    def get_num_cases_plot(self, data_store, day_numbers):
        fig = self._create_num_cases_plot(data_store, day_numbers, y_title='N.o. cases')
        return fig

    def get_case_duration_plot(self, data_store, day_numbers):
        fig = self._create_single_line_plot(day_numbers, data_store, 'mean_case_duration', 'Mean case duration', 'Avg. case duration by start day')
        return fig

    def get_case_arrival_plot(self, data_store, day_numbers):
        fig = self._create_single_line_plot(day_numbers, data_store, 'case_arrival_average', 'Avg. time difference between cases', 'Avg. time difference between cases by start day')
        return fig

    def get_rework_bar_chart(self, data_store, day_number):
        fig = self._create_2_level_bar_chart(day_number, data_store, 'rework_per_act', 'Activity', f'Amount of rework per activity for day {day_number}')
        return fig

    def get_rework_line_chart(self, data_store, day_numbers):
        fig = self._create_multi_line_chart(day_numbers, data_store, 'rework_per_act', 'Rework count', f'Amount of rework per activity by start day of case')
        return fig

    def get_start_activities_bar_chart(self, data_store, day_number):
        fig = self._create_2_level_bar_chart(day_number, data_store, 'start_activities', 'Activity', f'Count of activity being start activity for day {day_number}')
        return fig

    def get_start_activities_line_chart(self, data_store, day_numbers):
        fig = self._create_multi_line_chart(day_numbers, data_store, 'start_activities', 'Start activity count', f'Count of being start activity by start day of case')
        return fig

    def get_end_activities_bar_chart(self, data_store, day_number):
        fig = self._create_2_level_bar_chart(day_number, data_store, 'end_activities', 'Activity', f'Count of activity being end activity for day {day_number}')
        return fig

    def get_end_activities_line_chart(self, data_store, day_numbers):
        fig = self._create_multi_line_chart(day_numbers, data_store, 'end_activities', 'End activity count', f'Count of being end activity by start day of case')
        return fig

    def get_min_self_distances_bar_chart(self, data_store, day_number):
        fig = self._create_2_level_bar_chart(day_number, data_store, 'min_self_distances', 'Activity', f'Minimum self distance of activity for day {day_number}')
        return fig

    def get_min_self_distances_line_chart(self, data_store, day_numbers):
        fig = self._create_multi_line_chart(day_numbers, data_store, 'min_self_distances', 'Minimum self distance', f'Minimum self distance of activity by start day of case')
        return fig

    def get_num_case_attr_bar_chart(self, data_store, day_number):
        fig = self._create_2_level_bar_chart(day_number, data_store, 'num_case_attr_values', 'Attribute', f'Mean values of numerical case attributes for day {day_number}')
        return fig

    def get_num_case_attr_line_chart(self, data_store, day_numbers, attribute):
        fig = self._create_2_level_single_line_plot(day_numbers, data_store, 'num_case_attr_values', 'Mean value', f'Mean value of {attribute} value occurences by case start day', attribute)
        return fig

    def get_num_event_attr_bar_chart(self, data_store, day_number):
        fig = self._create_2_level_bar_chart(day_number, data_store, 'num_event_attr_values', 'Attribute', f'Mean values of numerical event attributes for day {day_number}')
        return fig

    def get_num_event_attr_line_chart(self, data_store, day_numbers, attribute):
        fig = self._create_2_level_single_line_plot(day_numbers, data_store, 'num_event_attr_values', 'Mean value', f'Mean value of {attribute} value occurences by case start day', attribute)
        return fig

    def get_cat_case_attr_stacked_bar_chart(self, data_store, day_number):
        fig = self._create_attribute_stacked_bar(day_number, data_store, 'cat_case_attr_values', 'Attribute', f'Occurence proportion of categorical case attributes for day {day_number}')
        return fig

    def get_cat_case_attr_line_chart(self, data_store, day_numbers, attribute):
        fig = self._create_2_level_multi_line_chart(day_numbers, data_store, 'cat_case_attr_values', 'Occurence amount', f'Count of {attribute} value occurences by case start day', attribute)
        return fig

    def get_cat_event_attr_line_chart(self, data_store, day_numbers, attribute):
        fig = self._create_2_level_multi_line_chart(day_numbers, data_store, 'cat_event_attr_values', 'Occurence amount', f'Count of {attribute} value occurences by case start day', attribute)
        return fig

    def get_cat_event_attr_stacked_bar_chart(self, data_store, day_number):
        fig = self._create_attribute_stacked_bar(day_number, data_store, 'cat_event_attr_values', 'Attribute', f'Occurence proportion of event case attributes for day {day_number}')
        return fig

    def get_activity_position_chart(self, data_store, day_numbers, activity):
        fig = self._create_activity_occurence_heatmap(day_numbers, data_store, 'activity_position_summaries', 'Count of position', f'Count of {activity} being at specified index in trace by case start day', activity)
        return fig