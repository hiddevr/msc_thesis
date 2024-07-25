from ..Models.base_evaluator import BaseEvaluator
import plotly.graph_objs as go
import plotly.subplots as sp
import numpy as np
import pandas as pd


class OutcomeClassification(BaseEvaluator):

    def __init__(self, evaluation_dicts, metrics, task_type):
        baselines = {'naive_baseline': 'naive_baseline'}
        super().__init__(evaluation_dicts, metrics, task_type, baselines, False)

    def plot_by_train_act_distance(self, metric_name):
        self.evaluation_df = self.evaluation_df.dropna(subset=['model_prediction', 'prediction_target'])
        plot_metric = self.get_metric(metric_name)
        unique_tasks = self.evaluation_df['task_name'].unique()

        fig = sp.make_subplots(rows=len(unique_tasks), cols=1, shared_xaxes=False,
                               subplot_titles=unique_tasks, vertical_spacing=0.1)

        for i, task in enumerate(unique_tasks):
            task_df = self.evaluation_df[self.evaluation_df['task_name'] == task]
            train_distances = sorted(task_df['train_sequence_distance'].unique())

            naive_baseline_acc = []
            model_acc = []

            for dist in train_distances:
                sub_df = task_df[task_df['train_sequence_distance'] == dist]

                naive_baseline_accuracy = plot_metric.evaluate(sub_df['naive_baseline_prediction'],
                                                               sub_df['prediction_target'])
                naive_baseline_acc.append(naive_baseline_accuracy)

                model_accuracy = plot_metric.evaluate(sub_df['model_prediction'], sub_df['prediction_target'])
                model_acc.append(model_accuracy)

            fig.add_trace(go.Scatter(x=train_distances, y=naive_baseline_acc, mode='lines+markers',
                                     name='Naive Baseline', line=dict(color=self.color_mappings['naive_baseline']),
                                     legendgroup=f'group_{i}', showlegend=(i == 0)),
                          row=i + 1, col=1, secondary_y=False)

            fig.add_trace(go.Scatter(x=train_distances, y=model_acc, mode='lines+markers',
                                     name='Model', line=dict(color=self.color_mappings['model']),
                                     legendgroup=f'group_{i}', showlegend=(i == 0)),
                          row=i + 1, col=1, secondary_y=False)

            fig.update_xaxes(title_text="Train Sequence Distance", row=i + 1, col=1)
            fig.update_yaxes(title_text=f"{plot_metric.name}", row=i + 1, col=1, secondary_y=False)

        fig.update_layout(height=300 * len(unique_tasks), width=800,
                          title_text=f"Prediction {plot_metric.name} by Train Sequence Distance")

        fig.show()

    def plot_lass_bar(self):
        self.evaluation_df = self.evaluation_df.dropna(subset=['model_prediction', 'prediction_target'])
        plot_metric = self.get_metric('LASS')

        unique_tasks = self.evaluation_df['task_name'].unique()
        fig = sp.make_subplots(rows=len(unique_tasks), cols=1, subplot_titles=unique_tasks)

        for i, task in enumerate(unique_tasks):
            task_df = self.evaluation_df[self.evaluation_df['task_name'] == task]
            model_metrics = plot_metric.evaluate(task_df['model_prediction'], task_df['prediction_target'])
            naive_bl_metrics = plot_metric.evaluate(task_df['naive_baseline'], task_df['prediction_target'])

            x_labels = list(model_metrics.keys())
            model_values = list(model_metrics.values())
            naive_bl = list(naive_bl_metrics.values())

            fig.add_trace(
                go.Bar(name='Model Prediction', x=x_labels, y=model_values, marker_color=self.color_mappings['model'],
                       showlegend=(i == 0)),
                row=i + 1, col=1
            )
            fig.add_trace(
                go.Bar(name='Naive Baseline', x=x_labels, y=naive_bl, marker_color=self.color_mappings['naive_baseline'],
                       showlegend=(i == 0)),
                row=i + 1, col=1
            )

            fig.update_xaxes(tickfont=dict(size=6), row=i + 1, col=1)

        fig.update_layout(
            height=400 * len(unique_tasks),
            title_text="LASS for model predictions vs distance baseline",
            showlegend=True,
            barmode='group'
        )

        fig.show()

    def plot_by_fraction_completed(self, metric_name):
        self.evaluation_df = self.evaluation_df.dropna(subset=['model_prediction', 'prediction_target'])
        plot_metric = self.get_metric(metric_name)

        unique_tasks = self.evaluation_df['task_name'].unique()

        fig = sp.make_subplots(rows=len(unique_tasks), cols=1,
                               subplot_titles=unique_tasks,
                               shared_xaxes=True,
                               vertical_spacing=0.05)

        bin_edges = np.arange(0, 1.05, 0.05)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        for i, task in enumerate(unique_tasks, 1):
            task_df = self.evaluation_df[self.evaluation_df['task_name'] == task]

            task_df['bin'] = pd.cut(task_df['fraction_completed'], bins=bin_edges, labels=bin_centers)

            model_metrics = task_df.groupby('bin').apply(
                lambda x: plot_metric.evaluate(x['model_prediction'], x['prediction_target']))
            baseline_metrics = task_df.groupby('bin').apply(
                lambda x: plot_metric.evaluate(x['naive_baseline'], x['prediction_target']))

            fig.add_trace(
                go.Bar(x=bin_centers, y=model_metrics, name='Model', marker_color=self.color_mappings['model'],
                       offsetgroup=0), row=i, col=1
            )
            fig.add_trace(
                go.Bar(x=bin_centers, y=baseline_metrics, name='Baseline', marker_color=self.color_mappings['naive_baseline'],
                       offsetgroup=1), row=i, col=1
            )

            fig.update_yaxes(title_text=f'{plot_metric.name}', row=i, col=1)

        fig.update_layout(
            barmode='group',
            height=300 * len(unique_tasks),
            width=1000,
            title_text="Metric Comparison: Model vs Baseline",
            showlegend=False
        )

        fig.update_xaxes(title_text='Fraction Completed', row=len(unique_tasks), col=1)

        fig.add_trace(go.Bar(x=[None], y=[None], name='Model', marker_color=self.color_mappings['model'],
                             showlegend=True))
        fig.add_trace(go.Bar(x=[None], y=[None], name='Baseline', marker_color=self.color_mappings['naive_baseline'],
                             showlegend=True))

        fig.show()