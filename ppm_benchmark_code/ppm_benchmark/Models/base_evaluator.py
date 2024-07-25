from abc import ABC, abstractmethod
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
import numpy as np


class BaseEvaluator(ABC):

    def __init__(self, evaluation_dicts, metrics, task_type, baselines, use_probas):
        self.evaluation_df = pd.DataFrame(evaluation_dicts)
        self.metrics = metrics
        self.use_probas = use_probas
        self.task_type = task_type
        self.baselines = baselines

        self.init_baseline_preds(baselines)
        self.color_mappings = self.init_colors()

    def init_baseline_preds(self, baselines):
        for baseline_name in baselines.keys():
            baseline_col_name = baselines[baseline_name]

            bl_series = self.evaluation_df[baseline_col_name].apply(pd.Series)
            bl_series = bl_series.add_prefix(f'{baseline_name}_')
            self.evaluation_df[f'{baseline_name}_prediction'] = self.evaluation_df[baseline_col_name]
            self.evaluation_df = pd.concat([self.evaluation_df, bl_series], axis=1)
        return

    def add_predictions(self, task_name, predictions):
        pred_rows = []
        if self.use_probas:
            for prediction_probas in predictions:
                pred_row = dict()
                for target_name, proba in prediction_probas.items():
                    pred_row[f'model_{target_name}'] = proba
                pred_row['model_prediction'] = prediction_probas
                pred_rows.append(pred_row)
        else:
            pred_rows = [{'model_prediction': pred} for pred in predictions]

        task_indices = self.evaluation_df[self.evaluation_df['task_name'] == task_name].index
        preds_df = pd.DataFrame(pred_rows, index=task_indices)
        for column in preds_df.columns:
            if column in self.evaluation_df.columns:
                self.evaluation_df.loc[preds_df.index, column] = preds_df[column]
            else:
                self.evaluation_df[column] = preds_df[column]
        return

    def evaluate(self):
        self.evaluation_df = self.evaluation_df.dropna(subset=['model_prediction', 'prediction_target'])
        tasks = self.evaluation_df['task_name'].unique()
        results = []

        for task in tasks:
            eval_df = self.evaluation_df[self.evaluation_df['task_name'] == task].reset_index(drop=True)
            for metric in self.metrics:
                model_metric_value = metric.evaluate(eval_df['model_prediction'], eval_df['prediction_target'])
                baseline_values = dict()
                for baseline_name in self.baselines.keys():
                    baseline_metric_value = metric.evaluate(eval_df[f'{baseline_name}_prediction'], eval_df['prediction_target'])
                    baseline_values[baseline_name] = baseline_metric_value

                result_dict = {
                    'task_name': task,
                    'metric': metric.name,
                    'model': model_metric_value,
                }
                result_dict.update(baseline_values)
                results.append(result_dict)
        return results

    def init_colors(self):
        colors = ['blue', 'red', 'green', 'orange', 'yellow']
        models = ['model']
        baselines = [name for name in self.baselines.keys()]
        models.extend(baselines)
        color_mappings = {model: color for model, color in zip(models, colors)}
        return color_mappings

    def get_metric(self, metric_name):
        plot_metric = None
        for metric in self.metrics:
            if metric.name == metric_name:
                plot_metric = metric
                break
        return plot_metric

    def plot_by_attr_drift_column(self, metric_name):
        self.evaluation_df = self.evaluation_df.dropna(subset=['model_prediction', 'prediction_target'])
        plot_metric = self.get_metric(metric_name)

        drift_columns = [col for col in self.evaluation_df.columns if col.startswith('attr_drift_')]
        unique_tasks = self.evaluation_df['task_name'].unique()

        penalties = {task: {} for task in unique_tasks}

        for task in unique_tasks:
            task_df = self.evaluation_df[self.evaluation_df['task_name'] == task]

            overall_accuracy = plot_metric.evaluate(task_df['model_prediction'], task_df['prediction_target'])

            for col in drift_columns:
                attribute_name = col.replace('attr_drift_', '')
                non_nan_df = task_df[task_df[col].notna()]
                if len(non_nan_df) > 0:
                    accuracy = plot_metric.evaluate(non_nan_df['model_prediction'], non_nan_df['prediction_target'])
                    weighted_accuracy_penalty = (overall_accuracy - accuracy)
                else:
                    weighted_accuracy_penalty = 0
                penalties[task][attribute_name] = weighted_accuracy_penalty

        fig = sp.make_subplots(rows=len(unique_tasks), cols=1, subplot_titles=unique_tasks, vertical_spacing=0.1, shared_xaxes=False)

        for i, task in enumerate(unique_tasks):
            attributes = list(penalties[task].keys())
            penalty_values = list(penalties[task].values())

            bar_data = go.Bar(x=attributes, y=penalty_values, marker_color=self.color_mappings['model'], showlegend=False)

            fig.add_trace(bar_data, row=i + 1, col=1)
            fig.update_yaxes(title_text=f"{plot_metric.name} Penalty", row=i + 1, col=1)
            fig.update_xaxes(title_text="Attribute", row=i + 1, col=1, tickmode='array', tickvals=list(range(len(attributes))), ticktext=attributes)

        fig.update_layout(height=300 * len(unique_tasks),
                          title_text=f'{plot_metric.name} Penalty by Drift Attribute for Each Task',
                          showlegend=False)

        fig.show()

