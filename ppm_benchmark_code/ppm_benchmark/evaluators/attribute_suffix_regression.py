from ..Models.base_evaluator import BaseEvaluator
import plotly.graph_objs as go
import plotly.subplots as sp
import numpy as np
import pandas as pd


class AttributeSuffixRegression(BaseEvaluator):

    def __init__(self, evaluation_dicts, metrics, task_type):
        baselines = {'baseline': 'baseline'}
        super().__init__(evaluation_dicts, metrics, task_type, baselines, False)