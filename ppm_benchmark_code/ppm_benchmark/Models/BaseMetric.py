from abc import ABC, abstractmethod
import plotly.graph_objs as go
import plotly.subplots as sp
import numpy as np
import pandas as pd


class BaseMetric(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def evaluate(self, predictions, targets):
        pass
