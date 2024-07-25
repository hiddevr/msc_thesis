from ppm_benchmark.Models.BaseMetric import BaseMetric
from scipy.stats import pearsonr
import numpy as np
import statistics


class Correlation(BaseMetric):

    def __init__(self):
        super().__init__('Correlation')

    def evaluate(self, predictions, targets):
        if len(predictions) < 2:
            return 0
        if statistics.variance(predictions) == 0:
            return 0
        else:
            correlation, p_value = pearsonr(list(predictions), list(targets))
        return correlation
