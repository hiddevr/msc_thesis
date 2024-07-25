from ppm_benchmark.Models.BaseMetric import BaseMetric
from sklearn.metrics import mean_absolute_error


class MAE(BaseMetric):

    def __init__(self):
        super().__init__('MAE')

    def evaluate(self, predictions, targets):
        return mean_absolute_error(list(targets), list(predictions))
