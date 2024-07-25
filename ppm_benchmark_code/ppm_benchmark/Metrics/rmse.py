from ppm_benchmark.Models.BaseMetric import BaseMetric
from sklearn.metrics import mean_squared_error


class RMSE(BaseMetric):

    def __init__(self):
        super().__init__('RMSE')

    def evaluate(self, predictions, targets):
        return mean_squared_error(list(targets), list(predictions), squared=False)
