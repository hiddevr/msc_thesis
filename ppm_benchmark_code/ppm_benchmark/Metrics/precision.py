from ppm_benchmark.Models.BaseMetric import BaseMetric
from sklearn.metrics import precision_score


class Precision(BaseMetric):

    def __init__(self):
        super().__init__('Precision')

    def _predictions_from_probas(self, prediction_probas):
        return [max(d, key=d.get) for d in prediction_probas]

    def evaluate(self, predictions, targets):
        predictions = self._predictions_from_probas(predictions)
        return precision_score(list(targets), list(predictions), average='weighted')
