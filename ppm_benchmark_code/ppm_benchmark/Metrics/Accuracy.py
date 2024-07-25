from ppm_benchmark.Models.BaseMetric import BaseMetric
from sklearn.metrics import accuracy_score


class Accuracy(BaseMetric):

    def __init__(self):
        super().__init__('Accuracy')

    def _predictions_from_probas(self, prediction_probas):
        return [max(d, key=d.get) for d in prediction_probas]

    def evaluate(self, predictions, targets):
        predictions = self._predictions_from_probas(predictions)
        return accuracy_score(list(targets), list(predictions))
