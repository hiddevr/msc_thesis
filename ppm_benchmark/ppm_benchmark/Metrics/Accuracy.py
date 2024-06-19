from ppm_benchmark.Models.BaseMetric import BaseMetric
from sklearn.metrics import accuracy_score


class Accuracy(BaseMetric):

    def __init__(self):
        super().__init__('Accuracy')

    def evaluate(self, predictions_dict, target_dict):
        predictions, targets = self.get_target_preds(predictions_dict, target_dict)
        accuracy = accuracy_score(targets, predictions)
        return accuracy
