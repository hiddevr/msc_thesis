from ppm_benchmark.Models.BaseMetric import BaseMetric
from fastDamerauLevenshtein import damerauLevenshtein


class DamerauLevenshtein(BaseMetric):

    def __init__(self):
        super().__init__('DamerauLevenshtein')

    def evaluate(self, predictions, targets):
        return damerauLevenshtein(list(targets), list(predictions), similarity=False)

