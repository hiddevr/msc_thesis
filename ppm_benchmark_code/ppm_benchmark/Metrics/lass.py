# Label Ambiguity Sensitivity Score (LASS)
from ppm_benchmark.Models.BaseMetric import BaseMetric


class LASS(BaseMetric):

    def __init__(self):
        super().__init__('LASS')

    def evaluate(self, predictions, targets):
        N = len(predictions)
        if N == 0:
            return None

        labels = set(predictions.iloc[0].keys()).union(set(targets.unique()))

        Rj_dict = {}
        for label in labels:
            Nj = len([pred for pred, target in zip(predictions, targets) if target == label])
            if Nj == 0:
                continue
            if Nj == N:
                Rj_dict[label] = 1
                continue

            pij_lst = [pred[label] if label in pred.keys() else 0 for pred in predictions]
            yij_lst = [1 if target == label else 0 for target in targets]

            numerator = sum(pij * (1 - yij) for pij, yij in zip(pij_lst, yij_lst)) * (1 / (N - Nj))
            denominator = sum(pij * yij for pij, yij in zip(pij_lst, yij_lst)) * (1 / Nj)
            Rj_dict[label] = numerator / denominator if denominator != 0 else 1

        return Rj_dict
