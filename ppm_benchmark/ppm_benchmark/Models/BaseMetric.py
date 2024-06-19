from abc import ABC, abstractmethod

class BaseMetric(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def evaluate(self, predictions, true_target):
        pass

    def get_target_preds(self, predictions_dict, targets_dict):
        if set(predictions_dict.keys()) != set(targets_dict.keys()):
            raise ValueError("The case IDs in the predictions and true target dictionaries do not match.")
        sorted_case_ids = sorted(predictions_dict.keys())
        predictions = [predictions_dict[case_id] for case_id in sorted_case_ids]
        targets = [targets_dict[case_id] for case_id in sorted_case_ids]
        return predictions, targets
