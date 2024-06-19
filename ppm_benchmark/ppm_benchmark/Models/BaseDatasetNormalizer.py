from abc import ABC, abstractmethod


class BaseDatasetNormalizer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def normalize(self, df):
        pass
