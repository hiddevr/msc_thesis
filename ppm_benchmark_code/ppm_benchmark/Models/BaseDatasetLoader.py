from abc import ABC, abstractmethod


class BaseDatasetLoader(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def load_data(self, location):
        pass
