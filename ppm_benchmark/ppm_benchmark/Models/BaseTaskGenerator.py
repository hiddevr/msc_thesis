from abc import ABC, abstractmethod


class BaseTaskGenerator(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def generate_task(self, normalized_data):
        pass
