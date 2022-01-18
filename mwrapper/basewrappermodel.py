from abc import ABC, abstractmethod


class BaseWrapperModel(ABC):
    """Interface between input requirements and ML model"""
    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def tune(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass