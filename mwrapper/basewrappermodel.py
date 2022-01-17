from abc import ABC, abstractmethod


class BaseWrapperModel(ABC):
    """Example interface between input requirements and ML model"""
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def tune(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self):
        pass