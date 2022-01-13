import pandas as pd

from src.basewrappermodel import BaseWrapperModel


class LinearRegressionModel(BaseWrapperModel):
    """Skeleton for baseline scikit-learn Linear Model interface"""

    def fit(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass

    def tune(self):
        pass

    def predict_proba(self):
        pass

    def preprocess(self):
        df = pd.read_csv("../data/data.csv")

    def setup(self):
        pass


if __name__ == '__main__':
    m = LinearRegressionModel()
    m.preprocess()
