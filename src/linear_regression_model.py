import pandas as pd

from src.basewrappermodel import BaseWrapperModel
from src.utils import drop_columns, preprocess_emp_length, preprocess_categorical_columns, preprocess_purpose_cat, \
    fillnans
from validators.data_schema import InputDataFrameSchema


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

    def preprocess(self, x):
        """
        Preprocessing function.

        :param x:pandas.DataFrame
        :return:pandas.DataFrame
        """

        valid_input_df = InputDataFrameSchema.validate(x)
        preprocessed_df = drop_columns(valid_input_df, self.constants.drop)
        preprocessed_df = preprocess_emp_length(preprocessed_df)
        preprocessed_df = preprocess_categorical_columns(preprocessed_df, self.constants.one_hot)
        preprocessed_df = preprocess_purpose_cat(preprocessed_df)

        for col in preprocessed_df.columns:
            preprocessed_df = fillnans(preprocessed_df, col)
        assert not pd.isnull(preprocessed_df).values.sum() > 0
        return preprocessed_df

    def setup(self):
        pass


if __name__ == '__main__':
    df = pd.read_csv("../data/data.csv")
    m = LinearRegressionModel()
    preprocessed_df = m.preprocess(df)


