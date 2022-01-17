from pytest import mark

from mvalidators.data_schema import InputDataFrameSchema, PreprocessedDataSchema


@mark.smoke
@mark.linear_regression_schema
class LinearRegressionSchemaTests:

    @mark.data_schema
    def test_data_schema(self, get_dataframe):
        d = get_dataframe
        d = InputDataFrameSchema(d)
        assert ~d.empty

    def test_pre_processing_schema(self, get_dataframe):
        df = get_dataframe
        df = InputDataFrameSchema(df)
        df = PreprocessedDataSchema(df)  # TODO : Add cleaned up preprocessing script from EDA.
        assert ~df.empty

    def test_post_processing_schema(self):
        assert False

    def test_predict_schema(self):
        assert False

    def test_tune_schema(self):
        assert False

    def test_evaluate_schema(self):
        assert False
