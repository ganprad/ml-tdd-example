from pytest import mark


@mark.smoke
@mark.linear_regression_schema
class LinearRegressionSchemaTests:

    @mark.data_schema
    def test_data_schema(self, get_dataframe):
        d = get_dataframe
        assert False

    def test_pre_processing_schema(self):
        assert False

    def test_post_processing_schema(self):
        assert False

    def test_predict_schema(self):
        assert False

    def test_tune_schema(self):
        assert False

    def test_evaluate_schema(self):
        assert False
