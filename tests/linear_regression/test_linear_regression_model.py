from pytest import mark

from src.linear_regression_model import LinearRegressionModel


@mark.smoke
@mark.linear_regression
class LinearRegressionTests:
    model = LinearRegressionModel()

    @mark.fit
    def test_fit(self, get_dataframe):
        data = get_dataframe
        assert False

    @mark.evaluate
    def test_evaluate(self):
        assert False

    @mark.predict
    def test_predict(self):
        assert False

    @mark.tune
    def test_tune(self):
        assert False

    @mark.predict_proba
    def test_predict_proba(self):
        assert False
