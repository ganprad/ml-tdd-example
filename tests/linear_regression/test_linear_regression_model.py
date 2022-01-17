from pytest import mark

from mwrapper.linear_regression_model import LinearRegressionModel

from pandera import SchemaModel
from pydantic import BaseModel, BaseConfig

from mvalidators.data_schema import InputDataFrameSchema, PreprocessedDataSchema, Constants
from mvalidators.linear_regression_model_schema import (
    JobParam,
    HyperParam,
    HyperC,
    HyperL1Ratio,
    HyperMaxIter,
    ModelParam,
    OptunaCVParam,
)


def test_if_datamodels_are_valid():
    assert issubclass(InputDataFrameSchema, SchemaModel)
    assert issubclass(PreprocessedDataSchema, SchemaModel)
    assert issubclass(HyperParam, BaseModel)
    assert issubclass(HyperC, BaseModel)
    assert issubclass(HyperMaxIter, BaseModel)
    assert issubclass(HyperL1Ratio, BaseModel)
    assert issubclass(JobParam, BaseModel)
    assert issubclass(OptunaCVParam, BaseModel)
    assert issubclass(ModelParam, BaseModel)
    assert issubclass(HyperParam.__config__, BaseConfig)
    assert issubclass(HyperC.__config__, BaseConfig)
    assert issubclass(HyperMaxIter.__config__, BaseConfig)
    assert issubclass(HyperL1Ratio.__config__, BaseConfig)
    assert issubclass(JobParam.__config__, BaseConfig)
    assert issubclass(Constants, BaseModel)
    assert issubclass(Constants.__config__, BaseConfig)
    assert issubclass(OptunaCVParam.__config__, BaseConfig)
    assert issubclass(ModelParam.__config__, BaseConfig)


# TODO:Hidden
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


