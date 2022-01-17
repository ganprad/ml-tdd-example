import os

from pytest import mark
import hypothesis
import hypothesis.strategies as strategies
from scipy.stats import stats

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


@mark.smoke
@mark.linear_regression
class LinearRegressionTests:
    @mark.fit
    def test_fit(self, test_df, test_model):
        target = Constants().target
        x, y = test_df.drop(target, axis=1), test_df[target].values
        if os.path.exists(test_model.job_parameters.model_file):
            os.remove(test_model.job_parameters.model_file)
        test_model.fit(x, y)
        assert os.path.exists(test_model.job_parameters.model_file)


    @mark.evaluate
    def test_evaluate(self, test_df, test_model):
        target = Constants().target
        x, y = test_df.drop(target, axis=1), test_df[target].values
        results = test_model.evaluate(x, y)
        assert results["auc"] > 0.6

    @mark.predict
    def test_predict(self, test_df, test_model):
        target = Constants().target
        x, y = test_df.drop(target, axis=1), test_df[target].values
        preds = test_model.predict(x)
        assert stats.ttest_ind(y, preds)[1] <= 0.05

    @mark.tune
    def test_tune(self, test_df, test_model):
        target = Constants().target
        x, y = test_df.drop(target, axis=1), test_df[target].values
        if os.path.exists(test_model.job_parameters.model_file):
            os.remove(test_model.job_parameters.model_file)
        _ = test_model.tune(x, y)
        assert os.path.exists(test_model.job_parameters.model_file)

    @mark.predict_proba
    def test_predict_proba(self, test_df, test_model):
        target = Constants().target
        x, y = test_df.drop(target, axis=1), test_df[target].values
        preds = test_model.predict_proba(x)
        preds = preds[:, 1]
        t_test = stats.ttest_ind_from_stats(mean1=y.mean(), mean2=preds.mean(), std1=y.std(), std2=preds.std(),
            nobs1=y.shape[0], nobs2=preds.shape[0])
        p_val = t_test[1]
        assert p_val <= 0.05


@mark.data_model
@hypothesis.given(strategies.builds(JobParam))
def test_job_parameter_datamodel(data):
    assert isinstance(data, JobParam)

@mark.data_model
@hypothesis.given(
    strategies.builds(
        HyperParam,
        C=strategies.builds(HyperC, value=strategies.floats(min_value=HyperC().min, max_value=HyperC().max)),
        max_iter=strategies.builds(
            HyperMaxIter, value=strategies.integers(min_value=HyperMaxIter().min, max_value=HyperMaxIter().max)
        ),
        l1_ratio=strategies.builds(
            HyperL1Ratio, value=strategies.floats(min_value=HyperL1Ratio().min, max_value=HyperL1Ratio().max)
        ),
    )
)
@mark.data_model
def test_hyper_parameter_datamodel(data):
    assert isinstance(data, HyperParam)

@mark.data_model
@hypothesis.given(strategies.builds(ModelParam))
def test_solver_parameters_datamodel(data):
    assert isinstance(data, ModelParam)


@mark.data_model
@hypothesis.given(strategies.builds(OptunaCVParam))
def test_optuna_cv_parameters_datamodel(data):
    assert isinstance(data, OptunaCVParam)


