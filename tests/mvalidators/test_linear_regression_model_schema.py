import hypothesis
from hypothesis import strategies as st
from pandera import SchemaModel
from pydantic import BaseModel, BaseConfig
from pytest import mark

from mvalidators.data_schema import InputDataFrameSchema, PreprocessedDataSchema, Constants
from mvalidators.linear_regression_model_schema import (JobParam, HyperParam, HyperC, HyperL1Ratio, HyperMaxIter,
                                                        ModelParam, OptunaCVParam, )


@mark.data_model
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


@mark.data_model
@hypothesis.given(st.builds(JobParam))
def test_job_parameter_datamodel(data):
    assert isinstance(data, JobParam)


@mark.data_model
@hypothesis.given(
    st.builds(HyperParam, C=st.builds(HyperC, value=st.floats(min_value=HyperC().min, max_value=HyperC().max)),
              max_iter=st.builds(HyperMaxIter,
                                 value=st.integers(min_value=HyperMaxIter().min, max_value=HyperMaxIter().max)),
              l1_ratio=st.builds(HyperL1Ratio,
                                 value=st.floats(min_value=HyperL1Ratio().min, max_value=HyperL1Ratio().max)), ))
@mark.data_model
def test_hyper_parameter_datamodel(data):
    """Checks the validity of input model hyper-parameters given their acceptable ranges.
    Hypothesis: Is there any way to find a set of values that will invalidate HyperParam data class.
    """
    assert isinstance(data, HyperParam)


@mark.data_model
@hypothesis.given(st.builds(ModelParam))
def test_solver_parameters_datamodel(data):
    """Hypothesis: Are there a set of values that will invalidate ModelParam data class."""
    assert isinstance(data, ModelParam)


@mark.data_model
@hypothesis.given(st.builds(OptunaCVParam))
def test_optuna_cv_parameters_datamodel(data):
    """Hypothesis: Are there a set of values that will invalidate OptunaCVParam data class"""
    assert isinstance(data, OptunaCVParam)
