import pandas
import pytest

from config import Config
from mvalidators.linear_regression_model_schema import HyperParam, ModelParam, JobParam, OptunaCVParam
from mwrapper.linear_regression_model import LinearRegressionModel


def pytest_addoption(parser):
    parser.addoption("--fn", action="store", help="Data file for testing.")


@pytest.fixture(scope="session")
def get_fn(request):
    return request.config.getoption("--fn")  # Add other types of data for deployment, retraining, known edge cases etc.


@pytest.fixture(scope="function")
def test_df(get_fn):
    cfg = Config(get_fn)
    data = pandas.read_csv(cfg.DATA_PATH)
    return data


@pytest.fixture(scope="session")
def get_test_fpath(get_fn):
    cfg = Config(get_fn)
    return cfg.TEST_MODEL_FILE


@pytest.fixture(scope="function")
def test_model(get_test_fpath):
    hyper_parameters = HyperParam()
    model_parameters = ModelParam()
    job_parameters = JobParam(is_test=True, fn=get_test_fpath)
    optuna_cv_parameters = OptunaCVParam()
    return LinearRegressionModel(hyper_parameters, model_parameters, job_parameters, optuna_cv_parameters)
