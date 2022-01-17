import pandas
import pytest

from config import Config
from mvalidators.linear_regression_model_schema import HyperParam, ModelParam, JobParam, OptunaCVParam


def pytest_addoption(parser):
    parser.addoption("--fn", action="store", help="Data file for testing.")


@pytest.fixture(scope="session")
def get_data_path(request):
    return request.config.getoption("--fn")  # Add other types of data for deployment, retraining, known edge cases etc.


@pytest.fixture(scope="function")
def get_dataframe(get_data_path):
    cfg = Config(get_data_path)
    data = pandas.read_csv(cfg.DATA_PATH)
    return data


@pytest.fixture(scope="function")
def test_model(test_model_file):
    hyper_parameters = HyperParam()
    model_parameters = ModelParam()
    job_parameters = JobParam(model_file=test_model_file)
    optuna_cv_parameters = OptunaCVParam()
    return LendingClubSKLearnLR(hyper_parameters, model_parameters, job_parameters, optuna_cv_parameters)
