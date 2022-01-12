import pandas
import pytest

from config import Config


def pytest_addoption(parser):
    parser.addoption("--fn", action="store", help="Data file for testing.")


@pytest.fixture(scope="session")
def get_data_path(request):
    return request.config.getoption("--fn")


@pytest.fixture(scope="function")
def get_dataframe(get_data_path):
    cfg = Config(get_data_path)
    data = pandas.read_csv(cfg.DATA_PATH)
    return data
