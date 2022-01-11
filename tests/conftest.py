from pathlib import Path

import pandas
import pytest

from validators.data_schema import Constants


@pytest.fixture(scope="session")
def get_data_path():
    return Constants().data_file

@pytest.fixture(scope="function")
def get_dataframe(get_data_path):
    data = pandas.read_csv(get_data_path)
    return data
