import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, validator
PKG_PATH = Path(__file__).parents[1].resolve()
DATA_PATH = str(PKG_PATH / "data/data.csv")


class Constants(BaseModel):
    data_file: Literal[DATA_PATH] = DATA_PATH # Check for a data file names data.csv

class DataFileValidator(BaseModel):
    data_file: str = Constants().data_file

    @validator("data_file")
    def check_if_training_data_exists(cls, value):
        if os.path.exists(value):
            return value
        else:
            raise ValueError(f"No training data available. Expected data file : {Constants().data_file}")


#TODO:
# Input dataframe schema -> validate the inputs to the application -> preprocess -> Validate preprocessing outputs -> model
# Input schemas for model params.
# Input schemas for tune functionality.