import os
from pathlib import Path

from pandera.schemas import DataFrameSchema
from pydantic import BaseModel, validator

PKG_PATH = Path(__file__).parents[1].resolve()
DATA_PATH = str(PKG_PATH / "data/data.csv")


class Constants(BaseModel):
    data_file: str = DATA_PATH  # Check for a data file names data.csv

    @validator("data_file")
    def check_if_data_in_allowed_filenames(cls, value):
        FILENAMES = ["data", "retrain", "deployment"]
        if value in FILENAMES:
            return value
        else:
            raise ValueError(f"These are the allowed filenames for the training data: {FILENAMES}")


class DataFileValidator(BaseModel):
    data_file: str = Constants().data_file

    @validator("data_file")
    def check_if_training_data_exists(cls, value):
        PKG_PATH = Path(__file__).parents[1].resolve()
        DATA_PATH = str(PKG_PATH / f"data/{value}.csv")
        if os.path.exists(DATA_PATH):
            return DATA_PATH
        else:
            raise ValueError(f"No training data available. Expected data file : {Constants().data_file}")


class InputDataSchemaValidator(DataFrameSchema):
    # TODO: Perform elementary EDA and build dataframe schema
    pass

# TODO:
# Input dataframe schema -> validate the inputs to the application -> preprocess -> Validate preprocessing outputs -> model
# Input schemas for model params.
# Input schemas for tune functionality.
