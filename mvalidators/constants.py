from pathlib import Path
from pydantic.typing import Literal
from pydantic import BaseModel



PKG_PATH = Path(__file__).parents[2].resolve()
DATA_FILE = PKG_PATH / "data/data.csv"
ENCODER_DIR = PKG_PATH / "ml_wrapper/encoders"
MODELS_DIR = PKG_PATH / "ml_wrapper/saved_models"
MINMAX_ENCODER_FILENAME = ENCODER_DIR / "minmax.joblib"
ONEHOT_ENCODER_FILENAME = ENCODER_DIR / "onehot.joblib"

DROP = ["Id", "pymnt_plan", "zip_code", "initial_list_status", "mths_since_last_record"]

MINMAX = [
    "mths_since_last_delinq",
    "emp_length",
    "annual_inc",
    "debt_to_income",
    "inq_last_6mths",
    "pub_rec",
    "mths_since_last_major_derog",
    "collections_12_mths_ex_med",
    "delinq_2yrs",
    "revol_bal",
    "revol_util",
    "total_acc",
    "open_acc",
]

ONE_HOT = ["policy_code", "home_ownership", "verification_status", "purpose_cat", "addr_state"]

TARGET = ["is_bad"]


class Constants(BaseModel):
    data_file: Literal[DATA_FILE] = DATA_FILE
    encoder_dir: Literal[ENCODER_DIR] = ENCODER_DIR
    minmax_encoder_filename: Literal[MINMAX_ENCODER_FILENAME] = MINMAX_ENCODER_FILENAME
    onehot_encoder_filename: Literal[ONEHOT_ENCODER_FILENAME] = ONEHOT_ENCODER_FILENAME

    drop: Literal["Id", "pymnt_plan", "zip_code", "initial_list_status", "mths_since_last_record"] = DROP

    minmax: Literal[
        "mths_since_last_delinq",
        "emp_length",
        "annual_inc",
        "debt_to_income",
        "inq_last_6mths",
        "pub_rec",
        "mths_since_last_major_derog",
        "collections_12_mths_ex_med",
        "delinq_2yrs",
        "revol_bal",
        "revol_util",
        "total_acc",
        "open_acc",
    ] = MINMAX

    one_hot: Literal["policy_code", "home_ownership", "verification_status", "purpose_cat", "addr_state"] = ONE_HOT

    target: Literal["is_bad"] = TARGET
