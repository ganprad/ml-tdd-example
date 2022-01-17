import os.path
from pathlib import Path


class Config:
    def __init__(self, file):
        FILENAMES = ["data", "retrain", "deployment"]
        assert file in FILENAMES
        PKG_PATH = Path(__file__).parents[1].resolve()
        TEST_MODELS_DIR = PKG_PATH / "tests/models"
        DATA_PATH = PKG_PATH / f"data/{file}.csv"
        fn_data_path_constants_map = {
            f"{file}": DATA_PATH}
        fn_model_path_constants_map = {
            f"{file}":  str(TEST_MODELS_DIR / f"test_{file}_logistic_regression.joblib")}
        assert os.path.exists(fn_data_path_constants_map[file])
        self.DATA_PATH = fn_data_path_constants_map[file]
        self.TEST_MODEL_FILE = fn_model_path_constants_map[file]

