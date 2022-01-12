import os.path
from pathlib import Path


class Config:
    def __init__(self, file):
        FILENAMES = ["data", "retrain", "deployment"]
        assert file in FILENAMES
        PKG_PATH = Path(__file__).parents[1].resolve()
        DATA_PATH = str(PKG_PATH / f"data/{file}.csv")
        assert os.path.exists()
        self.DATA_PATH = DATA_PATH
