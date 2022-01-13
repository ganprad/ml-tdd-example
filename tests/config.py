import os.path
from pathlib import Path


class Config:
    def __init__(self, file):
        FILENAMES = ["data", "retrain", "deployment"]
        assert file in FILENAMES
        PKG_PATH = Path(__file__).parents[1].resolve()
        fn_constants_map_dict = {
            f"{file}": f"../../data/{file}.csv"}
        assert os.path.exists(fn_constants_map_dict[file])
        self.DATA_PATH = fn_constants_map_dict[file]
