import os
from pathlib import Path
from pathlib import PosixPath

from pydantic import BaseModel, validator, conint
from pydantic.typing import Literal

N_JOBS = 1
N_SPLITS = 3
N_REPEATS = 1
N_TRIALS = 1
RANDOM_STATE = 42
VERBOSE = 0

# Ranges
C_MIN = 2e-2
C_MAX = 2e-1
ITER_MIN = 10
ITER_MAX = 100
L1_RATIO_MIN = 0.05
L1_RATIO_MAX = 0.95
DATA_FILENAME = "baseline"


class JobParam(BaseModel):
    is_test: bool
    fn: Literal[DATA_FILENAME] = DATA_FILENAME
    n_jobs: Literal[conint(gt=0)] = N_JOBS
    random_state: Literal[42] = RANDOM_STATE
    verbose: Literal[conint(ge=0)] = VERBOSE

    @validator("is_test")
    def set_dirs(cls, value):
        PKG_PATH = Path(__file__).parents[1].resolve()
        cls.data_dir = PKG_PATH / "data"
        if value == True:
            TEST_MODELS_DIR = PKG_PATH / "tests/models"
            cls.models_dir = TEST_MODELS_DIR
            assert os.path.exists(cls.models_dir)
            return value
        else:
            MODELS_DIR = PKG_PATH / "saved_models"
            cls.models_dir = MODELS_DIR
            assert os.path.exists(cls.models_dir)
            return value

    @validator("fn")
    def set_model_file(cls, value):
        filename = f"{value}_logistic_regression.joblib"
        cls.model_file = cls.models_dir / filename
        return value

    @validator("fn")
    def get_data_path(cls, value):
        cls.data_path = str(cls.data_dir / f"{value}_data.csv")
        assert os.path.exists(cls.data_path)
        return value


class TrainedModelExists(BaseModel):
    model_file: PosixPath

    @validator("model_file")
    def check_if_trained_model_exists(cls, value):
        if os.path.exists(value):
            return value
        else:
            raise ValueError(f"Trained model does not exist. Expected file path: {value}."
                             f"Fit a model on training validators if it doesn't exist.")


class ModelParam(BaseModel):
    """
    Data model for solver parameters
    """

    solver: Literal["saga"] = "saga"
    fit_intercept: bool = True
    warm_start: bool = True
    multi_class: Literal["multinomial"] = "multinomial"
    penalty: Literal["elasticnet"] = "elasticnet"
    class_weight: Literal["balanced"] = "balanced"
    metric: Literal["f1_score"] = "f1"


class OptunaCVParam(BaseModel):
    n_splits: Literal[conint(ge=2)] = N_SPLITS
    n_repeats: Literal[conint(gt=0)] = N_REPEATS
    n_trials: Literal[conint(gt=0)] = N_TRIALS
    refit: bool = True
    return_train_score: bool = True


class HyperC(BaseModel):
    name: str = "C"
    value: float = C_MIN
    min: Literal[C_MIN] = C_MIN
    max: Literal[C_MAX] = C_MAX

    @validator("value")
    def check_value(cls, value):
        if C_MIN <= value <= C_MAX:
            return value
        else:
            raise ValueError(f"Value not in range. expected range {C_MIN, C_MAX}")


class HyperMaxIter(BaseModel):
    name: str = "max_iter"
    value: int = ITER_MIN
    min: Literal[ITER_MIN] = ITER_MIN
    max: Literal[ITER_MAX] = ITER_MAX

    @validator("value")
    def check_range(cls, value):
        if ITER_MIN <= value <= ITER_MAX:
            return value
        else:
            raise ValueError(f"Value not in range. expected range {ITER_MIN, ITER_MAX}")


class HyperL1Ratio(BaseModel):
    name: str = "l1_ratio"
    value: float = L1_RATIO_MIN
    min: Literal[L1_RATIO_MIN] = L1_RATIO_MIN
    max: Literal[L1_RATIO_MAX] = L1_RATIO_MAX

    @validator("value")
    def check_range(cls, value):
        if L1_RATIO_MIN <= value <= L1_RATIO_MAX:
            return value
        else:
            raise ValueError(f"Value not in range. expected range {L1_RATIO_MIN, L1_RATIO_MAX}")


class HyperParam(BaseModel):
    C: HyperC = HyperC()
    max_iter: HyperMaxIter = HyperMaxIter()
    l1_ratio: HyperL1Ratio = HyperL1Ratio()
