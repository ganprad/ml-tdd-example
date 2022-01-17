import os
from typing import Optional
from pathlib import Path
import pandera
from pandera import SchemaModel
from pandera.typing import Series


from pandera.schemas import DataFrameSchema
from pydantic import BaseModel, validator
from mvalidators.constants import Constants, ZIPCODES


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


class InputDataFrameSchema(SchemaModel):
    """
    Input dataframe specification.

    """

    Id: Series[int] = pandera.Field(nullable=False)
    is_bad: Optional[Series[int]] = pandera.Field(nullable=False, le=1, ge=0)
    annual_inc: Series[float] = pandera.Field(nullable=True, le=900000.0, ge=2000.0)
    debt_to_income: Series[float] = pandera.Field(nullable=True, le=29.99, ge=0.0)
    delinq_2yrs: Series[float] = pandera.Field(nullable=True, le=11.0, ge=0.0)
    inq_last_6mths: Series[float] = pandera.Field(nullable=True, le=25.0, ge=0.0)
    mths_since_last_delinq: Series[float] = pandera.Field(nullable=True, le=120.0, ge=0.0)
    mths_since_last_record: Series[float] = pandera.Field(nullable=True, le=119.0, ge=0.0)
    collections_12_mths_ex_med: Series[float] = pandera.Field(nullable=True, eq=0.0)
    mths_since_last_major_derog: Series[int] = pandera.Field(nullable=True, le=3, ge=1)
    open_acc: Series[float] = pandera.Field(nullable=True, le=39.0, ge=1.0)
    pub_rec: Series[float] = pandera.Field(nullable=True, le=3.0, ge=0.0)
    revol_bal: Series[int] = pandera.Field(nullable=True, le=1207359, ge=0)
    revol_util: Series[float] = pandera.Field(nullable=True, le=100.6, ge=0.0)
    total_acc: Series[float] = pandera.Field(nullable=True, le=90, ge=1.0)

    emp_length: Series[str]
    pymnt_plan: Series[str] = pandera.Field(isin=["n", "y"])
    zip_code: Series[str] = pandera.Field(isin=set(ZIPCODES.replace("\n", "").split(",")))

    home_ownership: Series[str] = pandera.Field(isin=["RENT", "MORTGAGE", "OWN", "OTHER", "NONE"], coerce=True)
    verification_status: Series[str] = pandera.Field(
        isin={"VERIFIED - income", "not verified", "VERIFIED - income source"}, coerce=True
    )

    purpose_cat: Series[str] = pandera.Field(
        isin={
            "wedding small business",
            "car small business",
            "home improvement small business",
            "educational",
            "wedding",
            "home improvement",
            "car",
            "vacation",
            "credit card",
            "major purchase",
            "other",
            "debt consolidation",
            "major purchase small business",
            "small business",
            "medical",
            "house",
            "moving",
            "vacation small business",
            "renewable energy",
            "moving small business",
            "small business small business",
            "educational small business",
            "credit card small business",
            "debt consolidation small business",
            "house small business",
            "medical small " "business",
            "other small business",
        },
        coerce=True,
    )

    addr_state: Series[str] = pandera.Field(
        isin={
            "OH",
            "OK",
            "IA",
            "NC",
            "NM",
            "MA",
            "AZ",
            "PA",
            "UT",
            "NY",
            "TN",
            "KS",
            "MI",
            "RI",
            "KY",
            "DE",
            "SD",
            "WY",
            "NE",
            "OR",
            "NH",
            "DC",
            "LA",
            "WA",
            "CO",
            "WV",
            "IL",
            "WI",
            "ME",
            "SC",
            "AL",
            "MO",
            "TX",
            "AK",
            "IN",
            "ID",
            "FL",
            "CT",
            "GA",
            "VT",
            "NJ",
            "AR",
            "MS",
            "HI",
            "CA",
            "MN",
            "MT",
            "VA",
            "MD",
            "NV",
        },
        coerce=True,
    )

    initial_list_status: Series[str]
    policy_code: Series[str] = pandera.Field(isin=["PC1", "PC5", "PC3", "PC4", "PC2"], coerce=True)

class PreprocessedDataSchema(SchemaModel):
    """
    Preprocessed DataModel specification.
    """

    is_bad: Optional[Series[float]] = pandera.Field(nullable=False, coerce=True, le=1.0, ge=0.0)

    emp_length: Series[float] = pandera.Field(nullable=False, coerce=True, le=1.0, ge=0.0)
    annual_inc: Series[float] = pandera.Field(nullable=False, coerce=True, le=1.0, ge=0.0)
    delinq_2yrs: Series[float] = pandera.Field(nullable=False, coerce=True, le=1.0, ge=0.0)
    inq_last_6mths: Series[float] = pandera.Field(nullable=False, coerce=True, le=1.0, ge=0.0)
    mths_since_last_delinq: Series[float] = pandera.Field(nullable=False, coerce=True, le=1.0, ge=0.0)
    open_acc: Series[float] = pandera.Field(nullable=False, coerce=True, le=1.0, ge=0.0)
    pub_rec: Series[float] = pandera.Field(nullable=False, coerce=True, le=1.0, ge=0.0)
    revol_bal: Series[float] = pandera.Field(nullable=False, coerce=True, le=1.0, ge=0.0)
    total_acc: Series[float] = pandera.Field(nullable=False, coerce=True, le=1.0, ge=0.0)
    collections_12_mths_ex_med: Series[float] = pandera.Field(nullable=False, coerce=True, le=1.0, ge=0.0)
    mths_since_last_major_derog: Series[float] = pandera.Field(nullable=False, coerce=True, le=1.0, ge=0.0)

    home_ownership_MORTGAGE: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    home_ownership_NONE: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    home_ownership_OTHER: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    home_ownership_OWN: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    home_ownership_RENT: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    verification_status_VERIFIED__income: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    verification_status_VERIFIED__income_source: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    verification_status_not_verified: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    purpose_cat_car: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    purpose_cat_credit_card: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    purpose_cat_debt_consolidation: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    purpose_cat_educational: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    purpose_cat_home_improvement: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    purpose_cat_house: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    purpose_cat_major_purchase: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    purpose_cat_medical: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    purpose_cat_moving: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    purpose_cat_other: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    purpose_cat_renewable_energy: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    purpose_cat_small_business: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    purpose_cat_vacation: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    purpose_cat_wedding: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_AK: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_AL: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_AR: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_AZ: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_CA: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_CO: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_CT: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_DC: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_DE: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_FL: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_GA: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_HI: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_IA: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_ID: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_IL: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_IN: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_KS: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_KY: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_LA: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_MA: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_MD: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_ME: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_MI: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_MN: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_MO: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_MS: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_MT: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_NC: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_NE: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_NH: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_NJ: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_NM: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_NV: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_NY: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_OH: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_OK: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_OR: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_PA: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_RI: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_SC: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_SD: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_TN: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_TX: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_UT: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_VA: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_VT: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_WA: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_WI: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_WV: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)
    addr_state_WY: Series[int] = pandera.Field(nullable=False, coerce=True, le=1, ge=0)


# TODO:
# Input dataframe schema -> validate the inputs to the application -> preprocess -> Validate preprocessing outputs -> model
# Input schemas for model params.
# Input schemas for tune functiona
