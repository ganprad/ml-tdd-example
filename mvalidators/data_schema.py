import os
from pathlib import Path
import os
from typing import Optional
from pathlib import Path
import pandera
from pandera import SchemaModel
from pandera.typing import Series
from pydantic.typing import Literal
from pydantic import BaseModel, validator


from pandera.schemas import DataFrameSchema
from pydantic import BaseModel, validator

PKG_PATH = Path(__file__).parents[1].resolve()
DATA_PATH = str(PKG_PATH / "data/data.csv")

# Expected Constants:
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

# categorical columns:
ZIPCODES = """766xx,660xx,916xx,124xx,439xx,200xx,103xx,891xx,612xx,926xx,921xx,980xx,198xx,920xx,604xx,324xx,076xx,
925xx,606xx,741xx,234xx,908xx,038xx,236xx,142xx,010xx,614xx,918xx,287xx,972xx,304xx,232xx,802xx,932xx,070xx,088xx,
210xx,060xx,633xx,978xx,535xx,335xx,857xx,115xx,104xx,207xx,272xx,913xx,951xx,774xx,221xx,954xx,971xx,554xx,453xx,
483xx,188xx,601xx,790xx,995xx,275xx,333xx,720xx,100xx,787xx,791xx,330xx,334xx,700xx,209xx,958xx,930xx,331xx,760xx,
297xx,201xx,309xx,208xx,306xx,432xx,173xx,946xx,223xx,551xx,984xx,296xx,080xx,952xx,752xx,170xx,310xx,021xx,982xx,
013xx,967xx,455xx,852xx,301xx,903xx,806xx,481xx,856xx,256xx,157xx,144xx,063xx,773xx,799xx,761xx,662xx,600xx,440xx,
402xx,112xx,900xx,782xx,605xx,762xx,706xx,703xx,302xx,907xx,181xx,945xx,286xx,922xx,268xx,750xx,775xx,029xx,024xx,
935xx,111xx,941xx,785xx,105xx,631xx,917xx,997xx,347xx,246xx,231xx,902xx,077xx,113xx,067xx,320xx,106xx,350xx,712xx,
454xx,295xx,290xx,083xx,959xx,928xx,352xx,480xx,968xx,566xx,735xx,730xx,068xx,238xx,241xx,919xx,609xx,303xx,553xx,
064xx,336xx,014xx,543xx,890xx,576xx,648xx,801xx,950xx,177xx,146xx,871xx,219xx,488xx,140xx,211xx,906xx,273xx,658xx,
085xx,983xx,421xx,087xx,016xx,795xx,127xx,018xx,030xx,117xx,235xx,244xx,548xx,437xx,356xx,557xx,128xx,027xx,065xx,
074xx,616xx,863xx,570xx,152xx,110xx,988xx,180xx,837xx,711xx,329xx,299xx,765xx,923xx,325xx,933xx,245xx,724xx,770xx,
800xx,940xx,130xx,756xx,905xx,194xx,934xx,550xx,300xx,107xx,990xx,341xx,114xx,278xx,726xx,846xx,346xx,705xx,721xx,
444xx,212xx,337xx,442xx,326xx,641xx,992xx,956xx,571xx,914xx,815xx,216xx,280xx,400xx,011xx,079xx,974xx,452xx,406xx,
125xx,164xx,328xx,199xx,081xx,260xx,228xx,598xx,189xx,023xx,071xx,158xx,274xx,237xx,129xx,491xx,283xx,540xx,850xx,
230xx,086xx,358xx,066xx,716xx,376xx,577xx,710xx,559xx,853xx,981xx,191xx,731xx,342xx,217xx,403xx,171xx,206xx,652xx,
613xx,748xx,441xx,498xx,430xx,327xx,985xx,953xx,820xx,174xx,549xx,082xx,960xx,646xx,957xx,120xx,532xx,360xx,486xx,
542xx,226xx,344xx,841xx,824xx,546xx,028xx,468xx,622xx,490xx,020xx,222xx,840xx,825xx,482xx,254xx,314xx,317xx,751xx,
101xx,218xx,276xx,410xx,620xx,977xx,119xx,740xx,947xx,927xx,626xx,193xx,131xx,339xx,996xx,398xx,025xx,495xx,456xx,
261xx,053xx,316xx,359xx,233xx,279xx,783xx,948xx,357xx,190xx,322xx,534xx,811xx,282xx,611xx,701xx,109xx,450xx,108xx,
313xx,425xx,075xx,707xx,220xx,804xx,292xx,630xx,780xx,323xx,831xx,031xx,354xx,975xx,619xx,936xx,910xx,404xx,037xx,
229xx,895xx,847xx,991xx,737xx,931xx,851xx,349xx,608xx,484xx,073xx,243xx,153xx,986xx,786xx,176xx,563xx,122xx,915xx,
361xx,763xx,657xx,663xx,970xx,061xx,150xx,185xx,242xx,538xx,531xx,805xx,640xx,955xx,939xx,911xx,308xx,017xx,949xx,
494xx,610xx,381xx,436xx,294xx,489xx,015xx,617xx,541xx,560xx,240xx,904xx,592xx,284xx,777xx,797xx,121xx,722xx,116xx,
136xx,145xx,629xx,338xx,197xx,305xx,754xx,492xx,708xx,019xx,912xx,809xx,496xx,875xx,434xx,759xx,253xx,645xx,184xx,
195xx,672xx,365xx,154xx,032xx,050xx,133xx,054xx,285xx,591xx,362xx,530xx,163xx,259xx,497xx,351xx,321xx,147xx,973xx,
500xx,293xx,446xx,062xx,794xx,602xx,435xx,291xx,445xx,431xx,998xx,371xx,894xx,424xx,597xx,448xx,665xx,281xx,784xx,
370xx,137xx,674xx,937xx,161xx,368xx,628xx,897xx,149xx,625xx,363xx,884xx,893xx,829xx,123xx,394xx,182xx,880xx,026xx,
813xx,405xx,271xx,704xx,764xx,089xx,676xx,224xx,265xx,944xx,539xx,655xx,451xx,132xx,447xx,433xx,165xx,993xx,225xx,
745xx,670xx,757xx,828xx,961xx,315xx,816xx,364xx,624xx,187xx,160xx,427xx,078xx,443xx,594xx,767xx,723xx,727xx,134xx,
148xx,493xx,729xx,264xx,318xx,864xx,179xx,392xx,882xx,141xx,458xx,744xx,661xx,457xx,178xx,562xx,183xx,685xx,257xx,
830xx,634xx,943xx,976xx,607xx,252xx,793xx,668xx,126xx,118xx,262xx,366xx,803xx,639xx,267xx,810xx,401xx,175xx,544xx,
719xx,215xx,778xx,156xx,713xx,166xx,547xx,596xx,599xx,379xx,072xx,135xx,355xx,186xx,808xx,270xx,397xx,084xx,277xx,
618xx,056xx,255xx,307xx,319xx,499xx,635xx,860xx,407xx,409xx,258xx,666xx,069xx,251xx,469xx,057xx,034xx,035xx,565xx,
739xx,168xx,650xx,196xx,877xx,138xx,749xx,654xx,412xx,151xx,564xx,656xx,969xx,989xx,603xx,627xx,788xx,669xx,781xx,
796xx,527xx,827xx,636xx,420xx,651xx,388xx,288xx,874xx,653xx,479xx,673xx,139xx,247xx,537xx,678xx,391xx,717xx,167xx,
012xx,395xx,836xx,714xx,681xx,102xx,883xx,227xx,411xx,667xx,638xx,572xx,647xx,471xx,312xx,826xx,558xx,462xx,385xx,
214xx,298xx,423xx,390xx,041xx,415xx,768xx,843xx,728xx,743xx,051xx,798xx,593xx,664xx,545xx,266xx,159xx,007xx,637xx"""

PKG_PATH = Path(__file__).parents[2].resolve()
DATA_FILE = PKG_PATH / "data/data.csv"
ENCODER_DIR = PKG_PATH / "mwrapper/encoders"
MODELS_DIR = PKG_PATH / "mwrapper/saved_models"
MINMAX_ENCODER_FILENAME = ENCODER_DIR / "minmax.joblib"
ONEHOT_ENCODER_FILENAME = ENCODER_DIR / "onehot.joblib"


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
