from pathlib import Path
from pydantic.typing import Literal
from pydantic import BaseModel



PKG_PATH = Path(__file__).parents[1].resolve()
DATA_FILE = PKG_PATH / "data/data.csv"
ENCODER_DIR = PKG_PATH / "encoders"
MODELS_DIR = PKG_PATH / "saved_models"
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
