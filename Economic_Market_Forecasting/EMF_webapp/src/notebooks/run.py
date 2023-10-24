from utils.libraries import *
from utils.objects import *
from utils.functions import *
from definitions import *
from Economic_Market_Forecasting_ML import *

EMF_ml = Economic_Market_Forecasting()

ROOT_PATH = Path(ROOT_PATH)
DATA_PATH = r"C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data"
config_paths = [
    "config\Market_Data_Config.csv",
    "config\Economic_Data_Config.csv",
    "config\Calc_Data_Config.csv"
]
market_config = ROOT_PATH.joinpath(config_paths[0])
economic_config = ROOT_PATH.joinpath(config_paths[1]).abspath()
calc_config = ROOT_PATH.joinpath(config_paths[2]).abspath()
target_list = pd.read_csv(market_config, sep=";", header=0).loc[:, "Market"].to_list()[0:2]
# target_list = ['SP500', 'Gold', 'CorporateBonds']

model_pred = {}
y_pred = {}
y_test = {}
top_model = {}
full_test = {}

for target in target_list:
    print(">>Target: "+target)
    model_pred[target], y_pred[target], y_test[target], top_model[target], full_test[target] = EMF_ml.Economic_Market_Forecasting_ML(years=20, 
                                                                                                                                     period="W", 
                                                                                                                                     root_path=ROOT_PATH, 
                                                                                                                                     target=target, 
                                                                                                                                     cutoff_date="2023-01-01", 
                                                                                                                                     cross_val=5, 
                                                                                                                                     data_path=DATA_PATH)