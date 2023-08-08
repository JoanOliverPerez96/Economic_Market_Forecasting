from utils.libraries import *
from utils.objects import *
from utils.functions import *
from definitions import *

class Economic_Market_Forecasting:
    def __init__(self, data=None):
        """
        Initializes a new instance of the class with a data parameter.

        :param data: The data to be assigned to the instance variable, self.df.
        :type data: Any
        """
        self.df = data
    def Economic_Market_Forecasting_ML(self,years, period="W", root_path=ROOT_PATH, target="SP500", cutoff_date="2023-01-01", cross_val=5, data_path=None):
        ## Collecting & Preparing the data
        ### Setting up the configuration
        # Se utiliza un rango de 20 años para la extraccion de datos económicos
        timeframe = 365*years
        today = datetime.today()
        end = today.strftime("%Y-%m-%d")
        start = (today - dt.timedelta(days=timeframe)).strftime("%Y-%m-%d")
        periods = period

        medidas = ["mean", "median", "mode", "Min", "Percentil_25", "Percentil_75","Max", "var", "std", "skew", "kurt"]

        ROOT_PATH = Path(root_path)
        config_paths = [
            "config\Market_Data_Config.csv",
            "config\Economic_Data_Config.csv",
            "config\Calc_Data_Config.csv"
        ]
        market_config = ROOT_PATH.joinpath(config_paths[0])
        economic_config = ROOT_PATH.joinpath(config_paths[1]).abspath()
        calc_config = ROOT_PATH.joinpath(config_paths[2]).abspath()

        target_list = pd.read_csv(market_config, sep=";", header=0).loc[:, "Codigo"].to_list()

        markets_used = pd.read_csv(market_config, sep=";", header=0).loc[:, "Codigo"].to_list()
        # markets_used = ['SPY', 'GDX', 'BND']

        target = target

        # ML random seed
        seed = 2

        extract = True

        cutoff_date = cutoff_date
        ### Loading and extracting the data
        # Initialize the Economic Data Analyzer class
        eda = EconomicDataAnalyzer()
        # Load the Market Data
        print("> Load the market data config")
        market = eda.read_config(market_config)
        market_dict = eda.convert_to_dictionary(markets_used=markets_used)
        market_dict = market_dict['Market']
        # Load the economic data config
        print("> Load the economic data config")
        econ = eda.read_config(economic_config)
        fred_series_dict = eda.convert_to_dictionary(markets_used=None)
        fred_series_dict = fred_series_dict["Indicador"]
        calc = eda.read_config(calc_config)
        series_calc_dict = eda.convert_to_dictionary(markets_used=None)
        series_calc_dict = series_calc_dict["Indicador"]
        # Setting up the indicator dictionaries
        print("> Setting up the indicator dictionaries")
        indicators = {}
        for ind in list(econ["Tipo"].unique()):
            indicators[ind] = econ[econ["Tipo"] == ind]["Indicador"].to_list()
        if extract == True:
            # Extracting the indicator data
            print("> Extracting the indicator data")
            indicators_df = eda.indicator_extraction(fred_series_dict, series_calc_dict, root_path=ROOT_PATH)
            # Extracting the market data
            print("> Extracting the market data")
            stocks = list(market_dict.keys())
            market_df = eda.market_extraction(stocks, start, end, root_path=ROOT_PATH)
        else:
            print("No data extraction, reading data from data file")
            path = ROOT_PATH.joinpath('data', 'raw', 'indicators_df.csv')
            indicators_df = pd.read_csv(path)
            path = ROOT_PATH.joinpath('data', 'raw', 'market_df.csv')
            market_df = pd.read_csv(path)
        ## Preparing the data
        ### Data cleaning
        # Cleaning the indicator data
        print("> Cleaning the indicator data")
        df_indicators, df_indicators_cum, df_indicators_diff, df_indicators_rets, df_indicators_limpio = eda.limpiar_indicators(
            df_indicators=indicators_df, 
            indicator_dict=indicators, 
            resample=periods, 
            fill_method="ffill", 
            start=start, 
            end=end, 
            root_path=ROOT_PATH)
        # Cleaning the market data
        print("> Cleaning market data")
        df_market, df_markets_rets, df_markets_cum, df_markets_diff  = eda.limpiar_markets(
            markets_dict=market_dict,
            df_markets=market_df,
            resample=periods, 
            fill_method="ffill", 
            start=start, 
            end=end, 
            root_path=ROOT_PATH)
        ### Merge indicator and market data
        list_market_dfs = [df_market,df_markets_rets,df_markets_cum,df_markets_diff]
        list_indicators_dfs = [df_indicators_limpio,df_indicators_rets,df_indicators_cum,df_indicators_diff]

        df_all_data, df_all_data_rets, df_all_data_cum, df_all_data_diff = eda.merge_data(list_market_dfs, list_indicators_dfs, root_path=ROOT_PATH)
        ## Feature Engineering
        ### Remove Outliers 
        df = eda.remove_outliers(df_all_data_rets)
        ### Adding features
        df_all_data["CAPE Ratio"] = df_all_data["SP500"]/(df_all_data["Corporate Profits"]*0.01)
        df["CAPE Ratio"] = df_all_data["SP500"]/(df_all_data["Corporate Profits"]*0.01)

        # df_all_data["CAPE Ratio"].plot()
        def trend_line(df, name, deg=2):
            coef = np.polyfit(range(0,len(df[name])), df[name], deg)
            x_trend = np.linspace(0,len(df[name]),len(df[name]))
            y_trend = np.polyval(coef, x_trend)
            df = pd.DataFrame(y_trend, index=df.index, columns=[name])
            return df

        # df_all_data = pd.DataFrame()
        df_all_data["SP_GDP"] = df_all_data["SP500"]/(df_all_data["GDP"]*.01)
        df_all_data["SP_GDP_trend"] = trend_line(df_all_data, "SP_GDP", deg=5)
        df["SP_GDP"] = df_all_data["SP500"]/(df_all_data["GDP"]*.01)
        df["SP_GDP_trend"] = trend_line(df_all_data, "SP_GDP", deg=5)

        # df_all_data = pd.DataFrame()
        std = df_all_data["SP_GDP"].std()
        df_all_data["SP_GDP_1std"] = df_all_data["SP_GDP_trend"] + (std)
        df["SP_GDP_1std"] = df_all_data["SP_GDP_trend"] + (std)

        # df_all_data = df_all_data.copy()
        # df_ts = df_all_data.loc[:,df_all_data.columns.str.contains(f"t-")]
        # df_all_data.drop(df_ts.columns,axis=1,inplace=True)
        for ma in df_all_data.columns:
            df_all_data[f"{ma}_MA"] = df_all_data[[ma]].rolling(window=52).mean().fillna(method="ffill").fillna(method="bfill")
            df_all_data[f"{ma}_trend"] = trend_line(df_all_data[[ma]], ma, deg=6)
            df_all_data[f"{ma}_MA_trend_dif"] = df_all_data[f"{ma}_trend"] - df_all_data[f"{ma}_MA"]
            
            df[f"{ma}_MA"] = df_all_data[[ma]].rolling(window=52).mean().fillna(method="ffill").fillna(method="bfill")
            df[f"{ma}_trend"] = trend_line(df_all_data[[ma]], ma, deg=6)
            df[f"{ma}_MA_trend_dif"] = df_all_data[f"{ma}_trend"] - df_all_data[f"{ma}_MA"]
        ### Creating lags in the data
        list_data_dfs = [df_all_data,df_all_data_rets,df_all_data_cum,df_all_data_diff]

        df_all_lag_data, df_all_lag_data_rets, df_all_lag_data_cum, df_all_lag_data_diff = eda.lag_data(list_data_dfs, target, n_lags=12)
        df = eda.remove_outliers(df_all_lag_data_rets)
        
        ## Data Preprocessing
        econ_ml = Preprocessor()
        ### Feature Reduction
        #### Feature selection by correlation
        df_feat_corr = pd.DataFrame(df.corr().loc[target,:].sort_values(ascending=False))
        df_feat_relevant_corr = df_feat_corr[(df_feat_corr[target]>0.05) | (df_feat_corr[target]<-0.05)]
        df_feat_relevant_corr
        #### Indentifying the most important features
        ##### Splitting the data

        ##### Creating the baseline for feature importance
        baseline_models = econ_ml.define_baseline_models()

        X_train, X_test, y_train, y_test = econ_ml.train_test_split_data(data=df, target_col=target, test_size=0.15)
        model_results, baseline_preds, best_model, best_model_name = econ_ml.baseline_ml(target, X_train, X_test, y_train, y_test, baseline_models)

        print("> Performing feature importance analysis")
        df_top_data, feature_importance, top_feature_importance, score = econ_ml.feature_importance(target=target, 
                                                                                                        df_data=df.loc[:cutoff_date],
                                                                                                        model=best_model,
                                                                                                        accepted_importance=0.9)
        #### Feature removal
        def feature_removal(df, df_top_data, model_results, best_model_name, score):
            best_model_score = model_results.loc[best_model_name,"score"]
            if score > best_model_score*.9:
                print("We choose to remove "+str(len(df.columns)-len(df_top_data.columns))+" features")
                df = df_top_data.copy()
            else:
                print("We choose to keep the original df with "+str(len(df_top_data.columns))+" features")
            return df

        df = feature_removal(df, df_top_data, model_results, best_model_name, score)
        ## Performing Machine Learning
        ### Pick the best model
        X_train, X_test, y_train, y_test = econ_ml.train_test_split_data(data=df, target_col=target, test_size=0.15)
        model_results, baseline_preds, best_model, best_model_name = econ_ml.baseline_ml(target, X_train, X_test, y_train, y_test, baseline_models)
        print("> Performing Machine Learning")
        ### Define the grids
        params_RandomForest = {
            "n_estimators": [120],
            "max_depth": [3,5,10,15,17],
            "max_features": ["sqrt", 3, 4]                          
            }

        params_GradientBoosting = {
            'n_estimators': [50, 100, 150],  
            'learning_rate': [0.01, 0.05, 0.1],  
            'max_depth': [3, 5, 7],  
            }

        params_XGBRegressor = {
            'n_estimators': [50, 100, 150],  
            'learning_rate': [0.01, 0.05, 0.1],  
            'max_depth': [ 5, 7, 11],  
            'min_child_weight': [ 3, 5],  
            'subsample': [0.8, 1.0],  
            'colsample_bytree': [0.8, 1.0],  
            }

        params_KNeighborsRegressor = {
            'n_neighbors': [3, 5, 7, 9],  
            'weights': ['uniform', 'distance'],  
            'p': [1, 2],  
            
            }

        params_SVR = {
            'C': [0.1, 1.0, 10.0],  
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  
            'degree': [2, 3, 4],  
            'gamma': ['scale', 'auto', 0.1, 1.0],  
            }

        baseline_params = {
            "RandomForest":params_RandomForest,
            "GradientBoosting":params_GradientBoosting,
            "SVR":params_SVR,
            "KNeighborsRegressor":params_KNeighborsRegressor,
            "XGBRegressor":params_XGBRegressor
        }
        print(">>> Performing Grid Search")
        def model_gridSearch(baseline_models,baseline_params,model_results,X_train,y_train,X_test,y_test,cross_val=5):
            y_test = y_test.copy()
            models_gridsearch = {}
            for name, model in baseline_models.items():
                if name in model_results.index.values:
                    for mod,params in baseline_params.items():
                        if name == mod:
                            models_gridsearch[mod] = GridSearchCV(model, params, cv=cross_val, scoring="neg_root_mean_squared_error", verbose=1, n_jobs=1)
                            models_gridsearch[mod].fit(X_train, y_train)
            best_grids = [(i, j.best_score_) for i, j in models_gridsearch.items()]
            best_grids = pd.DataFrame(best_grids, columns=["Grid", "Best score"]).sort_values(by="Best score", ascending=False)
            y_pred = models_gridsearch[best_grids.loc[0,"Grid"]].predict(X_test)
            y_pred = pd.DataFrame(y_pred, columns=[target+"_Prediction"],index=y_test.index)
            y_pred.index, y_test.index = pd.to_datetime(y_test.index), pd.to_datetime(y_test.index)
            model_pred = pd.concat([y_test, y_pred], axis=1)
            top_model = models_gridsearch[best_grids.loc[0,"Grid"]]
            return models_gridsearch, best_grids, y_pred, y_test, model_pred, top_model
        models_gridsearch, best_grids, y_pred, y_test, model_pred, top_model = model_gridSearch(baseline_models,baseline_params,model_results,X_train,y_train,X_test,y_test,cross_val=cross_val)
        try:
            X_test.index = pd.to_datetime(X_test.index)
        except:
            pass
        full_test = pd.concat([model_pred, X_test], axis=1)
        print(">>> Saving the best model and the data")
        # Save the best model
        dump(top_model, r'C:\Users\Joan Oliver\Documents\GitHub\Economic_Market_Forecasting\Economic_Market_Forecasting\EMF_webapp\EMF_project\models'+f"\{target}_best_model.joblib")
        # Save the data
        model_pred.to_csv(data_path+f"\{target}_data.csv")

        return model_pred, y_pred, y_test, top_model, full_test