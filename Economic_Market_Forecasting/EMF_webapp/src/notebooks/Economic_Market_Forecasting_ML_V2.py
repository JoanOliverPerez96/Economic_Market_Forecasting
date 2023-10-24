from utils.libraries import *
from utils.objects import *
from utils.functions import *
from definitions import *

class Economic_Market_Forecasting:
    def __init__(self, data=None):
        self.df = data



    def Economic_Market_Forecasting_ML(self, years, period="W", date=datetime.today(), root_path=ROOT_PATH, target="SP500",ticker="SPY", accepted_importance=0.8, cutoff_date="2023-01-01", cross_val=5, data_path=None):
        ## Collecting & Preparing the data
        ### Setting up the configuration
        # Se utiliza un rango de 20 años para la extraccion de datos económicos
        timeframe = 365*years
        today = date
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

        cutoff_date=date.strftime('%Y-%m-%d')
        Ymd_str = date.strftime('%Y%m%d')
        Ym_str = date.strftime('%Y%m')
        Y_str = date.strftime('%Y')

        ## Paths variables
        PARENT_DIR = "C:/Users/Joan Oliver/Documents/GitHub/Economic_Market_Forecasting/Economic_Market_Forecasting/EMF_webapp/src/"
        DATA_FOLDER = "data/result/processed_data"
        PREDICT_FOLDER = "data/result/prediction_data"
        MODEL_FOLDER = "model"

        data_path = PARENT_DIR+"/"+ DATA_FOLDER+"/"+ Y_str+"/"+ Ym_str+"/"+ Ymd_str
        predict_path = PARENT_DIR+"/"+ PREDICT_FOLDER+"/"+ Y_str+"/"+ Ym_str+"/"+ Ymd_str
        model_path = PARENT_DIR+"/"+ MODEL_FOLDER+"/"+ Y_str+"/"+ Ym_str+"/"+ Ymd_str
        try:
            os.makedirs(data_path)
        except:
            print("No folder created: "+DATA_FOLDER)
        try:
            os.makedirs(predict_path)
        except:
            print("No folder created: "+PREDICT_FOLDER)
        try:
            os.makedirs(model_path)
        except:
            print("No folder created: "+MODEL_FOLDER)

        market_config = ROOT_PATH.joinpath(config_paths[0])
        economic_config = ROOT_PATH.joinpath(config_paths[1]).abspath()
        calc_config = ROOT_PATH.joinpath(config_paths[2]).abspath()
        target_list = pd.read_csv(market_config, sep=";", header=0).loc[:, "Codigo"].to_list()
        markets_used = pd.read_csv(market_config, sep=";", header=0).loc[:, "Codigo"].to_list()
        # markets_used = ['SPY', 'GDX', 'BND']
        target = target
        markets_remove = [x for x in markets_used if x != ticker]
        # ML random seed
        seed = 2
        extract = True
        cutoff_date = cutoff_date

        if os.path.exists(predict_path+f"\prediction_{target}_{Ymd_str}.csv"):
            print(f"{target} prediction exists: prediction_{target}_{Ymd_str}.csv")
            model_pred = None
            y_pred = None
            y_test = None
            top_model = None
            df_all_predictions = pd.read_csv(predict_path+f"/prediction_{target}_{Ymd_str}.csv", index_col="Date")
            df_future_preds = None
            return model_pred, y_pred, y_test, top_model, df_all_predictions, df_future_preds
        else:
            print(f"{target} prediction does not exist: prediction_{target}_{Ymd_str}.csv")
            if os.path.exists(data_path+f"/processed_data_{target}_{Ymd_str}.csv"):
                print(f"{target} extraction exists: processed_data_{target}_{Ymd_str}.csv")
                ## Reading the processed data
                df = pd.read_csv(data_path+f"/processed_data_{target}_{Ymd_str}.csv", index_col="Date")
                econ_ml = Preprocessor()
                baseline_models = econ_ml.define_baseline_models()
            else:
                print(f"{target} extraction does not exist: processed_data_{target}_{Ymd_str}.csv")
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
                # Preparing the data
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
                ### Feature Engineering
                ### Remove Outliers 
                df = eda.remove_outliers(df_all_data_rets)
                print(df.columns)

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
                    df_all_data[f"{ma}_std"] = df_all_data[[ma]].rolling(window=52).std().fillna(method="ffill").fillna(method="bfill")
                    df_all_data[f"{ma}_trend"] = trend_line(df_all_data[[ma]], ma, deg=6)
                    df_all_data[f"{ma}_MA_trend_dif"] = df_all_data[f"{ma}_trend"] - df_all_data[f"{ma}_MA"]
                    
                    df[f"{ma}_MA"] = df_all_data[[ma]].rolling(window=52).mean().fillna(method="ffill").fillna(method="bfill")
                    df[f"{ma}_trend"] = trend_line(df_all_data[[ma]], ma, deg=6)
                    df[f"{ma}_MA_trend_dif"] = df_all_data[f"{ma}_trend"] - df_all_data[f"{ma}_MA"]
                ### Creating lags in the data
                list_data_dfs = [df_all_data,df_all_data_rets,df_all_data_cum,df_all_data_diff]

                df_all_lag_data, df_all_lag_data_rets, df_all_lag_data_cum, df_all_lag_data_diff = eda.lag_data(list_data_dfs, target, n_lags=24)
                df = eda.remove_outliers(df_all_lag_data_rets)
                # Remove target columns from the data
                for mkt in markets_remove:
                    if mkt == ticker:
                        pass
                    else:
                        for df_col in df.columns:
                            if mkt in df_col:
                                try:
                                    df.drop(df_col, axis=1, inplace=True)
                                except:
                                    pass

                ### Feature Reduction
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
                                                                                                                accepted_importance=accepted_importance)
                #### Feature removal
                def feature_removal(df, df_top_data, model_results, best_model_name, score):
                    best_model_score = model_results.loc[best_model_name,"score"]
                    print("The best feature importance model score is: "+str(best_model_score))
                    print("The model we chose is: "+str(best_model_name))
                    if score > best_model_score*.9:
                        print("We choose to remove "+str(len(df.columns)-len(df_top_data.columns))+" features out of "+str(len(df.columns))+" for a total of "+str(len(df_top_data.columns))+" features")
                        df = df_top_data.copy()
                        print("The top features are: "+str(df_top_data.head().columns))
                    else:
                        print("We choose to keep the original df with "+str(len(df_top_data.columns))+" features")
                    return df

                df = feature_removal(df, df_top_data, model_results, best_model_name, score)
                ## Saving the processed data (ready for ML)
                df.to_csv(data_path+f"/processed_data_{target}_{Ymd_str}.csv", index=True, index_label="Date")

            # else:
            #     ## Reading the processed data
            #     df = pd.read_csv(data_path+f"/processed_data_{target}_{Ymd_str}.csv", index_col="Date")
            #     econ_ml = Preprocessor()
            #     baseline_models = econ_ml.define_baseline_models()




            # Performing Machine Learning
            ### Pick the best model
            ## Data Preprocessing
            econ_ml = Preprocessor()
            X_train, X_test, y_train, y_test = econ_ml.train_test_split_data(data=df, target_col=target, test_size=0.15)
            model_results, baseline_preds, best_model, best_model_name = econ_ml.baseline_ml(target, X_train, X_test, y_train, y_test, baseline_models)
            ### Define the grids
            params_RandomForest = {
                "n_estimators": [150, 250],
                "max_depth": [10,15,17],
                "max_features": ["sqrt", "log2", None],}
            params_GradientBoosting = {
                'n_estimators': [150, 250],  # 50, 
                'learning_rate': [0.01, 0.05, 0.1],  
                'max_depth': [5,10,17],}
            params_XGBRegressor = {
                'n_estimators': [150, 250],  # 100
                'learning_rate': [0.01, 0.05, 0.1],  
                'max_depth': [ 5, 10, 15],}
            params_KNeighborsRegressor = {
                'n_neighbors': [3, 5, 7, 9],  
                'weights': ['uniform', 'distance'],  
                'p': [1, 2],}
            params_SVR = {
                'C': [0.1, 1.0, 10.0],  
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  
                'degree': [2, 3, 4],  
                'gamma': ['scale', 'auto', 0.1, 1.0],}
            baseline_params = {
                "RandomForest":params_RandomForest,
                "GradientBoosting":params_GradientBoosting,
                "SVR":params_SVR,
                "KNeighborsRegressor":params_KNeighborsRegressor,
                "XGBRegressor":params_XGBRegressor}
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
                top_model = models_gridsearch[best_grids.loc[0,"Grid"]]
                # y_pred = models_gridsearch[best_grids.loc[0,"Grid"]].predict(X_test)
                print(best_grids)
                print("Best model: "+str(top_model)+" with score: "+str(best_grids.loc[0,"Best score"]))
                return models_gridsearch, best_grids, top_model
            ### Performing Grid Search
            models_gridsearch, best_grids, top_model = model_gridSearch(baseline_models,
                                                                        baseline_params,
                                                                        model_results,
                                                                        X_train,
                                                                        y_train,
                                                                        X_test,
                                                                        y_test,
                                                                        cross_val=cross_val)
            ### Creating the prediction df
            actual = pd.DataFrame(y_test)
            actual.index = pd.to_datetime(actual.index)
            pred = pd.DataFrame(top_model.best_estimator_.predict(X_test), index=pd.to_datetime(y_test.index), columns=[target + " pred"])
            # Saving the best model
            dump(top_model, model_path+f"\{target}_best_model_{Ymd_str}.joblib")
            # plt.plot(actual.cumsum())
            # plt.plot(pred.cumsum())
            # Forecasting Feature Data
            ### Monte Carlo Simulation
            # Calculating the stats
            mean = df.mean()
            var = df.var()
            drift = mean - (.5 * var)
            std = df.std()

            # Setting the Monte Carlo Varuables
            ind = 0
            T = 104
            num_ports = 100
            date_range = pd.date_range(start=cutoff_date, periods=T, freq="W")

            dict_future = {}
            df_mean_future = pd.DataFrame(index=pd.date_range(start=cutoff_date, periods=T, freq="W"))
            df_max_future = pd.DataFrame(index=pd.date_range(start=cutoff_date, periods=T, freq="W"))
            df_min_future = pd.DataFrame(index=pd.date_range(start=cutoff_date, periods=T, freq="W"))

            # Running through indicator columns to forecast
            for ind, col in enumerate(df.columns):
                # Calculating the Weekly Returns
                weekly_rets = np.exp(drift.values[ind] + std.values[ind] * norm.ppf(np.random.rand(T, num_ports)))

                # Getting the most current weekly return (run it back if it's too small)
                n = -1
                S0 = 0
                while (S0 < 0.01) and (S0 > -0.01):
                    S0 = df.cumsum().iloc[n,ind]
                    n = n - 1
                # Creating the empty list and filling the first row
                price_list = np.zeros_like(weekly_rets)
                price_list[0] = S0

                # Performing Monte Carlo Situlation a 'num_ports' number of times
                for t in range(1,T):
                    price_list[t] = price_list[t-1] * weekly_rets[t]
                    dict_future[col] = pd.DataFrame(price_list,index=date_range)
            ### Working on the forecast data
            ##### Creating a list of forecasted futures
            ### Each item is a dataframe of a possible future of all indicators

            list_futures = []
            for n in range(0,num_ports):
                globals()["df_future_"+str(n)] = pd.DataFrame()
                for indicator in dict_future.keys():
                    globals()["df_future_"+str(n)][indicator] = dict_future[indicator][n]
                list_futures.append(globals()["df_future_"+str(n)])
            ## Define a custom merging function
            # def merge_columns_with_nans(row):
            #     merged_values = []
            #     for value in row:
            #         if pd.notna(value):
            #             merged_values.append(value)
            #     return merged_values
            ##### Merging the forecasted data with the historical data
            df.index = pd.to_datetime(df.index)
            list_present_future = []
            for n,future in enumerate(list_futures):
                globals()["df_future_"+str(n)].index = pd.to_datetime(globals()["df_future_"+str(n)].index)
                df_presentVSfuture = pd.concat([df,globals()["df_future_"+str(n)].pct_change()], axis=1, join="outer")
                globals()["df_present_future_"+str(n)] = pd.DataFrame()
                for col in df_presentVSfuture.columns:
                    # df1 = df_presentVSfuture[[col]].apply(merge_columns_with_nans, axis=1).apply(pd.Series)
                    # globals()["df_present_future_"+str(n)][col] = df1
                    globals()["df_present_future_"+str(n)][col] = df_presentVSfuture[col].fillna(0).iloc[:,0] + df_presentVSfuture[col].fillna(0).iloc[:,1]
                list_present_future.append(globals()["df_present_future_"+str(n)])

            ### ML Prediction based on Monte Carlo Simulation
            def best_prediction(models_gridsearch, best_grids, top_model, X_test, y_test):
                y_pred = models_gridsearch[best_grids.loc[0,"Grid"]].predict(X_test)
                y_pred = pd.DataFrame(y_pred, columns=[target+"_Prediction"],index=y_test.index)
                y_pred.index, y_test.index = pd.to_datetime(y_test.index), pd.to_datetime(y_test.index)
                model_pred = pd.concat([y_test, y_pred], axis=1)
                return y_pred, y_test, model_pred

            df_prediction = df[[target]].cumsum()
            list_prediction = [df_prediction]
            for n, present_future in enumerate(list_present_future):
                df_pred = present_future.fillna(method="ffill")
                
                test_size = T/len(df_pred)

                X_train, X_test, y_train, y_test = econ_ml.train_test_split_data(data=df_pred, target_col=target, test_size=test_size)
                # model_results, baseline_preds, best_model, best_model_name = econ_ml.baseline_ml(target, X_train, X_test, y_train, y_test, baseline_models)
                y_pred = top_model.predict(X_test)
                y_pred = pd.DataFrame(y_pred, columns=[target+"_Prediction"],index=y_test.index)
                y_pred.index, y_test.index = pd.to_datetime(y_test.index), pd.to_datetime(y_test.index)
                model_pred = pd.concat([y_test, y_pred], axis=1)
                model_pred.columns = [target+"_"+str(n),target+"_Prediction"+"_"+str(n)]
                latest_actual = df[[target]].cumsum().loc[df.index[-1]].values[0]
                model_pred = model_pred.cumsum()+latest_actual
                # df_prediction[target+"_"+str(n)] = model_pred[target+"_"+str(n)]
                # df_prediction[target+"_Prediction_"+str(n)] = model_pred[target+"_Prediction_"+str(n)]
                # df_prediction = pd.concat([df[target].cumsum(),model_pred.cumsum()+latest_actual], axis=1)
                list_prediction.append(model_pred)
            df_all_predictions = pd.concat(list_prediction,axis=1)
            df_all_predictions.to_csv(predict_path+f"\prediction_{target}_{Ymd_str}.csv", index=True, index_label="Date")
            df_all_predictions.plot(figsize=(20,8), legend=False, grid=True, title="Prediction "+target)
            # df_all_predictions
            df_future_preds = (df_all_predictions.loc[:,df_all_predictions.columns.str.contains("Prediction")]-latest_actual).dropna()
            # for n in np.linspace(0,1,5):
            #     print(round(n,2))
            #     df_future_preds.quantile(round(n,2),axis=1).plot(figsize=(20,8), legend=False, grid=True)

            return model_pred, y_pred, y_test, top_model, df_all_predictions, df_future_preds # full_test