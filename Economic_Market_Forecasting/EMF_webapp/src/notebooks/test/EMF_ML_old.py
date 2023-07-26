from utils.libraries import *
from utils.objects import *
from utils.functions import *
from definitions import *
## Collecting & Preparing the data
### Setting up the configuration
# Se utiliza un rango de 20 años para la extraccion de datos económicos
def EMF_ML(years, period="W", target="SP500", cutoff_date="2023-01-01"):
    timeframe = 365*years
    today = datetime.today()
    end = today.strftime("%Y-%m-%d")
    start = (today - dt.timedelta(days=timeframe)).strftime("%Y-%m-%d")
    periods = period

    medidas = ["mean", "median", "mode", "Min", "Percentil_25", "Percentil_75","Max", "var", "std", "skew", "kurt"]

    root_path = Path(ROOT_PATH)
    config_paths = [
        "config\Market_Data_Config.csv",
        "config\Economic_Data_Config.csv",
        "config\Calc_Data_Config.csv"
    ]
    market_config = root_path.joinpath(config_paths[0])
    economic_config = root_path.joinpath(config_paths[1]).abspath()
    calc_config = root_path.joinpath(config_paths[2]).abspath()

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
    market_dict
    # Load the economic data config
    print("> Load the economic data config")
    econ = eda.read_config(economic_config)
    fred_series_dict = eda.convert_to_dictionary(markets_used=None)
    fred_series_dict = fred_series_dict["Indicador"]
    calc = eda.read_config(calc_config)
    series_calc_dict = eda.convert_to_dictionary(markets_used=None)
    series_calc_dict = series_calc_dict["Indicador"]
    econ
    # Setting up the indicator dictionaries
    print("> Setting up the indicator dictionaries")
    indicators = {}
    for ind in list(econ["Tipo"].unique()):
        indicators[ind] = econ[econ["Tipo"] == ind]["Indicador"].to_list()
    if extract == True:
        # Extracting the indicator data
        print("> Extracting the indicator data")
        indicators_df = eda.indicator_extraction(fred_series_dict, series_calc_dict, root_path=root_path)
        # Extracting the market data
        print("> Extracting the market data")
        stocks = list(market_dict.keys())
        market_df = eda.market_extraction(stocks, start, end, root_path=root_path)
    else:
        print("No data extraction, reading data from data file")
        path = root_path.joinpath('data', 'raw', 'indicators_df.csv')
        indicators_df = pd.read_csv(path)
        path = root_path.joinpath('data', 'raw', 'market_df.csv')
        market_df = pd.read_csv(path)
    ### Cleaning the data
    # Cleaning the indicator data
    print("> Cleaning the indicator data")
    df_indicators, df_indicators_cum, df_indicators_diff, df_indicators_rets, df_indicators_limpio = eda.limpiar_indicators(
        df_indicators=indicators_df, 
        indicator_dict=indicators, 
        resample=periods, 
        fill_method="ffill", 
        start=start, 
        end=end, 
        root_path=root_path)
    # Cleaning the market data
    print("> Cleaning market data")
    df_market, df_markets_rets, df_markets_cum, df_markets_diff  = eda.limpiar_markets(
        markets_dict=market_dict,
        df_markets=market_df,
        resample=periods, 
        fill_method="ffill", 
        start=start, 
        end=end, 
        root_path=root_path)

    ### Merge indicator and market data
    list_market_dfs = [df_market,df_markets_rets,df_markets_cum,df_markets_diff]
    list_indicators_dfs = [df_indicators_limpio,df_indicators_rets,df_indicators_cum,df_indicators_diff]

    df_all_data, df_all_data_rets, df_all_data_cum, df_all_data_diff = eda.merge_data(list_market_dfs, list_indicators_dfs)
    df = eda.remove_outliers(df_all_data_rets)
    df
    # df = pd.concat([df,df_MKT_GDP], axis=1).dropna()
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
        
        
    df.shape
    # df_add_feat = df_all_data.loc[:,df_all_data.columns.str.contains("_")]
    # for feat in df_add_feat.columns:
    #     df_all_data_rets[feat] = df_add_feat[feat].pct_change().fillna(method="ffill").fillna(method="bfill")
    # df_all_data, df_MKT_GDP = eda.mkt_gdp(df_all_data, "MKT_GDP", n_stds=2, deg=2)
    # df_MKT_GDP.plot()
    # df_MKT_GDP.plot()
    ### Creating lags in the data
    list_data_dfs = [df_all_data,df_all_data_rets,df_all_data_cum,df_all_data_diff]

    df_all_lag_data, df_all_lag_data_rets, df_all_lag_data_cum, df_all_lag_data_diff = eda.lag_data(list_data_dfs, target, n_lags=12)
    ## Data Preprocessing & Feature Engineering
    econ_ml = Preprocessor()
    ### Splitting the data
    X_train, X_test, y_train, y_test = econ_ml.train_test_split_data(data=df_all_lag_data_rets, target_col=target, test_size=0.15)
    #### Creating the baseline
    baseline_models = econ_ml.define_baseline_models()
    model_results = pd.DataFrame()
    model_results_dict = {}
    for name, model in baseline_models.items():
        if name == "PolynomialFeatures":
            pass
        else:
            model.fit(X_train, y_train)
            score = model.score(X_train, y_train)
            print(name+" "+str(round(score,5)))
            model_results_dict[name] = score
    model_results = model_results.append(model_results_dict, ignore_index=True).T.sort_values(by=0, ascending=False)
    baseline_models[model_results.index[0]]
    ### Indentifying the most important features
    print("> Performing feature importance analysis")
    df_top_data, feature_importance, top_feature_importance, score = econ_ml.feature_importance(target=target, 
                                                                                                    df_data=df.loc[:cutoff_date],
                                                                                                    model=baseline_models[model_results.index[0]],
                                                                                                    accepted_importance=0.95)
    X_train, X_test, y_train, y_test = econ_ml.train_test_split_data(data=df_top_data, target_col=target, test_size=0.15)
    baseline_models = econ_ml.define_baseline_models()
    model_results = pd.DataFrame()
    model_results_dict = {}
    for name, model in baseline_models.items():
        if name == "PolynomialFeatures":
            pass
        else:
            model.fit(X_train, y_train)
            score = model.score(X_train, y_train)
            print(name+" "+str(round(score,5)))
            model_results_dict[name] = score
    model_results = model_results.append(model_results_dict, ignore_index=True).T.sort_values(by=0, ascending=False)
    ### Scaling the data
    X_train_scale, X_test_scale = econ_ml.scaler(X_train=X_train, X_test=X_test)
    ## Performing Machine Learning
    def lstm_model(epochs,batch_size,validation_split):
        # LSTM model
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=64, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], 1)))
        lstm_model.add(LSTM(32))
        lstm_model.add(Dense(units=1))
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.fit(X_train_scale, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return lstm_model

    def gru_model(epochs,batch_size,validation_split):
        # gru model
        gru_model = Sequential()
        gru_model.add(GRU(units=64, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], 1)))
        gru_model.add(GRU(32))
        gru_model.add(Dense(units=1))
        gru_model.compile(optimizer='adam', loss='mse')
        gru_model.fit(X_train_scale, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return gru_model

    def cnn_model(epochs,batch_size,validation_split):
        # cnn model
        cnn_model = Sequential()
        cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(units=1))
        cnn_model.compile(optimizer='adam', loss='mse')
        cnn_model.fit(X_train_scale, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return cnn_model

    models = {
        "LSTM":lstm_model(epochs=15,batch_size=16,validation_split=0.2),
        "GRU":gru_model(epochs=15,batch_size=16,validation_split=0.2),
        "CNN":cnn_model(epochs=15,batch_size=16,validation_split=0.2)
    }
    # for name, model in models.items():
    #     print(f"Evaluate the model: {name}: {model.evaluate(X_test_scale, y_test)}")
    #     # model.evaluate(X_test_scale, y_test)
    # for name, model in models.items():
    #     print(f"Review of model: {name}")
    #     # if name != "CNN":
    #     try:
    #         df_history = pd.DataFrame({"Loss":model.history.history['loss'], "Val_Loss":model.history.history['val_loss']})
    #         df_history.plot(figsize=(10, 6), title=name)
    #     except:
    #         pass
    #     pred = model.predict(X_test_scale)
    #     predictions = pd.DataFrame(pred, index=y_test.index)
    #     predictions.index = pd.to_datetime(predictions.index)
    #     model.save(f'models\models\{name}_best_model_no_rets.h5')
    #     model.save_weights(f'models\weights\{name}_best_model_no_rets.h5') 
    #     # print(model.summary())
    #     plt.figure(figsize=(10, 6))
    #     plt.title(f"{name} -> {X_test.index[0]} - {X_test.index[-1]}")
    #     # plt.plot(df_all_lag_data["SP500"].pct_change())
    #     plt.plot(pd.to_datetime(y_test.index), y_test.cumsum(), label='Actual')
    #     plt.plot((predictions.cumsum()), label="Pred")
    #     plt.legend()

    model_eval = {}
    for name, model in models.items():
        loss_score = model.evaluate(X_test_scale, y_test)
        print(f"Evaluate the model: {name}: {loss_score}")
        model_eval[name] = loss_score
        # model.evaluate(X_test_scale, y_test)
    model_eval_df = pd.DataFrame(model_eval.values(), index=model_eval.keys()).sort_values(by=0, ascending=True)
    best_model = model_eval_df.iloc[0,:]
    for name, model in models.items():
        print(f"Review of model: {name}")
        # if name != "CNN":
        try:
            df_history = pd.DataFrame({"Loss":model.history.history['loss'], "Val_Loss":model.history.history['val_loss']})
            df_history.plot(figsize=(10, 6), title=name)
        except:
            pass
        pred = model.predict(X_test_scale)
        predictions = pd.DataFrame(pred, index=y_test.index)
        predictions.index = pd.to_datetime(predictions.index)
        predictions.columns = [target+"_Pred"]

        model.save(f'models\models\{name}_best_model.h5')
        model.save_weights(f'models\weights\{name}_best_model.h5') 

        y_test.index = pd.to_datetime(y_test.index)
        EMF_forecast = pd.concat([y_test, predictions], axis=1).cumsum()
        # print(model.summary())
        plt.figure(figsize=(10, 6))
        plt.title(f"{name} -> {X_test.index[0]} - {X_test.index[-1]}")
        plt.plot(EMF_forecast, label=EMF_forecast.columns)
        plt.legend()
        if name == best_model.name:
            EMF_forecast.to_csv(r"C:\Users\Joan Oliver\Documents\GitHub\Economic_Market_Forecasting\Economic_Market_Forecasting\EMF_webapp\EMF_project"+f"\EMF_forecast_{target}.csv")