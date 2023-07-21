from utils.libraries import *
    
class EconomicDataAnalyzer:
    def __init__(self, config=None):
        self.config = config
        
    def read_config(self, file_path):
        self.config = pd.read_csv(file_path, sep=';', decimal=',', header=0)
        return self.config

    def convert_to_dictionary(self, markets_used):
        self.config = self.config.dropna()
        if markets_used is None:
            pass
        else:
            self.markets_used = markets_used
            self.config = self.config[self.config["Codigo"].isin(markets_used)]
        self.config.set_index('Codigo', inplace=True)
        if self.config is None:
            raise ValueError("No config loaded.")
        if self.config.columns.size >2:
            self.config = self.config.iloc[:, :2]
        return self.config.to_dict()
    
    def indicator_extraction(self, fred_dict, calc_dict, root_path):
        """
        Extrae datos de indicadores económicos de FRED y realiza cálculos según el indicador.

        Args:
        - fred_dict (dict): Un diccionario que contiene el código y el nombre de cada indicador económico a extraer.
        - calc_dict (dict): Un diccionario que contiene el código y el nombre de cada columna calculada para agregar al DataFrame.

        Returns:
        - indicators_df (pd.DataFrame): Un DataFrame que contiene los datos de los indicadores económicos extraídos y las columnas calculadas.
        """
        fred = Fred(api_key=fred_key)

        indicators_df = pd.DataFrame() # DataFrame para almacenar los datos extraidos
        # Acceder al diccionario de indicadores economicos para extraer sus datos (utilizando el objeto 'fred' y la funcion 'get_series') y almacenarlos en un DataFrame
        for code,indicator in fred_dict.items():
            print(f'   -->Extracting {indicator}...')
            indicators_df[indicator] = fred.get_series(code)
            # Printamos el resultado de la extraccion de cada indicador con su codigo, indicador, rows y fecha de inicio
            # Realizar los siguientes calculos dependiendo del indicador y añadirlo al DataFrame
            if code == 'CES0500000003':
                indicators_df[calc_dict[code]] = (indicators_df[indicator] - indicators_df[indicator].shift(12))/indicators_df[indicator]*100
            # elif code == 'CPIAUCSL':
            #     indicators_df[calc_dict[code]] = indicators_df[indicator].pct_change()
            # elif code == 'PPIFIS':
            #     indicators_df[calc_dict[code]] = indicators_df[indicator].pct_change()
            elif code == 'GFDEBTN':
                indicators_df[calc_dict[code]] = (indicators_df[indicator] * .1) / indicators_df['GDP']
            elif code == 'BOPTIMP':
                indicators_df[calc_dict[code]] = indicators_df[indicator] - indicators_df['Exports']
            elif code == 'IEABC':
                indicators_df[calc_dict[code]] = indicators_df[indicator] / indicators_df["GDP"]
        # Crear nuevas columnas para los 'spread' de los tipos de interes (tipos a largo plazo - tipos a cortos plazo)
        # 3m5y,3m10y, 2y10y, 2y20y, 5y10y, 10y30y, 10yTrea30yFRM
        indicators_df["3m5y"] =  indicators_df["5-Year Treasury Yield"] - indicators_df["3-Month Treasury Yield"]
        indicators_df["3m10y"] = indicators_df["10-Year Treasury Yield"] - indicators_df["3-Month Treasury Yield"]
        indicators_df["2y10y"] = indicators_df["10-Year Treasury Yield"] - indicators_df["2-Year Treasury Yield"]
        indicators_df["2y20y"] = indicators_df["20-Year Treasury Yield"] - indicators_df["2-Year Treasury Yield"]
        indicators_df["5y10y"] = indicators_df["10-Year Treasury Yield"] - indicators_df["5-Year Treasury Yield"]
        indicators_df["10y30y"] = indicators_df["30-Year Treasury Yield"] - indicators_df["10-Year Treasury Yield"]
        indicators_df["10yTrea30yFRM"] = indicators_df["30-Year Mortgage Rate"] - indicators_df["10-Year Treasury Yield"]
        # Guardar el DataFrame como un archivo csv
        path = root_path.joinpath('data', 'raw', 'indicators_df.csv')
        indicators_df.to_csv(path)
        print(f'Indicators Extracted: {indicators_df.columns}')
        return indicators_df
    
    def market_extraction(self, stocks, start, end, root_path):
        """
        Extrae datos de mercado históricos para una lista determinada de acciones entre una fecha de inicio y finalización especificada.

        Args:
        - stocks (lst): Lista de acciones para extraer datos.
        - start (str): fecha de inicio en formato yyyy-mm-dd.
        - end (str): fecha de finalización en formato yyyy-mm-dd.

        Returns:
        - market_hist (pd.DataFrame): Pandas DataFrame que contiene datos de mercado históricos para las acciones y el rango de fechas especificados.
        """
        # Permite crear el DataFrame
        yfin.pdr_override()
        # Extraer los precios de !Yahoo Finanzas para cada uno de los indices y almacenarlos en el DataFrame 'markets'
        markets = pdr.get_data_yahoo(stocks,start=start,end=end)
        # Filtrar el DataFrame quedandonos con la columna de 'Adj Close' y el rango temporal previamente definido
        market_hist = markets["Adj Close"].loc[start:end]# Guardar el DataFrame como un archivo csv
        path = root_path.joinpath('data', 'raw', 'market_df.csv')
        market_hist.to_csv(path)
        return market_hist

    def limpiar_markets(self,markets_dict, df_markets, resample, fill_method, start, end, root_path):
        """
        Limpia los datos de mercado de los archivos de datos sin procesar y guarda los archivos procesados en el
        carpeta procesada. Filtra datos de mercado para el intervalo de tiempo especificado y
        crea marcos de datos de rendimiento diarios y acumulativos.

        Args:
        - start (str): fecha de inicio en formato YYYY-MM-DD
        - end (str): fecha de finalización en formato YYYY-MM-DD

        Returns:
        - df_markets (pd.DataFrame): DataFrame de datos de mercado filtrados
        - df_market_rets (pd.DataFrame): DataFrame de rentabilidades diarias del mercado
        - df_market_cum (pd.DataFrame): DataFrame de rentabilidades acumuladas del mercado
        - df_market_diff (pd.DataFrame): DataFrame de diferenciales del mercado
        """
        # Extraer datos de la carpeta 'raw'
        # df_markets = pd.read_csv(r'C:\Users\Joan Oliver\BullGlobe\Investing Scripts\Economic_Market_Forecasting\src\data\raw\market_df.csv',index_col=0, header=0)
        df_markets = df_markets.rename(columns=markets_dict)
        # Filtrar los datos de mercado de los primeros 23 años
        df_markets = df_markets.loc[start:end]
        df_markets.index = pd.to_datetime(df_markets.index, utc=True, format='%Y-%m-%d')
        # df_markets.index = df_markets.index.strftime('%Y-%m-%d')

        # Hacer el resampleo de datos
        df_markets = df_markets.resample(resample).fillna(method=fill_method)
        if fill_method == 'ffill':
            # Rellenar los siguientes datos vacios con el ultimo dato
            df_markets.fillna(method='bfill', inplace=True) 

        # Crear DataFrame de rendimiento diario de mercados
        df_market_rets = df_markets.pct_change().fillna(0)
        df_market_rets.index = pd.to_datetime(df_market_rets.index, utc=True, format='%Y-%m-%d')
        df_market_rets.index = df_market_rets.index.strftime('%Y-%m-%d')
        # Crear DataFrame de rendimiento acumulado de mercados
        df_market_cum = df_market_rets.cumsum().fillna(0)

        # Crear DataFrame de diferencial de mercados
        df_market_diff = df_markets.diff().fillna(0)

        # Guardar tablas procesadas de mercados
        
        path = root_path.joinpath('data', 'processed', 'markets')
        df_markets.to_csv(path.joinpath('market_hist.csv'))
        df_market_rets.to_csv(path.joinpath('market_rets.csv'))
        df_market_cum.to_csv(path.joinpath('market_cum.csv'))
        df_market_diff.to_csv(path.joinpath('market_diff.csv'))

        return df_markets, df_market_rets, df_market_cum, df_market_diff

    def limpiar_indicators(self,df_indicators,indicator_dict, resample, fill_method, start, end, root_path):
        """
        Función que limpia y procesa los datos del DataFrame de indicadores económicos.

        Args:
        - df_indicators (DataFrame): DataFrame de los indicadores económicos.
        - indicator_dict (dict): Diccionario con los nombres y columnas de los indicadores.
        - resample (str): Periodo de tiempo a resamplear.
        - fill_method (str): Metodo de relleno.
        - start (str): Fecha de inicio del periodo a limpiar.
        - end (str): Fecha de fin del periodo a limpiar.

        Returns:
        - df_indicators (DataFrame): DataFrame de los indicadores económicos sin procesar.
        - df_indicators_limpio (DataFrame): DataFrame de los indicadores económicos procesado y limpio.
        - df_indicators_diff (DataFrame): DataFrame de los indicadores económicos procesado y diferenciado.
        - df_indicators_rets (DataFrame): DataFrame del rendimiento de los indicadores económicos.
        """
        # Limpiar datos del DataFrame de indicadores economicos
        df_indicators = df_indicators.loc[start:end]
        df_indicators.index = pd.to_datetime(df_indicators.index, utc=True, format='%Y-%m-%d')
        df_indicators = df_indicators.resample(resample).last()
        # Rellenar los datos vacios con el dato anterior
        df_indicators_limpio = df_indicators.fillna(method=fill_method)
        if fill_method == 'ffill':
            # Rellenar los siguientes datos vacios con el ultimo dato
            df_indicators_limpio.fillna(method='bfill', inplace=True) 
        # Guardar las tablas de indicadores en csv
        for ind_name, ind_list in indicator_dict.items():
            path = root_path.joinpath('data', 'processed', 'indicators', ind_name+'.csv')
            df_indicators[ind_list].dropna().to_csv(path)
        # Generar el diferencial de los datos
        df_indicators_diff = df_indicators_limpio.diff().fillna(0)
        # Generar el dataframe de rendimiento de los datos
        df_indicators_rets = df_indicators_limpio.pct_change().fillna(0)
        dfs = [df_indicators, df_indicators_limpio, df_indicators_diff, df_indicators_rets]
        for df in dfs:
            try:
                df.index = df.index.strftime("%Y-%m-%d")
            except:
                print("Error in date formatting")
        df_indicators_cum = df_indicators_rets.cumsum().fillna(0)
        return df_indicators, df_indicators_cum, df_indicators_diff, df_indicators_rets, df_indicators_limpio

    def merge_data(self, list_indicators_dfs, list_market_dfs):
        """
        Merge data from two lists of dataframes containing indicators and market data.

        Args:
            list_indicators_dfs (list): A list of dataframes containing indicators data.
            list_market_dfs (list): A list of dataframes containing market data.

        Returns:
            tuple: A tuple of dataframes merged from the input data. The tuple contains four
                dataframes corresponding to the input lists.
        """
        list_all_dfs = []
        for df_indicators, df_market in zip(list_indicators_dfs, list_market_dfs):
            if type(df_indicators.index[0]) != str:
                try:
                    df_indicators.index = df_indicators.index.strftime("%Y-%m-%d")
                except:
                    print("Don't change date format")
            if type(df_market.index[0]) != str:
                try:
                    df_market.index = df_market.index.strftime("%Y-%m-%d").str.split(" ").str[0]
                except:
                    print("Don't change date format")
            df = pd.merge(df_indicators,df_market, left_index=True, right_index=True,how='outer').fillna(method='ffill')
            list_all_dfs.append(df)
        return list_all_dfs[0], list_all_dfs[1], list_all_dfs[2], list_all_dfs[3]

    def lag_data(self, list_data_dfs, target, n_lags):
        """
        Lag the given list of dataframes by shifting their columns to create lagged versions of their features.
        :param list_data_dfs: A list of pandas dataframes to be lagged.
        :param n_lags: An integer specifying the number of lags to be created.
        :return: A tuple of pandas dataframes where each dataframe is a copy of the input dataframe with its features lagged.
        """
        list_dfs = []
        n_lags = 12

        for df in list_data_dfs:
            df = df.copy()
            lagged_df = pd.DataFrame()
            for column in df.columns:
                if target in column:
                    pass
                else:
                    for i in range(1, n_lags + 1):
                        lagged_df[f'{column} (t-{i})'] = df[column].shift(i)
            # Combine the original DataFrame with the lagged DataFrame
            df = pd.concat([df, lagged_df], axis=1).dropna()
            list_dfs.append(df)
        return list_dfs[0], list_dfs[1], list_dfs[2], list_dfs[3]

    def merge_clean_all_data(self,df_indicators_rets,df_markets_rets,std_threshold,root_path, periods, historical=False):
        """
        Merge two dataframes, df_indicators_rets and df_markets_rets, by their index. 
        Remove any outliers that exceed a given standard deviation threshold. 
        Forward fill any remaining NaN values. 
        Calculate the cumulative sum of the resulting dataframe. 

        :param df_indicators_rets: A pandas dataframe containing indicator returns.
        :type df_indicators_rets: pandas.core.frame.DataFrame
        :param df_markets_rets: A pandas dataframe containing market returns.
        :type df_markets_rets: pandas.core.frame.DataFrame
        :param std_threshold: A float representing the standard deviation threshold for removing outliers.
        :type std_threshold: float

        :return: A tuple containing two pandas dataframes. The first is the cleaned and merged dataframe, and the second is the cumulative sum of the cleaned and merged dataframe.
        :rtype: tuple(pandas.core.frame.DataFrame, pandas.core.frame.DataFrame)
        """        
        if type(df_indicators_rets.index[0]) != str:
            try:
                df_indicators_rets.index = df_indicators_rets.index.strftime("%Y-%m-%d")
            except:
                pass
        if type(df_markets_rets.index[0]) != str:
            try:
                df_markets_rets.index = df_markets_rets.index.strftime("%Y-%m-%d")
            except:
                pass
        df_all_data = pd.merge(df_indicators_rets,df_markets_rets, left_index=True, right_index=True,how='outer').fillna(method='ffill')
        z_scores = stats.zscore(df_all_data)
        threshold = std_threshold
        df_no_outliers = df_all_data[abs(z_scores) < threshold]
        df_no_outliers.fillna(method='ffill', inplace=True)
        df_all_data = df_no_outliers.dropna(axis=1)
        path = root_path.joinpath('data', 'processed', 'indicators', 'model_data_'+periods+'.csv')
        df_all_data.to_csv(path)
        if historical == True:
            df_all_data_hist = df_all_data.cumsum()
            path = root_path.joinpath('data', 'processed', 'indicators', 'model_data_hist'+periods+'.csv')
            df_all_data_hist.to_csv(path)
        else:
            df_all_data_hist = None
            df_all_data_diff = df_all_data.diff().fillna(0)
        return df_all_data, df_all_data_hist, df_all_data_diff

    def remove_outliers(self, df, threshold_mad=6):    
        median = np.median(df)
        mad = np.median(np.abs(df - median))
        threshold_mad = threshold_mad
        modified_z_scores = 0.6745 * (df - median) / mad
        outliers_mad = df[np.abs(modified_z_scores) > threshold_mad]
        df_no_outliers = df[np.abs(modified_z_scores) <= threshold_mad]

        for column in df_no_outliers.columns:
            null_indexes = df_no_outliers[column].isnull()
            null_indexes_shifted = null_indexes.shift(1, fill_value=False)
            null_indexes_shifted_rev = null_indexes.shift(-1, fill_value=False)
            mask = null_indexes | null_indexes_shifted | null_indexes_shifted_rev

            while mask.any():  # Repeat until there are no more NaN values surrounded by non-NaN values
                avg_values = df_no_outliers[column].rolling(3, min_periods=1, center=True).mean()  # Calculate rolling average of size 3
                df_no_outliers[column] = np.where(mask, avg_values, df_no_outliers[column])  # Replace NaN values with rolling average values

                null_indexes = df_no_outliers[column].isnull()
                null_indexes_shifted = null_indexes.shift(1, fill_value=False)
                null_indexes_shifted_rev = null_indexes.shift(-1, fill_value=False)
                mask = null_indexes | null_indexes_shifted | null_indexes_shifted_rev
        return df_no_outliers
    def analisis_univariante(self, indicators, medidas):
        """
        This function calculates several univariate statistical measures for each indicator in the given pandas dataframes.

        Args:
        - indicators: A dictionary of pandas dataframes. Each dataframe represents an indicator.
        - medidas: A list of strings with the names of the statistical measures to be calculated.

        Returns:
        - A transposed pandas dataframe with the indicators as columns and the statistical measures as rows.
        """
        ind_dict = {}
        df_measures = pd.DataFrame(index=medidas)
        for inds, df in indicators.items():
            if inds == "market_cum" or inds == "market_hist":
                pass # Pasar para esos dos indicadores 
            else:
                if inds == "market_rets":
                    df = df*100
                else:
                    df = df.diff().fillna(0)
                for n, indicator in enumerate(df.columns):
                    # Definir las medidas de Posicion
                    mean = df.mean()
                    mode = df[indicator].mode()
                    median = df[indicator].median()

                    # Definir los cuartile
                    Percentil_0 = df[indicator].quantile(0)
                    Percentil_25 = df[indicator].quantile(0.25)
                    Percentil_75 = df[indicator].quantile(0.75)
                    Percentil_100 = df[indicator].quantile(1)

                    # Definir las medidas de Variabilidad
                    var = df[indicator].var()
                    std = df[indicator].std()

                    # Definir las medidas de Forma
                    skew = df[indicator].skew()
                    kurt = df[indicator].kurt()

                    data=[mean, median, mode[0], Percentil_0, Percentil_25, Percentil_75, Percentil_100, var, std, skew, kurt]

                    df_measures[indicator] = pd.Series(data=data,index = medidas,name=indicator)
                                        
                    medidas_tup = zip(medidas, data)

                    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 3))
                    # plt.subplots_adjust(hspace=0.5)
                    # fig.suptitle(indicator, fontsize=12, y=0.9)
                    
                    # sns.histplot(df[indicator], kde=True, ax=axs[0])
                    # axs[0].axvline(mean, color="red", linestyle="--", label="Mean")
                    # axs[0].axvline(median, color="green", linestyle="--", label="Median")
                    # axs[0].axvline(mode[0], color="blue", linestyle="--", label="Mode")
                    # axs[0].axvline(Percentil_25, color="yellow", linestyle="--", label="25th Percentile")
                    # axs[0].axvline(Percentil_75, color="yellow", linestyle="--", label="75th Percentile")
                    # axs[0].legend(loc='upper right', fontsize=10)
                    # axs[1].boxplot(df[indicator])
                    # plt.tight_layout();
                    # plt.close(fig)

                    # # Create an HTML table
                    # table = HTML(df_measures[[indicator]].round(4).to_html(index=True))
                    # # Display the chart and table side by side
                    # display(table, fig)
                    ind_dict[indicator] = df[indicator]
        return df_measures.T, ind_dict
    
    def plot_univariante(self, indicator, indictator_dict, medidas):
        df_measures = pd.DataFrame(index=medidas)
        df = indictator_dict
        # Definir las medidas de Posicion
        mean = df[indicator].mean()
        mode = df[indicator].mode()
        median = df[indicator].median()
        # Definir los cuartile
        Percentil_0 = df[indicator].quantile(0)
        Percentil_25 = df[indicator].quantile(0.25)
        Percentil_75 = df[indicator].quantile(0.75)
        Percentil_100 = df[indicator].quantile(1)
        # Definir las medidas de Variabilidad
        var = df[indicator].var()
        std = df[indicator].std()
        # Definir las medidas de Forma
        skew = df[indicator].skew()
        kurt = df[indicator].kurt()

        data=[mean, median, mode[0], Percentil_0, Percentil_25, Percentil_75, Percentil_100, var, std, skew, kurt]

        df_measures[indicator] = pd.Series(data=data,index = medidas,name=indicator)
                            
        medidas_tup = zip(medidas, data)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 3))
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle(indicator, fontsize=12, y=0.9)
        
        sns.histplot(df[indicator], kde=True, ax=axs[0])
        axs[0].axvline(mean, color="red", linestyle="--", label="Mean")
        axs[0].axvline(median, color="green", linestyle="--", label="Median")
        axs[0].axvline(mode[0], color="blue", linestyle="--", label="Mode")
        axs[0].axvline(Percentil_25, color="yellow", linestyle="--", label="25th Percentile")
        axs[0].axvline(Percentil_75, color="yellow", linestyle="--", label="75th Percentile")
        axs[0].legend(loc='upper right', fontsize=10)

        axs[1].boxplot(df[indicator])

        # plt.figure(figsize=(15, 3))
        # plt.plot(df[indicator].cumsum().fillna(0))
        plt.tight_layout();
        plt.close(fig)

        # Create an HTML table
        table = HTML(df_measures[[indicator]].round(4).to_html(index=True))
        # Display the chart and table side by side
        display(table, fig)

    
    def analisis_bivariante(self, markets, indicators_names, stock_start, stock_end, df_market_hist, df_market_cum, inds_dict):
        """
        Genera gráficos y tablas de análisis bivariados para datos del mercado de valores vs indicadores económicos.

        Args:
        - markets (lista): Lista de nombres de mercado (str) para analizar.
        - indicators_names (lista): Lista de nombres de indicadores (str) a analizar.
        - stock_start (str): Fecha de inicio (AAAA-MM-DD) para el análisis.
        - stock_end (str): Fecha de finalización (AAAA-MM-DD) para el análisis.
        - df_market_rets (pandas.DataFrame): DataFrame con datos de rentabilidad del mercado.
        - df_market_cum (pandas.DataFrame): DataFrame con datos de rentabilidad acumulada del mercado.
        - df_inds (pandas.DataFrame): DataFrame con datos del indicador.

        Returns:
        - df_ind_mkt (pandas.DataFrame): DataFrame con datos de mercado e indicadores combinados.
        - df_ind_mkt_values ​​(pandas.DataFrame): DataFrame con rendimientos acumulados del mercado fusionado y datos de indicadores.
        """
        for ind, df_ind in inds_dict.items():
            df_ind_chg = df_ind.pct_change().fillna(0)
            df_ind_mkt = pd.merge(df_market_hist[markets], df_ind_chg[indicators_names], left_index=True, right_index=True)
            df_ind_mkt[markets] = df_ind_mkt[markets].pct_change().fillna(0)
            # if ind == "interest_rate_spread" or ind == "interest_rates":
            #     df_ind_mkt_values = pd.merge(df_market_cum[markets], df_ind[indicators_names].loc[stock_start:stock_end].fillna(0), left_index=True, right_index=True)
            # else:
            #     df_ind_mkt_values = pd.merge(df_market_cum[markets], df_ind[indicators_names].loc[stock_start:stock_end].pct_change().cumsum().fillna(0), left_index=True, right_index=True)
            df_ind_mkt_values = df_ind_mkt.cumsum().fillna(0)
            df_corr = df_ind_mkt.corr()

        return df_ind_mkt, df_ind_mkt_values, df_corr
    

    def plot_bivariant(self, markets, indicators_names, df_ind_mkt, df_ind_mkt_values, df_corr):
        """
        Plots bivariate relationships between the given markets and indicators.

        :param markets: A list of markets to plot.
        :type markets: list
        :param indicators_names: A list of indicator names to plot.
        :type indicators_names: list
        :param df_ind_mkt: A dataframe containing the data to plot.
        :type df_ind_mkt: pandas.DataFrame
        :param df_ind_mkt_values: A dataframe containing the values to plot.
        :type df_ind_mkt_values: pandas.DataFrame
        :param df_corr: A dataframe containing the correlation values.
        :type df_corr: pandas.DataFrame
        :return: None.
        """
        for market in markets:
            for indicator in indicators_names:
                fig = plt.figure(figsize=(10,10))
                fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 8), )
                plt.subplots_adjust(hspace=0.5, top=1, wspace=0.5)
                fig.suptitle(market+" vs "+indicator, fontsize=12, y=1)
                sns.histplot(data=[df_ind_mkt[market],df_ind_mkt[indicator]], color="#4CB391", fill=True, alpha=0.5, kde=True, ax=axs[0,0])
                sns.boxplot(data=df_ind_mkt[[market,indicator]],ax=axs[1,0])
                sns.regplot(x=df_ind_mkt[market], y=df_ind_mkt[indicator],ax=axs[1,1])
                df_ind_mkt_values[[market,indicator]].plot(ax=axs[0,1])
                plt.tight_layout();
                plt.close(fig)

                df_corr = df_ind_mkt.corr()
                # Create an HTML table
                table = HTML(pd.DataFrame(df_corr[[market]].loc[indicator]).round(4).to_html(index=True))
                # Display the chart and table side by side
                display(table, fig)

    def indicators_dict(ind_mkt_types, market_analysis):
        """
        Creates a dictionary of indicators and market data.
        
        :param ind_mkt_types: A list of strings representing the indicator/market names to include in the dictionary.
        :type ind_mkt_types: list
        :param market_analysis: A list of strings representing the market analysis names to include in the dictionary.
        :type market_analysis: list
        
        :return: Two dictionaries - 'indicators_dict' containing all the indicators and 'data_dict' containing all the market data.
        :rtype: tuple
        """    
        # Save all indicators and market data in a dictionary
        indicators_dict = {}
        data_dict = {}
        for ind in ind_mkt_types:
            if "market" in ind:
                indicators_dict[ind] = pd.read_csv(r'C:\Users\Joan Oliver\BullGlobe\Investing Scripts\Economic_Market_Forecasting\src\data\processed\markets\{0}.csv'.format(ind),index_col=0, header=0)
                for mkt in market_analysis:
                    data_dict[mkt] = indicators_dict[ind][[mkt]]
            else:
                indicators_dict[ind] = pd.read_csv(r'C:\Users\Joan Oliver\BullGlobe\Investing Scripts\Economic_Market_Forecasting\src\data\processed\indicators\{0}.csv'.format(ind),index_col=0, header=0)
                data_dict[ind] = indicators_dict[ind]
        return indicators_dict, data_dict
   
    def trend_line(self, df, name, n_stds=2):
        coef = np.polyfit(range(0,len(df[name])), df[name], n_stds)
        x_trend = np.linspace(0,len(df[name]),len(df[name]))
        y_trend = np.polyval(coef, x_trend)
        df = pd.DataFrame(y_trend, index=df.index, columns=[name])
        return df 
    
    def mkt_gdp(self, df, name, n_stds, deg):
        def trend_line(self, df, name, deg):
            coef = np.polyfit(range(0,len(df[name])), df[name], deg)
            x_trend = np.linspace(0,len(df[name]),len(df[name]))
            y_trend = np.polyval(coef, x_trend)
            df = pd.DataFrame(y_trend, index=df.index, columns=[name])
            return df 
        df_MKT_GDP = pd.DataFrame()
        df_MKT_GDP[name] = df["SP500"]/(df["GDP"]*.01)
        df_MKT_GDP[name].fillna(method='ffill', inplace=True)
        df_MKT_GDP[f'{name}_trend'] = trend_line(df_MKT_GDP,name,deg)
        std = (df_MKT_GDP[f'{name}_trend']).std()
        n_stds = n_stds
        poles = ['pos',"neg"]
        for n_std in  np.linspace(-n_stds,n_stds,9):
            str_std = str(n_std).replace('.','_').replace('-','')
            if n_std<0:
                pole = poles[1]
            else:
                if n_std>0:
                    pole = poles[0]
                else:
                    pole = ""
            df_intermedio = pd.DataFrame()
            df_intermedio[f'{name}_trend_{str_std}std_{pole}'] = trend_line(df_MKT_GDP,name,n_stds,deg) + n_std*std
            df_MKT_GDP[f"{name}_{str_std}std_{pole}"] = (df_intermedio[f'{name}_trend_{str_std}std_{pole}']>df_MKT_GDP[name]).astype(int)
        df = pd.concat([df, df_MKT_GDP], axis=1)
        return df, df_MKT_GDP




class Preprocessor:
    def __init__(self, data=None):
        """
        Initializes a new instance of the class with a data parameter.

        :param data: The data to be assigned to the instance variable, self.df.
        :type data: Any
        """
        self.df = data

    def feature_importance(self, target, df_data, model, accepted_importance=0.95):
        """
        Calculates the feature importance of the given target variable and dataframe using the Random Forest Regressor model.
    	
    	:param target: The target variable column name in the dataframe.
    	:type target: str
    	
    	:param df_data: The dataframe containing the target variable and the features for which feature importance is to be calculated.
    	:type df_data: pd.DataFrame
    	
    	:param accepted_importance: The minimum cumulative feature importance to be considered while selecting the top features.
    	:type accepted_importance: float
    	
    	:return: A pandas dataframe containing the feature importance of all the features in the given dataframe, and another dataframe containing the top features selected based on their importance.
    	:rtype: tuple(pd.DataFrame, pd.DataFrame)
        """
        self.target = target
        self.df_data = df_data
        self.accepted_importance = accepted_importance
        # Feature importance
        X = self.df_data.drop([self.target], axis=1)
        y = self.df_data[self.target]

        feat_imp_model = model
        feat_imp_model.fit(X,y)
        score = feat_imp_model.score(X,y)

        feature_importance = feat_imp_model.feature_importances_

        df_feature_importance = pd.DataFrame(index=X.columns,data=feature_importance, columns=["Importance"]).sort_values(by="Importance", ascending=False)
        df_feature_importance["Cum_Importance"] = df_feature_importance.cumsum()
        df_top_feature_importance = df_feature_importance[df_feature_importance["Cum_Importance"] < self.accepted_importance]

        df_top_data = df_data.loc[:,df_data.columns.isin(df_top_feature_importance.index)]
        df_top_data = pd.concat([df_top_data, df_data[[target]]], axis=1).dropna()     

        return df_top_data, df_feature_importance, df_top_feature_importance, score

    def train_test_split_data(self, test_size=0.2, data=None, target_col=None):
        """
        Splits the data into training and testing sets using the train_test_split method from scikit-learn.
        
        :param test_size: The proportion of the data to be used for testing. Default is 0.2.
        :param data: The input data to be split.
        :return: Returns four values: X_train, X_test, y_train, and y_test.
            - X_train: The training set of independent variables.
            - X_test: The testing set of independent variables.
            - y_train: The training set of dependent variable.
            - y_test: The testing set of dependent variable.
        """
        self.data = data
        self.X = self.data.drop([target_col], axis=1)
        self.y = self.data[target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, shuffle=False)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scaler(self, X_train=None, X_test=None):
        """
        Scales the data using the MinMaxScaler from sklearn.
        
        :param X_train: The training set of independent variables.
        :param X_test: The testing set of independent variables.
        :param y_train: The training set of dependent variable.
        :param y_test: The testing set of dependent variable.
        :return: Returns four values: X_train, X_test, y_train, and y_test.
            - X_train: The training set of independent variables.
            - X_test: The testing set of independent variables.
            - y_train: The training set of dependent variable.
            - y_test: The testing set of dependent variable.
        """
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train_scaled = scaler.transform(X_train)
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = None
        return X_train_scaled, X_test_scaled

    def lstm_model(X_train_scale,y_train,X_train,epochs=15,batch_size=16,validation_split=0.2):
        # LSTM model
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=64, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], 1)))
        lstm_model.add(LSTM(32))
        lstm_model.add(Dense(units=1))
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.fit(X_train_scale, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return lstm_model

    def gru_model(X_train_scale,y_train,X_train,epochs=15,batch_size=16,validation_split=0.2):
        # gru model
        gru_model = Sequential()
        gru_model.add(GRU(units=64, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], 1)))
        gru_model.add(GRU(32))
        gru_model.add(Dense(units=1))
        gru_model.compile(optimizer='adam', loss='mse')
        gru_model.fit(X_train_scale, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return gru_model

    def cnn_model(X_train_scale,y_train,X_train,epochs=15,batch_size=16,validation_split=0.2):
        # cnn model
        cnn_model = Sequential()
        cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(units=1))
        cnn_model.compile(optimizer='adam', loss='mse')
        cnn_model.fit(X_train_scale, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return cnn_model

    

    def define_baseline_models(self, baseline_models_dict=None,*args, **kwargs):
        """
        Splits the data into training and test sets.
        
        :param test_size: The proportion of the data to include in the test set.
        :type test_size: float
        
        :return: Two dataframes - 'train' and 'test' containing the training and test sets.
        :rtype: tuple
        """
        if baseline_models_dict is None:
            self.baseline_models_dict = {
                                    # "LinearRegression": LinearRegression(), 
                                    # "PolynomialFeatures": PolynomialFeatures(),
                                    # "DecisionTree": DecisionTreeRegressor(), 
                                    "RandomForest": RandomForestRegressor(),
                                    "GradientBoosting": GradientBoostingRegressor(), 
                                    "SVR": SVR(), 
                                    "KNeighborsRegressor": KNeighborsRegressor(),
                                    "XGBRegressor": XGBRegressor(),
                                    }
        else:
            self.baseline_models_dict = baseline_models_dict
        return self.baseline_models_dict
  

    def covariance(x: np.ndarray, y: np.ndarray) -> float:
        """ Covariance between x and y
        """
        cov_xy = np.cov(x, y)[0][1]
        return cov_xy

    def co_integration(x: np.ndarray, y: np.ndarray):
        """ Co-integration test between x and y
        """
        r, _, _ = tsa.coint(x, y)
        return r

    def correlation(x: np.ndarray,
                    y: np.ndarray,
                    method: str = "pearson"):
        """ Correlation between x and y
        """
        assert method in ["pearson", "spearman", "kendall"]
        corr, p_value = stats.pearsonr(x, y)
        return corr, p_value
    
    # def scale_data(self, df_data: pd.DataFrame, target: str, cutoff_date: str):
    #     """
    #     Scale the input dataframe by subtracting the mean and scaling to unit variance.
        
    #     Args:
    #         df_data (pd.DataFrame): Input data to be scaled.
    #         target (str): The target column name in the data to be used for scaling.
    #         cutoff_date (str): The maximum date for which data is considered in the output.
        
    #     Returns:
    #         tuple: A tuple of two elements. The first element is the scaled dataframe, and the second element
    #         is a pandas datetime object representing the training dates.
    #     """
    #     data = df_data.copy()
    #     if target in data.columns:
    #         pass
    #     else:
    #         data = pd.concat([data, df_data[[target]]], axis=1).dropna()

    #     data = data[data.index<=cutoff_date]
    #     training_date = pd.to_datetime(data.index)
    #     cols = list(data.columns)
    #     df_for_training = data[cols].astype(float)
    #     df_for_training
    #     scaler = StandardScaler()
    #     scaler = scaler.fit(df_for_training)
    #     df_for_training = scaler.transform(df_for_training)

    #     return df_for_training, training_date

    # def lag_generator(self, series: pd.Series, n_lags: int, horizon: int, return_Xy: bool = False):
    #     """
    #     Lag generator
    #     Time series for supervised learning
    #     :param series: time series as pd.Series
    #     :param n_lags: number of past values to used as explanatory variables
    #     :param horizon: how many values to forecast
    #     :param return_Xy: whether to return the lags split from future observations
    #     :return: pd.DataFrame with reconstructed time series
    #     """
    #     self.series = series
    #     self.n_lags = n_lags
    #     self.horizon = horizon
    #     self.return_Xy = return_Xy
    #     assert isinstance(self.series, pd.Series)
    #     if self.series.name is None:
    #         name = 'Series'
    #     else:
    #         name = self.series.name
    #     n_lags_iter = list(range(self.n_lags, -self.horizon, -1))
    #     df_list = [self.series.shift(i) for i in n_lags_iter]
    #     df = pd.concat(df_list, axis=1).dropna()
    #     df.columns = [f'{name}(t-{j - 1})'
    #                 if j > 0 else f'{name}(t+{np.abs(j) + 1})'
    #                 for j in n_lags_iter]
    #     df.columns = [re.sub('t-0', 't', x) for x in df.columns]
    #     if not self.return_Xy:
    #         return df

    #     is_future = df.columns.str.contains('\+')
    #     X = df.iloc[:, ~is_future]
    #     Y = df.iloc[:, is_future]
    #     if Y.shape[1] == 1:
    #         Y = Y.iloc[:, 0]
    #     return X, Y

    # def split_data(self, data, n_past, n_future):
    #     trainX = []
    #     trainY = []

    #     n_future = 1
    #     n_past = 14

    #     for i in range(n_past, len(data) - n_future + 1):
    #         trainX.append(data[i - n_past:i, :data.shape[1]])
    #         trainY.append(data[i + n_future - 1:i + n_future, -1])

    #     trainX, trainY = np.array(trainX), np.array(trainY)

    #     print(f'Shape of train X {trainX.shape}')
    #     print(f'Shape of train y {trainY.shape}')
    #     return trainX, trainY
 
class MachineLearning:
    def __init__(self, data=None):
        """
        Initializes a new instance of the class with a data parameter.

        :param data: The data to be assigned to the instance variable, self.df.
        :type data: Any
        """
        self.df = data