# General libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pandas_datareader import data as pdr
import yfinance as yfin
import datetime as dt
from datetime import datetime
import seaborn as sns
import urllib.request
from PIL import Image
import re
from path import Path
from scipy import stats
import statsmodels.api as sm
from IPython.display import display, HTML
from scipy.stats import pearsonr
import itertools
from scipy.stats import ttest_ind
import statsmodels.tsa.stattools as tsa

# FRED library
from fredapi import Fred
# API Key
fred_key = '2e3cf97d1b456831253eda002ce25948'

## Machine Learning libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
# Deep Learning Models
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, GRU, Conv1D, Dense, Flatten, Dropout
from statsmodels.tsa.statespace.varmax import VARMAX
from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier
from kerasbeats import prep_time_series, NBeatsModel

# Metrics and processing
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error as mape

import warnings
warnings.filterwarnings('ignore')
