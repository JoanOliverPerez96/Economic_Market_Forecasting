from django.shortcuts import render
from django.http import JsonResponse
import plotly.express as px
import pandas as pd
from django.views.generic import View
from rest_framework.views import APIView
from rest_framework.response import Response
import pandas as pd
import numpy as np
def Home(request):
    return render(request, 'index.html')

def User(request):
    username = request.GET['username']
    return render(request, 'user.html', {'name': username})

def Target(request):
    target = request.GET['target']
    return render(request, 'dashboard.html', {'target':target}) # context=mydict)

def get_data(request):
    data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\Economic_Market_Forecasting\Economic_Market_Forecasting\EMF_webapp\EMF_project\data\SP500_data.csv')
    data.set_index("Date", inplace=True)
    data.index = pd.to_datetime(data.index)
    data = data.cumsum().resample("M").last()
    # data = data["SP500_Prediction"]
    data.index = data.index.strftime('%m/%d/%Y')
    data["Date"] = data.index
    data_json = data.to_json()
    return JsonResponse(data_json, safe=False)

class ChartDataSP500(APIView):
    authorization_classes = []
    permission_classes = []

    def get(self, request, format=None):
        data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\SP500_data.csv', index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.cumsum().resample("M").last()
        data.index = data.index
        data["Date"] = data.index.strftime('%m/%y')
        labels = data["Date"].to_list()
        default_items = data[f"SP500_Prediction"].to_list()
        actual_data = data[f"SP500"].to_list()
        data = {
                "labels": labels,
                "default": default_items,
                "actual": actual_data
        }
        return Response(data)
        
def SP500(request):
    data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\SP500_data.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.cumsum().resample("M").last()
    data.index = data.index
    data["Date"] = data.index.strftime('%m/%y')
    default_items = data[f"SP500_Prediction"].to_list()[-1]
    if default_items > 0.3:
        prediction = "Strong Buy"
    elif default_items > 0.15:
        prediction = "Buy"
    elif default_items >= 0.05:
        prediction = "Neutral"
    elif default_items <= 0:
        prediction = "Sell"
    elif default_items < -0.05:
        prediction = "Strong Sell"
    returns = np.round(default_items*100,2)
    return render(request, 'SP500.html', {'prediction': prediction,'returns':returns})

class ChartDataNASDAQ(APIView):
    authorization_classes = []
    permission_classes = []

    def get(self, request, format=None):
        data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\NASDAQ_data.csv', index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.cumsum().resample("M").last()
        data.index = data.index
        data["Date"] = data.index.strftime('%m/%y')
        labels = data["Date"].to_list()
        default_items = data[f"NASDAQ_Prediction"].to_list()
        actual_data = data[f"NASDAQ"].to_list()
        data = {
                "labels": labels,
                "default": default_items,
                "actual": actual_data
        }
        return Response(data)
    
def NASDAQ(request):
    data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\NASDAQ_data.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.cumsum().resample("M").last()
    data.index = data.index
    data["Date"] = data.index.strftime('%m/%y')
    default_items = data[f"NASDAQ_Prediction"].to_list()[-1]
    if default_items > 0.3:
        prediction = "Strong Buy"
    elif default_items > 0.15:
        prediction = "Buy"
    elif default_items >= 0.05:
        prediction = "Neutral"
    elif default_items <= 0:
        prediction = "Sell"
    elif default_items < -0.05:
        prediction = "Strong Sell"
    returns = np.round(default_items*100,2)
    return render(request, 'NASDAQ.html', {'prediction': prediction,'returns':returns})

class ChartDataDow_Jones(APIView):
    authorization_classes = []
    permission_classes = []

    def get(self, request, format=None):
        data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Dow_Jones_data.csv', index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.cumsum().resample("M").last()
        data.index = data.index
        data["Date"] = data.index.strftime('%m/%y')
        labels = data["Date"].to_list()
        default_items = data[f"Dow_Jones_Prediction"].to_list()
        actual_data = data[f"Dow_Jones"].to_list()
        data = {
                "labels": labels,
                "default": default_items,
                "actual": actual_data
        }
        return Response(data)
        
def Dow_Jones(request):
    data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Dow_Jones_data.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.cumsum().resample("M").last()
    data.index = data.index
    data["Date"] = data.index.strftime('%m/%y')
    default_items = data[f"Dow_Jones_Prediction"].to_list()[-1]
    if default_items > 0.3:
        prediction = "Strong Buy"
    elif default_items > 0.15:
        prediction = "Buy"
    elif default_items >= 0.05:
        prediction = "Neutral"
    elif default_items <= 0:
        prediction = "Sell"
    elif default_items < -0.05:
        prediction = "Strong Sell"
    returns = np.round(default_items*100,2)
    return render(request, 'Dow_Jones.html', {'prediction': prediction,'returns':returns})

class ChartDataCorporateBonds(APIView):
    authorization_classes = []
    permission_classes = []

    def get(self, request, format=None):
        data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\CorporateBonds_data.csv', index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.cumsum().resample("M").last()
        data.index = data.index
        data["Date"] = data.index.strftime('%m/%y')
        labels = data["Date"].to_list()
        default_items = data[f"CorporateBonds_Prediction"].to_list()
        actual_data = data[f"CorporateBonds"].to_list()
        data = {
                "labels": labels,
                "default": default_items,
                "actual": actual_data
        }
        return Response(data)
        
def CorporateBonds(request):
    data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\CorporateBonds_data.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.cumsum().resample("M").last()
    data.index = data.index
    data["Date"] = data.index.strftime('%m/%y')
    default_items = data[f"CorporateBonds_Prediction"].to_list()[-1]
    if default_items > 0.3:
        prediction = "Strong Buy"
    elif default_items > 0.15:
        prediction = "Buy"
    elif default_items >= 0.05:
        prediction = "Neutral"
    elif default_items <= 0:
        prediction = "Sell"
    elif default_items < -0.05:
        prediction = "Strong Sell"
    returns = np.round(default_items*100,2)
    return render(request, 'CorporateBonds.html', {'prediction': prediction,'returns':returns})
