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


class ChartDataGold(APIView):
    authorization_classes = []
    permission_classes = []

    def get(self, request, format=None):
        data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Gold_data.csv', index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.cumsum().resample("M").last()
        data.index = data.index
        data["Date"] = data.index.strftime('%m/%y')
        labels = data["Date"].to_list()
        default_items = data[f"Gold_Prediction"].to_list()
        actual_data = data[f"Gold"].to_list()
        data = {
                "labels": labels,
                "default": default_items,
                "actual": actual_data
        }
        return Response(data)
        
def Gold(request):
    data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Gold_data.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.cumsum().resample("M").last()
    data.index = data.index
    data["Date"] = data.index.strftime('%m/%y')
    default_items = data[f"Gold_Prediction"].to_list()[-1]
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
    return render(request, 'Gold.html', {'prediction': prediction,'returns':returns})


class ChartDataDiscretionary(APIView):
    authorization_classes = []
    permission_classes = []

    def get(self, request, format=None):
        data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Discretionary_data.csv', index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.cumsum().resample("M").last()
        data.index = data.index
        data["Date"] = data.index.strftime('%m/%y')
        labels = data["Date"].to_list()
        default_items = data[f"Discretionary_Prediction"].to_list()
        actual_data = data[f"Discretionary"].to_list()
        data = {
                "labels": labels,
                "default": default_items,
                "actual": actual_data
        }
        return Response(data)
        
def Discretionary(request):
    data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Discretionary_data.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.cumsum().resample("M").last()
    data.index = data.index
    data["Date"] = data.index.strftime('%m/%y')
    default_items = data[f"Discretionary_Prediction"].to_list()[-1]
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
    return render(request, 'Discretionary.html', {'prediction': prediction,'returns':returns})


class ChartDataEnergy(APIView):
    authorization_classes = []
    permission_classes = []

    def get(self, request, format=None):
        data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Energy_data.csv', index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.cumsum().resample("M").last()
        data.index = data.index
        data["Date"] = data.index.strftime('%m/%y')
        labels = data["Date"].to_list()
        default_items = data[f"Energy_Prediction"].to_list()
        actual_data = data[f"Energy"].to_list()
        data = {
                "labels": labels,
                "default": default_items,
                "actual": actual_data
        }
        return Response(data)
        
def Energy(request):
    data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Energy_data.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.cumsum().resample("M").last()
    data.index = data.index
    data["Date"] = data.index.strftime('%m/%y')
    default_items = data[f"Energy_Prediction"].to_list()[-1]
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
    return render(request, 'Energy.html', {'prediction': prediction,'returns':returns})


class ChartDataFinancials(APIView):
    authorization_classes = []
    permission_classes = []

    def get(self, request, format=None):
        data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Financials_data.csv', index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.cumsum().resample("M").last()
        data.index = data.index
        data["Date"] = data.index.strftime('%m/%y')
        labels = data["Date"].to_list()
        default_items = data[f"Financials_Prediction"].to_list()
        actual_data = data[f"Financials"].to_list()
        data = {
                "labels": labels,
                "default": default_items,
                "actual": actual_data
        }
        return Response(data)
        
def Financials(request):
    data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Financials_data.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.cumsum().resample("M").last()
    data.index = data.index
    data["Date"] = data.index.strftime('%m/%y')
    default_items = data[f"Financials_Prediction"].to_list()[-1]
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
    return render(request, 'Financials.html', {'prediction': prediction,'returns':returns})


class ChartDataHealthcare(APIView):
    authorization_classes = []
    permission_classes = []

    def get(self, request, format=None):
        data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Healthcare_data.csv', index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.cumsum().resample("M").last()
        data.index = data.index
        data["Date"] = data.index.strftime('%m/%y')
        labels = data["Date"].to_list()
        default_items = data[f"Healthcare_Prediction"].to_list()
        actual_data = data[f"Healthcare"].to_list()
        data = {
                "labels": labels,
                "default": default_items,
                "actual": actual_data
        }
        return Response(data)
        
def Healthcare(request):
    data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Healthcare_data.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.cumsum().resample("M").last()
    data.index = data.index
    data["Date"] = data.index.strftime('%m/%y')
    default_items = data[f"Healthcare_Prediction"].to_list()[-1]
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
    return render(request, 'Healthcare.html', {'prediction': prediction,'returns':returns})


class ChartDataIndustrials(APIView):
    authorization_classes = []
    permission_classes = []

    def get(self, request, format=None):
        data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Industrials_data.csv', index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.cumsum().resample("M").last()
        data.index = data.index
        data["Date"] = data.index.strftime('%m/%y')
        labels = data["Date"].to_list()
        default_items = data[f"Industrials_Prediction"].to_list()
        actual_data = data[f"Industrials"].to_list()
        data = {
                "labels": labels,
                "default": default_items,
                "actual": actual_data
        }
        return Response(data)
        
def Industrials(request):
    data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Industrials_data.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.cumsum().resample("M").last()
    data.index = data.index
    data["Date"] = data.index.strftime('%m/%y')
    default_items = data[f"Industrials_Prediction"].to_list()[-1]
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
    return render(request, 'Industrials.html', {'prediction': prediction,'returns':returns})

class ChartDataMaterials(APIView):
    authorization_classes = []
    permission_classes = []

    def get(self, request, format=None):
        data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Materials_data.csv', index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.cumsum().resample("M").last()
        data.index = data.index
        data["Date"] = data.index.strftime('%m/%y')
        labels = data["Date"].to_list()
        default_items = data[f"Materials_Prediction"].to_list()
        actual_data = data[f"Materials"].to_list()
        data = {
                "labels": labels,
                "default": default_items,
                "actual": actual_data
        }
        return Response(data)
        
def Materials(request):
    data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Materials_data.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.cumsum().resample("M").last()
    data.index = data.index
    data["Date"] = data.index.strftime('%m/%y')
    default_items = data[f"Materials_Prediction"].to_list()[-1]
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
    return render(request, 'Materials.html', {'prediction': prediction,'returns':returns})


class ChartDataRealEstate(APIView):
    authorization_classes = []
    permission_classes = []

    def get(self, request, format=None):
        data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\RealEstate_data.csv', index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.cumsum().resample("M").last()
        data.index = data.index
        data["Date"] = data.index.strftime('%m/%y')
        labels = data["Date"].to_list()
        default_items = data[f"RealEstate_Prediction"].to_list()
        actual_data = data[f"RealEstate"].to_list()
        data = {
                "labels": labels,
                "default": default_items,
                "actual": actual_data
        }
        return Response(data)
        
def RealEstate(request):
    data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\RealEstate_data.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.cumsum().resample("M").last()
    data.index = data.index
    data["Date"] = data.index.strftime('%m/%y')
    default_items = data[f"RealEstate_Prediction"].to_list()[-1]
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
    return render(request, 'RealEstate.html', {'prediction': prediction,'returns':returns})


class ChartDataStaples(APIView):
    authorization_classes = []
    permission_classes = []

    def get(self, request, format=None):
        data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Staples_data.csv', index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.cumsum().resample("M").last()
        data.index = data.index
        data["Date"] = data.index.strftime('%m/%y')
        labels = data["Date"].to_list()
        default_items = data[f"Staples_Prediction"].to_list()
        actual_data = data[f"Staples"].to_list()
        data = {
                "labels": labels,
                "default": default_items,
                "actual": actual_data
        }
        return Response(data)
        
def Staples(request):
    data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Staples_data.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.cumsum().resample("M").last()
    data.index = data.index
    data["Date"] = data.index.strftime('%m/%y')
    default_items = data[f"Staples_Prediction"].to_list()[-1]
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
    return render(request, 'Staples.html', {'prediction': prediction,'returns':returns})


class ChartDataTechnology(APIView):
    authorization_classes = []
    permission_classes = []

    def get(self, request, format=None):
        data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Technology_data.csv', index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.cumsum().resample("M").last()
        data.index = data.index
        data["Date"] = data.index.strftime('%m/%y')
        labels = data["Date"].to_list()
        default_items = data[f"Technology_Prediction"].to_list()
        actual_data = data[f"Technology"].to_list()
        data = {
                "labels": labels,
                "default": default_items,
                "actual": actual_data
        }
        return Response(data)
        
def Technology(request):
    data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Technology_data.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.cumsum().resample("M").last()
    data.index = data.index
    data["Date"] = data.index.strftime('%m/%y')
    default_items = data[f"Technology_Prediction"].to_list()[-1]
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
    return render(request, 'Technology.html', {'prediction': prediction,'returns':returns})


class ChartDataUtilities(APIView):
    authorization_classes = []
    permission_classes = []

    def get(self, request, format=None):
        data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Utilities_data.csv', index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.cumsum().resample("M").last()
        data.index = data.index
        data["Date"] = data.index.strftime('%m/%y')
        labels = data["Date"].to_list()
        default_items = data[f"Utilities_Prediction"].to_list()
        actual_data = data[f"Utilities"].to_list()
        data = {
                "labels": labels,
                "default": default_items,
                "actual": actual_data
        }
        return Response(data)
        
def Utilities(request):
    data = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Utilities_data.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.cumsum().resample("M").last()
    data.index = data.index
    data["Date"] = data.index.strftime('%m/%y')
    default_items = data[f"Utilities_Prediction"].to_list()[-1]
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
    return render(request, 'Utilities.html', {'prediction': prediction,'returns':returns})

class ChartDataPredictions(APIView):
    authorization_classes = []
    permission_classes = []

    def get(self, request, format=None):
        data1 = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\SP500_data.csv', index_col=0)
        data2= pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\NASDAQ_data.csv', index_col=0)
        data3 = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Dow_Jones_data.csv', index_col=0)
        data4 = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\CorporateBonds_data.csv', index_col=0)
        data5 = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Gold_data.csv', index_col=0)
        data6 = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Discretionary_data.csv', index_col=0)
        data7 = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Energy_data.csv', index_col=0)
        data8 = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Financials_data.csv', index_col=0)
        data9 = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Healthcare_data.csv', index_col=0)
        data10 = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Industrials_data.csv', index_col=0)
        data11 = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Materials_data.csv', index_col=0)
        data12 = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\RealEstate_data.csv', index_col=0)
        data13 = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Staples_data.csv', index_col=0)
        data14 = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Technology_data.csv', index_col=0)
        data15 = pd.read_csv(r'C:\Users\Joan Oliver\Documents\GitHub\EMF_project\data\Utilities_data.csv', index_col=0)

        data_lst = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15]
        data = pd.DataFrame()
        data_dict = {}
        for dt in data_lst:
            for col in dt.columns:
                if "_Predict" in col:
                    data[col] = dt[col]
                    data_dict[col] = data[col]

        data.index = pd.to_datetime(data.index)
        data = data.cumsum().resample("M").last()
        data.index = data.index
        data["Date"] = data.index.strftime('%m/%y')
        labels = data["Date"].to_list()
        default_items = data[f"Predic_Prediction"].to_list()
        actual_data = data[f"Predic"].to_list()
        data = {
                "labels": labels,
                "default": default_items,
                "actual": actual_data
        }
        return Response(data_dict)