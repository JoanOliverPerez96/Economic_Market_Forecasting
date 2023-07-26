from django.urls import path, include
from . import views
from .views import *

urlpatterns = [
    path('', views.Home, name='home'),
    path('api/data/', get_data, name='data'),
    path('api/chart/data/SP500', ChartDataSP500.as_view(), name='chartdata'),
    path('prediction/SP500', views.SP500, name='target'),

    path('api/chart/data/NASDAQ', ChartDataNASDAQ.as_view(), name='chartdata'),
    path('prediction/NASDAQ', views.NASDAQ, name='target'),
    
    path('api/chart/data/Dow_Jones', ChartDataDow_Jones.as_view(), name='chartdata'),
    path('prediction/Dow_Jones', views.Dow_Jones, name='target'),
    
    path('api/chart/data/CorporateBonds', ChartDataCorporateBonds.as_view(), name='chartdata'),
    path('prediction/CorporateBonds', views.CorporateBonds, name='target'),
    
    path('api/chart/data/Gold', ChartDataGold.as_view(), name='chartdata'),
    path('prediction/Gold', views.Gold, name='target'),
    
    path('api/chart/data/Discretionary', ChartDataDiscretionary.as_view(), name='chartdata'),
    path('prediction/Discretionary', views.Discretionary, name='target'),
    
    path('api/chart/data/Energy', ChartDataEnergy.as_view(), name='chartdata'),
    path('prediction/Energy', views.Energy, name='target'),
    
    path('api/chart/data/Financials', ChartDataFinancials.as_view(), name='chartdata'),
    path('prediction/Financials', views.Financials, name='target'),
    
    path('api/chart/data/Healthcare', ChartDataHealthcare.as_view(), name='chartdata'),
    path('prediction/Healthcare', views.Healthcare, name='target'),
    
    path('api/chart/data/Industrials', ChartDataIndustrials.as_view(), name='chartdata'),
    path('prediction/Industrials', views.Industrials, name='target'),
    
    path('api/chart/data/Materials', ChartDataMaterials.as_view(), name='chartdata'),
    path('prediction/Materials', views.Materials, name='target'),
    
    path('api/chart/data/RealEstate', ChartDataRealEstate.as_view(), name='chartdata'),
    path('prediction/RealEstate', views.RealEstate, name='target'),
    
    path('api/chart/data/Staples', ChartDataStaples.as_view(), name='chartdata'),
    path('prediction/Staples', views.Staples, name='target'),
    
    path('api/chart/data/Technology', ChartDataTechnology.as_view(), name='chartdata'),
    path('prediction/Technology', views.Technology, name='target'),
    
    path('api/chart/data/Utilities', ChartDataUtilities.as_view(), name='chartdata'),
    path('prediction/Utilities', views.Utilities, name='target'),
]
