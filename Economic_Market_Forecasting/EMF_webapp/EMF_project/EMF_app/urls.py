from django.urls import path, include
from . import views
from .views import Home, Target, get_data, ChartDataSP500, ChartDataNASDAQ, ChartDataDow_Jones, ChartDataCorporateBonds

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
]