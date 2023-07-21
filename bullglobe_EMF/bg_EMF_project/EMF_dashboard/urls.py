from django.urls import path
from .views import main, dashboard, chart_data

app_name = 'bg_EMF_project'

urlpatterns = [
    path('', main, name='main'),
    path('<slug>/', dashboard, name='dashboard'),
    path('<slug>/chart/', chart_data, name='chart'),
    # path('<slug>/<str:datepicker>/', datepicker, name='datepicker'),
]