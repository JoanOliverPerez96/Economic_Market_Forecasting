from django.urls import path
from . import views

urlpatterns = [
    path('', views.Home, name='home'),
    path('target', views.Target, name='target'),
]