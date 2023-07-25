from django.contrib import admin
from django.urls import path, include
# from . import views
# from .views import *

urlpatterns = [
    path('', include("EMF_app.urls")),
    # path(r'^api/data/$', get_data, name='data'),
    path('admin/', admin.site.urls),
]
