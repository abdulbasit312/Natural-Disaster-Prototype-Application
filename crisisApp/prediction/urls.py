from operator import index
from django.urls import path
from . import views

app_name = "prediction"

urlpatterns = [
    path("", views.index),
    path("upload", views.send_files, name="uploads"),
    path("second",views.second, name="second"),
    path("second1",views.second1, name="second1"),
    path("second2",views.second2, name="second2"),
    path("second3",views.second3, name="second3"),
    path("second4",views.second4, name="second4"),
]