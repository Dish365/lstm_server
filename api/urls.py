# api/urls.py
from django.urls import path
from .views import predict_life_expectancy

urlpatterns = [
    path('predict/', predict_life_expectancy, name='predict_life_expectancy'),
]
