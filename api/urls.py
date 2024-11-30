# api/urls.py
from django.urls import path
from .views import (
    predict_life_expectancy, 
    predict_water_share,
    simulate_life_expectancy,
    simulate_water_share,
    simulate,
    health_check
)

urlpatterns = [
    path('predict/life-expectancy/', predict_life_expectancy, name='predict_life_expectancy'),
    path('predict/water-share/', predict_water_share, name='predict_water_share'),
    path('simulate/life-expectancy/', simulate_life_expectancy, name='simulate_life_expectancy'),
    path('simulate/water-share/', simulate_water_share, name='simulate_water_share'),
    path('simulate/', simulate, name='simulate'),
    path('health/', health_check, name='health_check'),
]