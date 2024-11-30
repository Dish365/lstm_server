# api/urls.py
from django.urls import path
from .views import (
    predict_life_expectancy, 
    simulate_life_expectancy,
    predict_water_share,
    simulate_water_share,
    simulate
)

urlpatterns = [
    # Prediction endpoints
    path('predict/life-expectancy/', predict_life_expectancy, name='predict_life_expectancy'),
    path('predict/water-share/', predict_water_share, name='predict_water_share'),
    
    # Simulation endpoints
    path('simulate/life-expectancy/', simulate_life_expectancy, name='simulate_life_expectancy'),
    path('simulate/water-share/', simulate_water_share, name='simulate_water_share'),
    
    # Generic simulation endpoint
    path('simulate/', simulate, name='simulate'),
]