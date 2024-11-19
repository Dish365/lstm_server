import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
import joblib
import os
from django.conf import settings
import tensorflow as tf

# Constants
MODEL_FILENAME = 'lstm2024.h5'
SCALER_FILENAME = 'scaler.save'
MODEL_PATH = os.path.join('new_models', MODEL_FILENAME)
SCALER_PATH = os.path.join('new_models', SCALER_FILENAME)

# Load model and scaler
custom_objects = {
    'MeanSquaredError': tf.keras.losses.MeanSquaredError,
    'MeanAbsoluteError': tf.keras.metrics.MeanAbsoluteError
}
model = load_model(MODEL_PATH, custom_objects=custom_objects)
scaler = joblib.load(SCALER_PATH)

def evaluate_model(model, scaler, X_data, y_true):
    # Preprocess test data
    X_scaled = scaler.transform(X_data)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    
    # Get predictions
    y_pred = model.predict(X_reshaped).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmae = np.sqrt(mae)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'RMAE': rmae,
        'R2': r2
    }

# Load your test data
df = pd.read_csv('Data_GHGE.csv')
feature_columns = [
    'FP index ', 'LP index ', 'Vegetal Pds-FS', 'Cereals -FS', 'Starchy Rts-FS',
    'Pulses-FS', 'Fruits -FS', 'Meat-FS', 'Fish-FS', 'Sugar & Swt-FS', 
    'Oils-FS ', 'Vegetables-FS ', 'Spices-FS', 'Eggs-FS', 'Milk-FS',
    'Cereals-LSF', 'Starchy Rts-LSF', 'Pulses-LSF', 'Meat-LSF', 'Fish-LSF',
    'Cereals-LS', 'Starchy-LS', 'Fruits-LS', 'Energy use', 'Renewable energy '
]

X = df[feature_columns].values
y = df['Life expectancy '].values

# Evaluate
metrics = evaluate_model(model, scaler, X, y)
print("\nModel Performance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")