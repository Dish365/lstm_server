# views.py
from django.http import JsonResponse
from rest_framework.decorators import api_view
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import os
from django.conf import settings
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Bidirectional, LSTM  # Import necessary layers

# Paths to model and scaler files
MODEL_FILENAME = 'lstm_life_expectancy.h5'  # Ensure this matches your saved model filename
FEATURE_SCALER_FILENAME = 'feature_scaler.save'
TARGET_SCALER_FILENAME = 'target_scaler.save'
MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', MODEL_FILENAME)
FEATURE_SCALER_PATH = os.path.join(settings.BASE_DIR, 'models', FEATURE_SCALER_FILENAME)
TARGET_SCALER_PATH = os.path.join(settings.BASE_DIR, 'models', TARGET_SCALER_FILENAME)

FEATURE_NAMES = [
    'FP index', 'LP index', 'Vegetal Pds-FS', 'Cereals -FS', 'Starchy Rts-FS',
    'Pulses-FS', 'Fruits -FS', 'Meat-FS', 'Fish-FS', 'Sugar & Swt-FS',
    'Oils-FS', 'Vegetables-FS', 'Spices-FS', 'Eggs-FS', 'Milk-FS',
    'Cereals-LSF', 'Starchy Rts-LSF', 'Pulses-LSF', 'Meat-LSF', 'Fish-LSF',
    'Cereals-LS', 'Starchy-LS', 'Fruits-LS', 'Energy use', 'Renewable energy'
]

def correlation_coefficient(y_true, y_pred):
    """Custom metric for correlation coefficient"""
    x = y_true - K.mean(y_true)
    y = y_pred - K.mean(y_pred)
    numerator = K.sum(x * y)
    denominator = K.sqrt(K.sum(K.square(x)) * K.sum(K.square(y)))
    return numerator / (denominator + K.epsilon())

def load_ml_models():
    """Load model and scalers"""
    paths = [MODEL_PATH, FEATURE_SCALER_PATH, TARGET_SCALER_PATH]
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found at: {path}")

    # Include custom_objects if any custom metrics or layers are used
    custom_objects = {
        'correlation_coefficient': correlation_coefficient,
        'Bidirectional': Bidirectional,
        'LSTM': LSTM
    }

    model = load_model(MODEL_PATH, custom_objects=custom_objects)
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)

    return model, feature_scaler, target_scaler

try:
    model, feature_scaler, target_scaler = load_ml_models()
except Exception as e:
    import warnings
    warnings.warn(f"Failed to load model or scalers: {str(e)}")
    model = feature_scaler = target_scaler = None

def create_sequence(scaled_data, sequence_length=3):
    """Create sequence for time-series prediction"""
    if len(scaled_data) < sequence_length:
        # Repeat the data to create minimum sequence length
        scaled_data = np.tile(scaled_data, (sequence_length, 1))
    else:
        scaled_data = scaled_data[-sequence_length:]
    return scaled_data.reshape(1, sequence_length, scaled_data.shape[1])

def preprocess_input(input_data):
    """Preprocess the input data for the model"""
    input_array = np.array([input_data])  # Shape: (1, num_features)
    scaled_input = feature_scaler.transform(input_array)  # Scale the input
    return create_sequence(scaled_input)  # Shape: (1, sequence_length, num_features)

@api_view(["POST"])
def predict_life_expectancy(request):
    try:
        if None in (model, feature_scaler, target_scaler):
            return JsonResponse({"error": "Model or scalers not loaded"}, status=500)

        input_data = request.data.get("features")
        if not input_data or len(input_data) != len(FEATURE_NAMES):
            return JsonResponse({"error": f"Expected {len(FEATURE_NAMES)} features"}, status=400)

        # Ensure all features are numbers
        try:
            input_data = [float(value) for value in input_data]
        except ValueError:
            return JsonResponse({"error": "All features must be numeric values"}, status=400)

        # Preprocess and predict
        model_input = preprocess_input(input_data)
        scaled_prediction = model.predict(model_input, verbose=0)
        prediction = target_scaler.inverse_transform(scaled_prediction)
        predicted_value = round(float(prediction[0][0]), 2)

        # Return results
        return JsonResponse({
            "predicted_life_expectancy": predicted_value,
            "features_received": dict(zip(FEATURE_NAMES, input_data))
        }, status=200)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
