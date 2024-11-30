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
from tensorflow.keras.layers import Bidirectional, LSTM
from typing import List, Dict, Any, Tuple, Union, Optional
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Paths for life expectancy model
MODEL_FILENAME = 'lstm_life_expectancy.h5'
FEATURE_SCALER_FILENAME = 'feature_scaler.save'
TARGET_SCALER_FILENAME = 'target_scaler.save'
MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', MODEL_FILENAME)
FEATURE_SCALER_PATH = os.path.join(settings.BASE_DIR, 'models', FEATURE_SCALER_FILENAME)
TARGET_SCALER_PATH = os.path.join(settings.BASE_DIR, 'models', TARGET_SCALER_FILENAME)

# Paths for water share model
ENV_MODEL_FILENAME = 'env_bilstm_model.h5'
ENV_FEATURE_SCALER_FILENAME = 'env_feature_scaler.save'
ENV_TARGET_SCALER_FILENAME = 'env_target_scaler.save'
ENV_MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', ENV_MODEL_FILENAME)
ENV_FEATURE_SCALER_PATH = os.path.join(settings.BASE_DIR, 'models', ENV_FEATURE_SCALER_FILENAME)
ENV_TARGET_SCALER_PATH = os.path.join(settings.BASE_DIR, 'models', ENV_TARGET_SCALER_FILENAME)

# Debug logging
print("Current working directory:", os.getcwd())
print("BASE_DIR:", settings.BASE_DIR)
print("Model paths:")
print(f"Life expectancy model: {MODEL_PATH}")
print(f"Life expectancy feature scaler: {FEATURE_SCALER_PATH}")
print(f"Life expectancy target scaler: {TARGET_SCALER_PATH}")
print(f"Water share model: {ENV_MODEL_PATH}")
print(f"Water share feature scaler: {ENV_FEATURE_SCALER_PATH}")
print(f"Water share target scaler: {ENV_TARGET_SCALER_PATH}")

# Verify file existence
for path in [MODEL_PATH, FEATURE_SCALER_PATH, TARGET_SCALER_PATH, 
             ENV_MODEL_PATH, ENV_FEATURE_SCALER_PATH, ENV_TARGET_SCALER_PATH]:
    print(f"File exists {path}: {os.path.exists(path)}")

# Feature names for both models
FEATURE_NAMES = [
    'FP index', 'LP index', 'Vegetal Pds-FS', 'Cereals -FS', 'Starchy Rts-FS',
    'Pulses-FS', 'Fruits -FS', 'Meat-FS', 'Fish-FS', 'Sugar & Swt-FS',
    'Oils-FS', 'Vegetables-FS', 'Spices-FS', 'Eggs-FS', 'Milk-FS',
    'Cereals-LSF', 'Starchy Rts-LSF', 'Pulses-LSF', 'Meat-LSF', 'Fish-LSF',
    'Cereals-LS', 'Starchy-LS', 'Fruits-LS', 'Energy use', 'Renewable energy'
]

ENV_FEATURE_NAMES = FEATURE_NAMES  # Both models use the same features

class ModelType(Enum):
    """Enum to differentiate between model types"""
    LIFE_EXPECTANCY = "life_expectancy"
    WATER_SHARE = "water_share"

class SimulationType(Enum):
    """Enumeration of supported simulation types"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    CUSTOM = "custom"

def correlation_coefficient(y_true, y_pred):
    """
    Custom metric for correlation coefficient used in both models
    Args:
        y_true: True values
        y_pred: Predicted values
    Returns:
        Correlation coefficient between true and predicted values
    """
    x = y_true - K.mean(y_true)
    y = y_pred - K.mean(y_pred)
    numerator = K.sum(x * y)
    denominator = K.sqrt(K.sum(K.square(x)) * K.sum(K.square(y)))
    return numerator / (denominator + K.epsilon())

def create_sequence(scaled_data: np.ndarray, sequence_length: int = 3) -> np.ndarray:
    """
    Create sequences for LSTM input, handling cases with insufficient data points
    Args:
        scaled_data: Scaled input data
        sequence_length: Desired sequence length for LSTM input
    Returns:
        Sequence data shaped for LSTM input (samples, sequence_length, features)
    """
    if len(scaled_data) < sequence_length:
        # If we don't have enough data points, repeat the data
        scaled_data = np.tile(scaled_data, (sequence_length, 1))
    
    # Take the last sequence_length points
    sequence = scaled_data[-sequence_length:]
    
    # Reshape for LSTM input (batch_size, sequence_length, features)
    return sequence.reshape(1, sequence_length, scaled_data.shape[1])

def preprocess_input(
    input_data: List[float],
    feature_scaler: Any,
    sequence_length: int = 3
) -> np.ndarray:
    """
    Preprocess input data for model prediction
    Args:
        input_data: Raw input features
        feature_scaler: Fitted scaler for features
        sequence_length: Length of sequence for LSTM input
    Returns:
        Preprocessed data ready for model input
    """
    # Convert to numpy array and reshape
    input_array = np.array(input_data).reshape(1, -1)
    
    # Scale the input
    scaled_input = feature_scaler.transform(input_array)
    
    # Create sequence
    return create_sequence(scaled_input, sequence_length)

def load_model_and_scalers(model_type: ModelType) -> Tuple[Any, Any, Any]:
    """
    Load model and scalers based on model type
    Args:
        model_type: Type of model to load (life expectancy or water share)
    Returns:
        Tuple of (model, feature_scaler, target_scaler)
    """
    try:
        if model_type == ModelType.LIFE_EXPECTANCY:
            model_path = MODEL_PATH
            feature_scaler_path = FEATURE_SCALER_PATH
            target_scaler_path = TARGET_SCALER_PATH
        else:
            model_path = ENV_MODEL_PATH
            feature_scaler_path = ENV_FEATURE_SCALER_PATH
            target_scaler_path = ENV_TARGET_SCALER_PATH

        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Loading feature scaler from: {feature_scaler_path}")
        logger.info(f"Loading target scaler from: {target_scaler_path}")

        # Check if files exist
        for path in [model_path, feature_scaler_path, target_scaler_path]:
            if not os.path.exists(path):
                logger.error(f"File not found at: {path}")
                raise FileNotFoundError(f"File not found at: {path}")

        # Load model with custom objects
        custom_objects = {
            'correlation_coefficient': correlation_coefficient,
            'Bidirectional': Bidirectional,
            'LSTM': LSTM
        }
        
        model = load_model(model_path, custom_objects=custom_objects)
        logger.info("Model loaded successfully")
        
        feature_scaler = joblib.load(feature_scaler_path)
        logger.info("Feature scaler loaded successfully")
        
        target_scaler = joblib.load(target_scaler_path)
        logger.info("Target scaler loaded successfully")

        return model, feature_scaler, target_scaler
    except Exception as e:
        logger.error(f"Error loading model and scalers: {str(e)}")
        raise

# Load both models at startup
try:
    life_model, life_feature_scaler, life_target_scaler = load_model_and_scalers(ModelType.LIFE_EXPECTANCY)
    water_model, water_feature_scaler, water_target_scaler = load_model_and_scalers(ModelType.WATER_SHARE)
except Exception as e:
    import warnings
    warnings.warn(f"Failed to load models or scalers: {str(e)}")
    life_model = life_feature_scaler = life_target_scaler = None
    water_model = water_feature_scaler = water_target_scaler = None

class SimulationHandler:
    """Handles simulation logic for both life expectancy and water share predictions"""
    
    def __init__(self, model_type: ModelType):
        """
        Initialize simulation handler with appropriate model and scalers
        Args:
            model_type: Type of model to use for simulation
        """
        if model_type == ModelType.LIFE_EXPECTANCY:
            self.model = life_model
            self.feature_scaler = life_feature_scaler
            self.target_scaler = life_target_scaler
            self.feature_names = FEATURE_NAMES
        else:
            self.model = water_model
            self.feature_scaler = water_feature_scaler
            self.target_scaler = water_target_scaler
            self.feature_names = ENV_FEATURE_NAMES

    def _update_features(
        self,
        current_features: List[float],
        simulation_type: str,
        change_rates: Dict[str, float],
        interval: int = 1
    ) -> List[float]:
        """
        Update feature values based on simulation parameters
        Args:
            current_features: Current feature values
            simulation_type: Type of simulation (linear or exponential)
            change_rates: Dictionary of feature names and their change rates
            interval: Time interval for changes
        Returns:
            Updated feature values
        """
        updated_features = current_features.copy()
        
        for feature_name, rate in change_rates.items():
            if feature_name in self.feature_names:
                idx = self.feature_names.index(feature_name)
                current_value = current_features[idx]
                
                if simulation_type == SimulationType.LINEAR.value:
                    change = current_value * (float(rate) / 100) * interval
                    updated_features[idx] = current_value + change
                else:  # exponential
                    compound_rate = (1 + float(rate) / 100) ** interval
                    updated_features[idx] = current_value * compound_rate
        
        return updated_features

    def run_simulation(
        self,
        initial_features: List[float],
        years: int,
        simulation_type: str,
        change_rates: Dict[str, float],
        interval: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Run complete simulation
        Args:
            initial_features: Starting feature values
            years: Number of years to simulate
            simulation_type: Type of simulation (linear or exponential)
            change_rates: Dictionary of feature names and their change rates
            interval: Time interval between predictions
        Returns:
            List of simulation results for each time step
        """
        results = []
        current_features = initial_features.copy()
        
        for year in range(0, years + 1, interval):
            # Preprocess and predict
            model_input = preprocess_input(current_features, self.feature_scaler)
            scaled_prediction = self.model.predict(model_input, verbose=0)
            prediction = self.target_scaler.inverse_transform(scaled_prediction)
            predicted_value = round(float(prediction[0][0]), 2)
            
            # Store results
            results.append({
                "year": year,
                "prediction": predicted_value,
                "features": dict(zip(self.feature_names, current_features))
            })
            
            # Update features for next iteration
            if year < years:
                current_features = self._update_features(
                    current_features,
                    simulation_type,
                    change_rates,
                    interval
                )
        
        return results

@api_view(["POST"])
def predict_life_expectancy(request):
    """API endpoint for life expectancy prediction"""
    try:
        if None in (life_model, life_feature_scaler, life_target_scaler):
            return JsonResponse({"error": "Life expectancy model or scalers not loaded"}, status=500)

        # Validate input
        input_data = request.data.get("features")
        if not input_data or len(input_data) != len(FEATURE_NAMES):
            return JsonResponse({"error": f"Expected {len(FEATURE_NAMES)} features"}, status=400)

        # Ensure all features are numbers
        try:
            input_data = [float(value) for value in input_data]
        except ValueError:
            return JsonResponse({"error": "All features must be numeric values"}, status=400)

        # Preprocess and predict
        model_input = preprocess_input(input_data, life_feature_scaler)
        scaled_prediction = life_model.predict(model_input, verbose=0)
        prediction = life_target_scaler.inverse_transform(scaled_prediction)
        predicted_value = round(float(prediction[0][0]), 2)

        return JsonResponse({
            "predicted_life_expectancy": predicted_value,
            "features_received": dict(zip(FEATURE_NAMES, input_data))
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@api_view(["POST"])
def predict_water_share(request):
    """API endpoint for water share prediction"""
    try:
        if None in (water_model, water_feature_scaler, water_target_scaler):
            return JsonResponse({"error": "Water share model or scalers not loaded"}, status=500)

        # Validate input
        input_data = request.data.get("features")
        if not input_data or len(input_data) != len(ENV_FEATURE_NAMES):
            return JsonResponse({"error": f"Expected {len(ENV_FEATURE_NAMES)} features"}, status=400)

        # Ensure all features are numbers
        try:
            input_data = [float(value) for value in input_data]
        except ValueError:
            return JsonResponse({"error": "All features must be numeric values"}, status=400)

        # Preprocess and predict
        model_input = preprocess_input(input_data, water_feature_scaler)
        scaled_prediction = water_model.predict(model_input, verbose=0)
        prediction = water_target_scaler.inverse_transform(scaled_prediction)
        predicted_value = round(float(prediction[0][0]), 2)

        return JsonResponse({
            "predicted_water_share": predicted_value,
            "features_received": dict(zip(ENV_FEATURE_NAMES, input_data))
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@api_view(["POST"])
def simulate_life_expectancy(request):
    """API endpoint for life expectancy simulation"""
    try:
        if None in (life_model, life_feature_scaler, life_target_scaler):
            return JsonResponse({"error": "Life expectancy model or scalers not loaded"}, status=500)

        # Extract and validate simulation parameters
        initial_features = request.data.get('initial_features')
        years = int(request.data.get('years', 10))
        simulation_type = request.data.get('simulation_type', 'linear')
        interval = int(request.data.get('interval', 1))
        change_rates = request.data.get('change_rates', {})

        # Validate inputs
        if not isinstance(initial_features, list) or len(initial_features) != len(FEATURE_NAMES):
            return JsonResponse(
                {"error": f"Expected {len(FEATURE_NAMES)} initial features"}, 
                status=400
            )

        # Initialize simulation handler
        handler = SimulationHandler(ModelType.LIFE_EXPECTANCY)
        
        # Run simulation
        results = handler.run_simulation(
            initial_features=initial_features,
            years=years,
            simulation_type=simulation_type,
            change_rates=change_rates,
            interval=interval
        )

        return JsonResponse({
            "simulation_results": results,
            "metadata": {
                "years": years,
                "interval": interval,
                "simulation_type": simulation_type,
                "feature_names": FEATURE_NAMES
            }
        })

    except ValueError as ve:
        return JsonResponse({"error": str(ve)}, status=400)
    except Exception as e:
        print(f"Simulation error: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)

@api_view(["POST"])
def simulate_water_share(request):
    """
    API endpoint for water share simulation
    
    This endpoint handles simulations of agricultural water share over time based on 
    various input parameters and change scenarios. It uses the same simulation logic
    as life expectancy but with the water share model and its corresponding scalers.
    """
    try:
        if None in (water_model, water_feature_scaler, water_target_scaler):
            return JsonResponse({"error": "Water share model or scalers not loaded"}, status=500)

        # Extract and validate simulation parameters
        initial_features = request.data.get('initial_features')
        years = int(request.data.get('years', 10))
        simulation_type = request.data.get('simulation_type', 'linear')
        interval = int(request.data.get('interval', 1))
        change_rates = request.data.get('change_rates', {})

        # Input validation
        if not isinstance(initial_features, list) or len(initial_features) != len(ENV_FEATURE_NAMES):
            return JsonResponse(
                {"error": f"Expected {len(ENV_FEATURE_NAMES)} initial features"}, 
                status=400
            )

        # Validate simulation type
        if simulation_type not in [e.value for e in SimulationType]:
            return JsonResponse(
                {"error": f"Invalid simulation type. Must be one of: {', '.join(e.value for e in SimulationType)}"}, 
                status=400
            )

        # Validate numeric inputs
        try:
            initial_features = [float(x) for x in initial_features]
            for feature, rate in change_rates.items():
                float(rate)  # Validate change rates are numeric
        except (TypeError, ValueError):
            return JsonResponse(
                {"error": "All features and change rates must be numeric values"}, 
                status=400
            )

        # Initialize simulation handler for water share
        handler = SimulationHandler(ModelType.WATER_SHARE)
        
        # Run simulation
        results = handler.run_simulation(
            initial_features=initial_features,
            years=years,
            simulation_type=simulation_type,
            change_rates=change_rates,
            interval=interval
        )

        return JsonResponse({
            "simulation_results": results,
            "metadata": {
                "years": years,
                "interval": interval,
                "simulation_type": simulation_type,
                "feature_names": ENV_FEATURE_NAMES
            }
        })

    except ValueError as ve:
        return JsonResponse({"error": str(ve)}, status=400)
    except Exception as e:
        print(f"Simulation error: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)

@api_view(['POST'])
def simulate(request):
    """
    Generic simulation endpoint that can handle both life expectancy and water share simulations
    
    This endpoint serves as a unified interface for running simulations with either model.
    It determines which model to use based on the 'model_type' parameter in the request.
    """
    try:
        data = request.data
        model_type_str = data.get('model_type', 'life_expectancy')
        
        # Validate and convert model type
        try:
            model_type = ModelType(model_type_str)
        except ValueError:
            return JsonResponse(
                {"error": f"Invalid model type. Must be one of: {', '.join(e.value for e in ModelType)}"}, 
                status=400
            )

        # Extract common simulation parameters
        baseline_year = int(data.get('baseline_year', 2024))
        years = int(data.get('years', 10))
        interval = int(data.get('interval', 1))
        simulation_type = data.get('simulation_type', 'linear')
        initial_features = data.get('initial_features', [])
        change_rates = data.get('change_rates', {})

        # Validate simulation parameters
        if simulation_type not in [e.value for e in SimulationType]:
            return JsonResponse(
                {"error": f"Invalid simulation type. Must be one of: {', '.join(e.value for e in SimulationType)}"}, 
                status=400
            )

        # Initialize appropriate simulation handler
        handler = SimulationHandler(model_type)
        
        # Validate feature count
        if len(initial_features) != len(handler.feature_names):
            return JsonResponse(
                {"error": f"Expected {len(handler.feature_names)} features, got {len(initial_features)}"}, 
                status=400
            )

        # Run simulation
        results = handler.run_simulation(
            initial_features=initial_features,
            years=years,
            simulation_type=simulation_type,
            change_rates=change_rates,
            interval=interval
        )
        
        # Adjust years in results to use baseline_year
        for result in results:
            result['year'] = baseline_year + result['year']
        
        return JsonResponse({
            "simulation_results": results,
            "metadata": {
                "model_type": model_type.value,
                "baseline_year": baseline_year,
                "years": years,
                "interval": interval,
                "simulation_type": simulation_type,
                "feature_names": handler.feature_names
            }
        })

    except ValueError as ve:
        return JsonResponse({"error": str(ve)}, status=400)
    except Exception as e:
        print(f"Simulation error: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)

@api_view(['GET'])
def health_check(request):
    """API health check endpoint"""
    return JsonResponse({
        'status': 'healthy',
        'message': 'API is running'
    })