from django.test import TestCase, Client
from django.urls import reverse
import numpy as np
import json
from unittest.mock import patch

class LifeExpectancyPredictionTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.predict_url = reverse('predict_life_expectancy')
        
        # Sample valid input features (25 features as per FEATURE_NAMES)
        self.valid_features = [
            1.0, 2.0, 65.3, 45.2, 12.1, 8.5, 15.3, 10.2, 
            7.8, 5.4, 9.2, 20.1, 2.3, 3.4, 8.9, 42.1, 
            11.5, 7.8, 9.4, 6.7, 40.2, 10.8, 14.2, 75.5, 30.2
        ]

    @patch('api.views.model.predict')
    @patch('api.views.scaler.transform')
    def test_successful_prediction(self, mock_transform, mock_predict):
        """Test successful prediction with valid input"""
        # Setup mock returns
        mock_transform.return_value = np.array([self.valid_features]).reshape(1, 1, 25)
        mock_predict.return_value = np.array([[75.5]])

        payload = {
            "features": self.valid_features
        }
        response = self.client.post(
            self.predict_url,
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        result = response.json()
        print("\nTest Successful Prediction Results:")
        print(f"Status Code: {response.status_code}")
        print(f"Predicted Life Expectancy: {result['predicted_life_expectancy']} years")
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue('predicted_life_expectancy' in result)
        prediction = result['predicted_life_expectancy']
        self.assertTrue(isinstance(prediction, float))
        self.assertEqual(prediction, 75.5)

    def test_missing_features(self):
        """Test error handling when features are missing"""
        payload = {}
        response = self.client.post(
            self.predict_url,
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()['error'],
            "Missing 'features' in request body"
        )

    def test_incorrect_feature_count(self):
        """Test error handling when wrong number of features is provided"""
        payload = {
            "features": [1.0, 2.0, 3.0]  # Only 3 features instead of 25
        }
        response = self.client.post(
            self.predict_url,
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()['error'],
            "Expected 25 features, but got 3"
        )

    @patch('api.views.model.predict')
    @patch('api.views.scaler.transform')
    def test_invalid_feature_values(self, mock_transform, mock_predict):
        """Test error handling when invalid feature values are provided"""
        mock_transform.side_effect = ValueError("Invalid input")
        
        payload = {
            "features": ["invalid"] * 25  # Invalid string values instead of numbers
        }
        response = self.client.post(
            self.predict_url,
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 500)

    @patch('api.views.model.predict')
    @patch('api.views.scaler.transform')
    def test_successful_prediction_with_mocks(self, mock_transform, mock_predict):
        """Test successful prediction with mocked model and scaler"""
        # Setup mock returns
        mock_transform.return_value = np.array([self.valid_features]).reshape(1, 1, 25)
        mock_predict.return_value = np.array([[75.5]])  # Mock prediction value
        
        payload = {
            "features": self.valid_features
        }
        response = self.client.post(
            self.predict_url,
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        result = response.json()
        print("\nTest Successful Prediction with Mocks Results:")
        print(f"Status Code: {response.status_code}")
        print(f"Predicted Life Expectancy: {result['predicted_life_expectancy']} years")
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(result['predicted_life_expectancy'], 75.5)
