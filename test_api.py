# test_api.py
import requests
import numpy as np
from typing import Dict, Any

class LifeExpectancyPredictor:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def predict(self, features: list) -> Dict[str, Any]:
        """Make a prediction using the life expectancy API"""
        url = f"{self.base_url}/api/predict/"  # Updated endpoint
        payload = {"features": features}
        
        try:
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result['predicted_life_expectancy']
                print(f"\nPredicted Life Expectancy: {prediction} years")
                
                if 'features_received' in result:
                    print("\nFeature Values (selected important features):")
                    important_features = ['FP index', 'LP index', 'Energy use', 'Renewable energy']
                    for name, value in result['features_received'].items():
                        if name in important_features:
                            print(f"{name}: {value}")
                return result
            else:
                error_message = response.json().get('error', 'Unknown error occurred')
                print(f"\nError: {error_message}")
                return {}
                    
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to the server. Ensure the server is running.")
        except Exception as e:
            print(f"Error: {str(e)}")
        return {}

def generate_test_cases():
    """Generate test cases with variations in feature values"""
    base_case = np.array([1.0, 2.0, 65.3, 45.2, 12.1, 8.5, 15.3, 10.2, 7.8, 5.4, 
                          9.2, 20.1, 2.3, 3.4, 8.9, 42.1, 11.5, 7.8, 9.4, 6.7, 
                          40.2, 10.8, 14.2, 75.5, 30.2])
    
    variations = [
        base_case,                   # Original values
        base_case * 1.2,             # 20% increase in all features
        base_case * 0.8,             # 20% decrease in all features
        np.zeros_like(base_case),    # Minimum plausible values (zeros)
        base_case * 1.5              # 50% increase in all features (maximum plausible values)
    ]
    
    return [var.tolist() for var in variations]

def main():
    predictor = LifeExpectancyPredictor()
    test_cases = generate_test_cases()
    
    print("\nTesting multiple feature sets:")
    descriptions = [
        "Base Case",
        "20% Increase",
        "20% Decrease",
        "Minimum Values",
        "Maximum Values"
    ]
    
    for i, (features, description) in enumerate(zip(test_cases, descriptions), 1):
        print(f"\nTest Case {i} - {description}:")
        predictor.predict(features)

if __name__ == "__main__":
    main()
