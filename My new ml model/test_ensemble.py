#!/usr/bin/env python3

from predictor import FertilizerRecommender

def test_ensemble():
    recommender = FertilizerRecommender()
    
    test_record = {
        'Temperature': 25,
        'Humidity': 60,
        'Moisture': 40,
        'Soil_Type': 'Clay',
        'Crop': 'Rice',
        'Nitrogen': 20,
        'Phosphorus': 15,
        'Potassium': 10,
        'pH': 6.5
    }
    
    predictions, confidences = recommender.predict(test_record)
    
    print("=== Ensemble Prediction Test ===")
    print(f"Input: {test_record}")
    print(f"Predictions: {predictions}")
    print(f"Confidences (first 3): {dict(list(confidences.items())[:3])}")
    print("Test completed successfully!")

if __name__ == "__main__":
    test_ensemble()
