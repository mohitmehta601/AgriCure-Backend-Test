#!/usr/bin/env python3
import sys
import os

# Add the new ML model directory to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_ML_MODEL_DIR = os.path.join(BASE_DIR, "My new ml model")
sys.path.insert(0, NEW_ML_MODEL_DIR)

print(f"Base directory: {BASE_DIR}")
print(f"New ML model directory: {NEW_ML_MODEL_DIR}")
print(f"Model directory exists: {os.path.exists(NEW_ML_MODEL_DIR)}")

try:
    from predictor import load_default, FertilizerRecommender
    print("✓ Successfully imported predictor modules")
    
    model = load_default()
    print("✓ Model loaded successfully")
    print(f"Features: {model.features}")
    print(f"Targets: {model.targets}")
    
    # Test prediction with sample data
    sample_data = {
        "Temperature": 25.0,
        "Humidity": 80.0,
        "Moisture": 30.0,
        "Soil_Type": "Sandy",
        "Crop": "Rice",
        "Nitrogen": 85.0,
        "Phosphorus": 35.0,
        "Potassium": 45.0,
        "pH": 6.5
    }
    
    predictions, confidences = model.predict(sample_data)
    print("✓ Prediction test successful")
    print(f"Predictions: {predictions}")
    print(f"Confidences: {confidences}")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
