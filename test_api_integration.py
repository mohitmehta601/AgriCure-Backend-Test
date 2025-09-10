#!/usr/bin/env python3
import requests
import json

# Test data
test_data = {
    "Temperature": 25.0,
    "Humidity": 80.0,
    "Moisture": 30.0,
    "Soil_Type": "Sandy",
    "Crop_Type": "Rice",
    "Nitrogen": 85.0,
    "Potassium": 45.0,
    "Phosphorous": 35.0,
    "pH": 6.5
}

print("Testing AgriCure Backend API with new ML model")
print("=" * 50)

# Test 1: Root endpoint
print("1. Testing root endpoint...")
try:
    response = requests.get("http://127.0.0.1:8002/")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Version: {data.get('version')}")
        print(f"Model loaded: {data.get('model_loaded')}")
        print(f"Model type: {data.get('model_type')}")
    print()
except Exception as e:
    print(f"Error: {e}")
    print()

# Test 2: Model info endpoint
print("2. Testing model info endpoint...")
try:
    response = requests.get("http://127.0.0.1:8002/model-info")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Features: {data.get('features')}")
        print(f"Targets: {data.get('targets')}")
        print("Available models per target:")
        for target, models in data.get('available_models', {}).items():
            print(f"  {target}: {models}")
    print()
except Exception as e:
    print(f"Error: {e}")
    print()

# Test 3: Basic prediction endpoint (backward compatible)
print("3. Testing basic prediction endpoint...")
try:
    response = requests.post("http://127.0.0.1:8002/predict", json=test_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Primary fertilizer: {data.get('fertilizer')}")
        print(f"Confidence: {data.get('confidence'):.4f}")
        print("All predictions from new model:")
        all_preds = data.get('prediction_info', {}).get('all_predictions', {})
        for target, prediction in all_preds.items():
            print(f"  {target}: {prediction}")
    else:
        print(f"Error response: {response.text}")
    print()
except Exception as e:
    print(f"Error: {e}")
    print()

# Test 4: Enhanced prediction endpoint (full new model output)
print("4. Testing enhanced prediction endpoint...")
try:
    response = requests.post("http://127.0.0.1:8002/predict-enhanced", json=test_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("All predictions:")
        for target, prediction in data.get('predictions', {}).items():
            confidence = data.get('confidences', {}).get(target, 0)
            print(f"  {target}: {prediction} (confidence: {confidence:.4f})")
    else:
        print(f"Error response: {response.text}")
    print()
except Exception as e:
    print(f"Error: {e}")
    print()

print("Testing completed!")
