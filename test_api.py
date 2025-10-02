#!/usr/bin/env python3

import requests
import json
import time

def test_api():
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Fertilizer Recommendation API...")
    print("=" * 50)
    
    print("⏳ Waiting for server to start...")
    time.sleep(2)
    
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed!")
            print(f"   Status: {data['status']}")
            print(f"   Model loaded: {data['model_loaded']}")
            print(f"   Model accuracy: {data['model_accuracy']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure it's running on port 8000")
        return False
    
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint working!")
            print(f"   Message: {data['message']}")
            print(f"   Version: {data['version']}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    print("\n3. Testing model info...")
    try:
        response = requests.get(f"{base_url}/model-info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved!")
            print(f"   Model type: {data['model_type']}")
            print(f"   Accuracy: {data['accuracy']}")
            print(f"   Available fertilizers: {len(data['available_fertilizers'])}")
            print(f"   Available soil types: {len(data['available_soil_types'])}")
            print(f"   Available crop types: {len(data['available_crop_types'])}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model info error: {e}")
    
    print("\n4. Testing prediction endpoint...")
    test_input = {
        "Temperature": 25.0,
        "Humidity": 80.0,
        "Moisture": 30.0,
        "Soil_Type": "Loamy",
        "Crop_Type": "rice",
        "Nitrogen": 85.0,
        "Potassium": 45.0,
        "Phosphorous": 35.0
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            headers={"Content-Type": "application/json"},
            json=test_input
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful!")
            print(f"   Input: Temperature={test_input['Temperature']}°C, Humidity={test_input['Humidity']}%")
            print(f"   Output: Fertilizer={data['fertilizer']}")
            print(f"   Confidence: {data['confidence']:.4f}")
            print(f"   Model accuracy: {data['prediction_info']['accuracy']:.4f}")
        else:
            error_data = response.json()
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {error_data.get('detail', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Prediction error: {e}")
    
    print("\n5. Testing input validation...")
    invalid_input = {
        "Temperature": 100.0,
        "Humidity": 80.0,
        "Moisture": 30.0,
        "Soil_Type": "Loamy",
        "Crop_Type": "rice",
        "Nitrogen": 85.0,
        "Potassium": 45.0,
        "Phosphorous": 35.0
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            headers={"Content-Type": "application/json"},
            json=invalid_input
        )
        
        if response.status_code == 400:
            error_data = response.json()
            print(f"✅ Input validation working!")
            print(f"   Rejected invalid temperature: {error_data['detail']}")
        else:
            print(f"❌ Input validation failed: Expected 400, got {response.status_code}")
    except Exception as e:
        print(f"❌ Input validation error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API testing completed!")
    print(f"📖 Interactive API docs available at: {base_url}/docs")
    print(f"🔗 API base URL: {base_url}")
    
    return True

if __name__ == "__main__":
    success = test_api()
    if not success:
        print("\n💥 Some tests failed. Please check the server is running.")
        print("Start the server with: python run_server.py")
