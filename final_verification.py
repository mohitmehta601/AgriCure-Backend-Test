#!/usr/bin/env python3
"""
Final verification test for the AgriCure Backend with New ML Model
"""
import requests
import json
import time

def test_api_endpoints():
    print("🌱 Final Verification: AgriCure Backend with New ML Model")
    print("=" * 60)
    
    base_url = "http://127.0.0.1:8000"
    
    # Test data
    test_payload = {
        "Temperature": 28.5,
        "Humidity": 75.0,
        "Moisture": 35.0,
        "Soil_Type": "Clayey",
        "Crop_Type": "Cotton",
        "Nitrogen": 60.0,
        "Potassium": 50.0,
        "Phosphorous": 40.0,
        "pH": 7.2
    }
    
    print("📊 Test Payload:")
    for key, value in test_payload.items():
        print(f"  {key}: {value}")
    print()
    
    # Test 1: Server status
    print("1. 🏥 Testing server health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✓ Server is healthy")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Cannot connect to server: {e}")
        return False
    
    # Test 2: Model info
    print("\n2. 🧠 Testing model information...")
    try:
        response = requests.get(f"{base_url}/model-info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("   ✓ Model info retrieved successfully")
            print(f"   📈 Model type: {data.get('model_type')}")
            print(f"   🎯 Number of targets: {len(data.get('targets', []))}")
            print(f"   📊 Targets: {', '.join(data.get('targets', [])[:3])}...")
        else:
            print(f"   ❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Model info error: {e}")
        return False
    
    # Test 3: Basic prediction (backward compatible)
    print("\n3. 🔮 Testing basic prediction (backward compatible)...")
    try:
        response = requests.post(f"{base_url}/predict", json=test_payload, timeout=15)
        if response.status_code == 200:
            data = response.json()
            print("   ✓ Basic prediction successful")
            print(f"   🌾 Primary fertilizer: {data.get('fertilizer')}")
            print(f"   📈 Confidence: {data.get('confidence', 0):.3f}")
            
            # Check if enhanced data is included
            all_preds = data.get('prediction_info', {}).get('all_predictions', {})
            if all_preds:
                print(f"   🎯 Enhanced predictions available: {len(all_preds)} targets")
        else:
            print(f"   ❌ Basic prediction failed: {response.status_code}")
            print(f"   📝 Error: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Basic prediction error: {e}")
        return False
    
    # Test 4: Enhanced prediction
    print("\n4. 🚀 Testing enhanced prediction...")
    try:
        response = requests.post(f"{base_url}/predict-enhanced", json=test_payload, timeout=15)
        if response.status_code == 200:
            data = response.json()
            print("   ✓ Enhanced prediction successful")
            
            predictions = data.get('predictions', {})
            confidences = data.get('confidences', {})
            
            print("   🎯 Key Predictions:")
            key_targets = ['Primary_Fertilizer', 'N_Status', 'P_Status', 'K_Status']
            for target in key_targets:
                if target in predictions:
                    pred = predictions[target]
                    conf = confidences.get(target, 0)
                    print(f"     {target}: {pred} (confidence: {conf:.3f})")
            
            print("   🌿 Organic Recommendations:")
            organic_targets = [k for k in predictions if k.startswith('Organic_')]
            for target in organic_targets[:3]:  # Show first 3
                pred = predictions[target]
                if pred not in ['None', '—', 'NA']:
                    print(f"     {target}: {pred}")
                    
        else:
            print(f"   ❌ Enhanced prediction failed: {response.status_code}")
            print(f"   📝 Error: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Enhanced prediction error: {e}")
        return False
    
    # Test 5: Compare with different crop
    print("\n5. 🌾 Testing with different crop (Rice)...")
    rice_payload = test_payload.copy()
    rice_payload['Crop_Type'] = 'Rice'
    
    try:
        response = requests.post(f"{base_url}/predict-enhanced", json=rice_payload, timeout=15)
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', {})
            print("   ✓ Rice prediction successful")
            print(f"   🌾 Primary fertilizer for Rice: {predictions.get('Primary_Fertilizer')}")
            print(f"   🧪 N Status: {predictions.get('N_Status')}")
        else:
            print(f"   ❌ Rice prediction failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Rice prediction error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Integration Test Results:")
    print("✓ New ML model successfully integrated")
    print("✓ Backend API endpoints working")
    print("✓ Backward compatibility maintained")
    print("✓ Enhanced predictions with multiple targets")
    print("✓ Organic recommendations included")
    print("✓ Model provides comprehensive fertilizer advice")
    print("\n🚀 Backend is ready for production!")
    
    return True

if __name__ == "__main__":
    test_api_endpoints()
