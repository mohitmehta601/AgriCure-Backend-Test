import requests
import json

# Test the new ML + LLM integration
def test_integration():
    base_url = "http://localhost:8000"
    
    # Test data
    test_data = {
        "Temperature": 25.0,
        "Humidity": 70.0,
        "Moisture": 60.0,
        "Soil_Type": "Loamy",
        "Crop_Type": "Rice",
        "Nitrogen": 45.0,
        "Potassium": 60.0,
        "Phosphorous": 35.0,
        "pH": 6.5,
        "Field_Size": 1.0,
        "Field_Unit": "hectares",
        "Sowing_Date": "2024-01-15"
    }
    
    print("🧪 Testing ML + LLM Integration")
    print("=" * 50)
    
    # Test 1: Basic health check
    print("1. Testing API health...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"   ✅ Status: {response.status_code}")
        data = response.json()
        print(f"   📊 Model loaded: {data.get('model_loaded')}")
        print(f"   🤖 LLM available: {data.get('llm_available')}")
        print(f"   📝 Version: {data.get('version')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test 2: Enhanced prediction
    print("\n2. Testing enhanced ML prediction...")
    try:
        response = requests.post(f"{base_url}/predict-enhanced", json=test_data)
        print(f"   ✅ Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   🎯 Predictions: {list(data.get('predictions', {}).keys())}")
            print(f"   🔢 Primary Fertilizer: {data.get('predictions', {}).get('Primary_Fertilizer')}")
            print(f"   📈 Confidence: {data.get('confidences', {}).get('Primary_Fertilizer', 0):.2f}")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: LLM-enhanced prediction
    print("\n3. Testing LLM-enhanced prediction...")
    try:
        response = requests.post(f"{base_url}/predict-llm-enhanced", json=test_data)
        print(f"   ✅ Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   🧠 ML Prediction: {data.get('ml_model_prediction', {}).get('name')}")
            print(f"   🥇 Primary: {data.get('primary_fertilizer', {}).get('name')} ({data.get('primary_fertilizer', {}).get('amount_kg')}kg)")
            print(f"   🥈 Secondary: {data.get('secondary_fertilizer', {}).get('name')} ({data.get('secondary_fertilizer', {}).get('amount_kg')}kg)")
            print(f"   💰 Total Cost: {data.get('cost_estimate', {}).get('total')}")
            print(f"   🌱 Organic Options: {len(data.get('organic_alternatives', []))}")
            print(f"   📊 Confidence: {data.get('ml_model_prediction', {}).get('confidence_percent')}%")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n🎉 Integration test completed!")

if __name__ == "__main__":
    test_integration()
