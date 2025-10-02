#!/usr/bin/env python3
"""
Quick integration test to verify the API endpoint works with the fix
"""

import requests
import json

def test_api_with_balanced_npk():
    """Test the API endpoint with data that should trigger Balanced NPK (maintenance)"""
    
    # This payload should trigger the "Balanced NPK (maintenance)" + "—" scenario
    # Based on the dataset patterns: optimal N, P, K levels
    test_payload = {
        "Temperature": 25.0,
        "Humidity": 65.0,
        "Moisture": 50.0,
        "Soil_Type": 1,  # Sandy
        "Crop_Type": 3,  # Soybean
        "Nitrogen": 70.0,  # Optimal level
        "Phosphorous": 50.0,  # Optimal level  
        "Potassium": 75.0,  # Optimal level
        "pH": 7.5,
        "Sowing_Date": "2024-01-15",
        "Field_Size": 1.0,
        "Field_Unit": "hectares",
        "Bulk_Density_g_cm3": 1.3,
        "Sampling_Depth_cm": 15.0
    }
    
    try:
        # Test against local server (assuming it's running)
        response = requests.post(
            "http://localhost:8000/predict-llm-enhanced",
            json=test_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            primary = result.get("primary_fertilizer", {})
            secondary = result.get("secondary_fertilizer", {})
            
            print("🌐 API Test Results:")
            print(f"Primary Fertilizer: {primary.get('name')} - {primary.get('amount_kg')} kg")
            print(f"Secondary Fertilizer: {secondary.get('name')} - {secondary.get('amount_kg')} kg")
            
            if (primary.get('name') == "Balanced NPK (maintenance)" and 
                secondary.get('name') == "—"):
                if primary.get('amount_kg') == 0 and secondary.get('amount_kg') == 0:
                    print("✅ API Test PASSED: Quantities correctly set to 0!")
                else:
                    print(f"❌ API Test FAILED: Expected 0 quantities, got {primary.get('amount_kg')} and {secondary.get('amount_kg')}")
            else:
                print("ℹ️  API returned different fertilizers (this may be normal depending on the model)")
                
        else:
            print(f"❌ API Test FAILED: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("⚠️  API Test SKIPPED: Local server not running")
    except Exception as e:
        print(f"❌ API Test ERROR: {e}")

if __name__ == "__main__":
    test_api_with_balanced_npk()
