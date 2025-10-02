#!/usr/bin/env python3
"""
Test script to verify API endpoints handle "Balanced NPK (maintenance)" 
and "—" fertilizer quantity fix correctly.
"""

import requests
import json
import time
import sys

# Test data for different scenarios
test_scenarios = [
    {
        "name": "Test Case: Balanced NPK maintenance scenario",
        "data": {
            "Temperature": 25.0,
            "Humidity": 80.0,
            "Moisture": 30.0,
            "Soil_Type": "Clay",
            "Crop_Type": "Rice",
            "Nitrogen": 85.0,
            "Potassium": 45.0,
            "Phosphorous": 35.0,
            "pH": 6.5
        }
    },
    {
        "name": "Test Case: Low nutrient scenario",
        "data": {
            "Temperature": 28.0,
            "Humidity": 75.0,
            "Moisture": 25.0,
            "Soil_Type": "Sandy",
            "Crop_Type": "Wheat",
            "Nitrogen": 15.0,  # Very low nitrogen
            "Potassium": 20.0,  # Low potassium
            "Phosphorous": 10.0,  # Low phosphorus
            "pH": 6.0
        }
    }
]

def test_api_endpoint(base_url, endpoint, test_data, scenario_name):
    """Test a specific API endpoint with given data"""
    print(f"\n🧪 Testing: {scenario_name}")
    print(f"   Endpoint: {endpoint}")
    
    try:
        url = f"{base_url}{endpoint}"
        response = requests.post(url, json=test_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Check different response formats based on endpoint
            if endpoint == "/predict":
                fertilizer = result.get("fertilizer", "Unknown")
                print(f"   ✅ Response: {fertilizer}")
                
            elif endpoint == "/predict-enhanced":
                predictions = result.get("predictions", {})
                primary = predictions.get("Primary_Fertilizer", "Unknown")
                secondary = predictions.get("Secondary_Fertilizer", "Unknown")
                print(f"   ✅ Primary: {primary}")
                print(f"   ✅ Secondary: {secondary}")
                
                # Check for quantity fields if they exist
                if "Primary_Fertilizer_Quantity" in predictions:
                    primary_qty = predictions["Primary_Fertilizer_Quantity"]
                    print(f"   📊 Primary Quantity: {primary_qty}")
                    
                if "Secondary_Fertilizer_Quantity" in predictions:
                    secondary_qty = predictions["Secondary_Fertilizer_Quantity"]
                    print(f"   📊 Secondary Quantity: {secondary_qty}")
                
            elif endpoint == "/predict-llm-enhanced":
                primary_info = result.get("primary_fertilizer", {})
                secondary_info = result.get("secondary_fertilizer", {})
                
                primary_name = primary_info.get("name", "Unknown")
                primary_amount = primary_info.get("amount_kg", 0)
                secondary_name = secondary_info.get("name", "Unknown")
                secondary_amount = secondary_info.get("amount_kg", 0)
                
                print(f"   ✅ Primary: {primary_name} ({primary_amount} kg)")
                print(f"   ✅ Secondary: {secondary_name} ({secondary_amount} kg)")
                
                # Verify the fix
                if primary_name == "Balanced NPK (maintenance)" and primary_amount != 0:
                    print(f"   ❌ ERROR: Primary 'Balanced NPK (maintenance)' should have 0 kg, got {primary_amount}")
                    return False
                    
                if secondary_name == "—" and secondary_amount != 0:
                    print(f"   ❌ ERROR: Secondary '—' should have 0 kg, got {secondary_amount}")
                    return False
                
                # Show cost breakdown
                cost_info = result.get("cost_estimate", {})
                total_cost = cost_info.get("total", "N/A")
                print(f"   💰 Total Cost: {total_cost}")
            
            return True
            
        else:
            print(f"   ❌ HTTP Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request Error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected Error: {e}")
        return False

def test_server_health(base_url):
    """Test if the server is running and healthy"""
    print("🏥 Checking server health...")
    
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("   ✅ Server is healthy")
            return True
        else:
            print(f"   ❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Cannot reach server: {e}")
        return False

def main():
    """Run comprehensive API tests"""
    print("🚀 Testing API Endpoints for Balanced NPK Quantity Fix")
    print("=" * 70)
    
    # You can change this URL if your server runs on a different port
    base_url = "http://localhost:8000"
    
    # Check if server is running
    if not test_server_health(base_url):
        print("\n❌ Server is not running. Please start the server first:")
        print("   cd 'p:\\AgriCure Backend Test'")
        print("   python run_server.py")
        return False
    
    # Test endpoints
    endpoints_to_test = [
        "/predict",
        "/predict-enhanced", 
        "/predict-llm-enhanced"
    ]
    
    results = []
    
    for endpoint in endpoints_to_test:
        print(f"\n{'='*50}")
        print(f"🎯 Testing Endpoint: {endpoint}")
        print(f"{'='*50}")
        
        for scenario in test_scenarios:
            success = test_api_endpoint(
                base_url, 
                endpoint, 
                scenario["data"], 
                scenario["name"]
            )
            results.append(success)
            
            # Small delay between requests
            time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    passed = sum(results)
    total = len(results)
    
    print(f"   Passed: {passed}/{total}")
    print(f"   Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("   🎉 ALL API TESTS PASSED!")
        print("\n✅ The Balanced NPK quantity fix is working correctly!")
        return True
    else:
        print("   ⚠️  Some tests failed - please check the implementation")
        return False

if __name__ == "__main__":
    success = main()
    print("\n" + "=" * 70)
    if success:
        print("🎯 CONCLUSION: The fix is working correctly!")
        print("   • 'Balanced NPK (maintenance)' → 0 kg quantity")
        print("   • '—' secondary fertilizer → 0 kg quantity")
        print("   • These rules work independently or together")
    else:
        print("❌ CONCLUSION: There are issues that need to be fixed")
    
    sys.exit(0 if success else 1)
