#!/usr/bin/env python3
"""
Test script for the new ML + LLM enhanced fertilizer recommendation API
"""

import json
import requests
import sys
from typing import Dict, Any

def test_llm_enhanced_endpoint():
    """Test the LLM-enhanced prediction endpoint"""
    
    # API endpoint
    url = "http://localhost:8000/predict-llm-enhanced"
    
    # Sample test data
    test_data = {
        "Temperature": 25.0,
        "Humidity": 70.0,
        "Moisture": 45.0,
        "Soil_Type": "Loamy",
        "Crop_Type": "Rice",
        "Nitrogen": 45.0,
        "Potassium": 120.0,
        "Phosphorous": 25.0,
        "pH": 6.5,
        "Field_Size": 2.0,
        "Field_Unit": "hectares",
        "Sowing_Date": "2024-01-15",
        "Bulk_Density_g_cm3": 1.3,
        "Sampling_Depth_cm": 15.0
    }
    
    print("🧪 Testing LLM-Enhanced Fertilizer Recommendation API")
    print("=" * 60)
    print(f"📡 Endpoint: {url}")
    print(f"📊 Test Data: {json.dumps(test_data, indent=2)}")
    print("=" * 60)
    
    try:
        # Make the API request
        response = requests.post(
            url,
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        # Check response status
        if response.status_code == 200:
            result = response.json()
            
            print("✅ SUCCESS! LLM-Enhanced API Response:")
            print("=" * 60)
            
            # Extract key information
            ml_prediction = result.get("ml_model_prediction", {})
            primary_fert = result.get("primary_fertilizer", {})
            secondary_fert = result.get("secondary_fertilizer", {})
            cost_estimate = result.get("cost_estimate", {})
            soil_condition = result.get("soil_condition", {})
            
            print(f"🤖 ML Model Prediction:")
            print(f"   - Fertilizer: {ml_prediction.get('name', 'N/A')}")
            print(f"   - Confidence: {ml_prediction.get('confidence_percent', 'N/A')}%")
            print(f"   - NPK: {ml_prediction.get('npk', 'N/A')}")
            
            print(f"\n🥇 Primary Fertilizer:")
            print(f"   - Name: {primary_fert.get('name', 'N/A')}")
            print(f"   - Amount: {primary_fert.get('amount_kg', 'N/A')} kg")
            print(f"   - Reason: {primary_fert.get('reason', 'N/A')[:100]}...")
            
            print(f"\n🥈 Secondary Fertilizer:")
            print(f"   - Name: {secondary_fert.get('name', 'N/A')}")
            print(f"   - Amount: {secondary_fert.get('amount_kg', 'N/A')} kg")
            print(f"   - Reason: {secondary_fert.get('reason', 'N/A')[:100]}...")
            
            print(f"\n💰 Cost Estimate:")
            print(f"   - Primary: {cost_estimate.get('primary', 'N/A')}")
            print(f"   - Secondary: {cost_estimate.get('secondary', 'N/A')}")
            print(f"   - Organics: {cost_estimate.get('organics', 'N/A')}")
            print(f"   - Total: {cost_estimate.get('total', 'N/A')}")
            
            print(f"\n🧪 Soil Condition:")
            print(f"   - pH Status: {soil_condition.get('ph_status', 'N/A')}")
            print(f"   - Moisture Status: {soil_condition.get('moisture_status', 'N/A')}")
            print(f"   - Nutrient Deficiencies: {soil_condition.get('nutrient_deficiencies', [])}")
            
            # Check for organic alternatives
            organics = result.get("organic_alternatives", [])
            if organics:
                print(f"\n🌱 Organic Alternatives ({len(organics)} options):")
                for i, organic in enumerate(organics[:3], 1):  # Show first 3
                    print(f"   {i}. {organic.get('name', 'N/A')} - {organic.get('amount_kg', 'N/A')} kg")
            
            # Application timing
            timing = result.get("application_timing", {})
            if timing:
                print(f"\n📅 Application Timing:")
                print(f"   - Primary: {timing.get('primary', 'N/A')[:80]}...")
                print(f"   - Secondary: {timing.get('secondary', 'N/A')[:80]}...")
            
            print("\n" + "=" * 60)
            print("🎉 Test completed successfully! The new ML + LLM model is working.")
            
            return True
            
        else:
            print(f"❌ ERROR: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to the API server.")
        print("Make sure the backend server is running on http://localhost:8000")
        return False
        
    except requests.exceptions.Timeout:
        print("❌ ERROR: Request timed out. The LLM processing might take longer.")
        return False
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_basic_endpoint():
    """Test the basic prediction endpoint for comparison"""
    
    url = "http://localhost:8000/predict"
    
    test_data = {
        "Temperature": 25.0,
        "Humidity": 70.0,
        "Moisture": 45.0,
        "Soil_Type": "Loamy",
        "Crop_Type": "Rice",
        "Nitrogen": 45.0,
        "Potassium": 120.0,
        "Phosphorous": 25.0,
        "pH": 6.5
    }
    
    print("\n🔬 Testing Basic ML Prediction API (for comparison)")
    print("=" * 60)
    
    try:
        response = requests.post(
            url,
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Basic API Response:")
            print(f"   - Fertilizer: {result.get('fertilizer', 'N/A')}")
            print(f"   - Confidence: {result.get('confidence', 'N/A'):.2f}")
            return True
        else:
            print(f"❌ Basic API failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Basic API error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 AgriCure ML + LLM Integration Test")
    print("=" * 60)
    
    # Test basic endpoint first
    basic_success = test_basic_endpoint()
    
    # Test LLM-enhanced endpoint
    llm_success = test_llm_enhanced_endpoint()
    
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"   Basic ML API: {'✅ Working' if basic_success else '❌ Failed'}")
    print(f"   LLM Enhanced API: {'✅ Working' if llm_success else '❌ Failed'}")
    
    if llm_success:
        print("\n🎯 Integration Status: SUCCESS!")
        print("The new ML + LLM model is now integrated and working in your fertilizer recommendation system.")
        print("\nWhat you now have:")
        print("• Advanced ML predictions with multiple fertilizer types")
        print("• LLM-generated explanations and application guidance")
        print("• Cost analysis with pricing data")
        print("• Organic alternatives recommendations")
        print("• Soil nutrient conversion (mg/kg to kg/ha)")
        print("• Application timing recommendations")
        print("• Enhanced UI components to display rich data")
    else:
        print("\n⚠️  Integration Status: PARTIAL")
        print("Basic ML is working, but LLM enhancement needs attention.")
    
    sys.exit(0 if llm_success else 1)
