#!/usr/bin/env python3
"""
Test script to verify that when ML model predicts:
- Primary Fertilizer: "Balanced NPK (maintenance)"  
- Secondary Fertilizer: "—"
Then both quantities should be 0.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm import generate_recommendation_report

def test_balanced_npk_maintenance_fix():
    """Test that Balanced NPK (maintenance) + — results in 0 quantities"""
    
    # Mock base inputs
    base_inputs = {
        "Temperature": 25.0,
        "Humidity": 60.0,
        "Moisture": 45.0,
        "Soil_Type": 1,
        "Crop": 1,
        "Nitrogen": 50.0,
        "Phosphorus": 40.0,
        "Potassium": 60.0,
        "pH": 7.0,
        "Sowing_Date": "2024-01-01",
        "Field_Size": 1.0,
        "Field_Unit": "hectares",
        "Bulk_Density_g_cm3": 1.3,
        "Sampling_Depth_cm": 15.0
    }
    
    # Mock predictions with the specific case we want to test
    predictions = {
        "Primary_Fertilizer": "Balanced NPK (maintenance)",
        "Secondary_Fertilizer": "—",
        "N_Status": "Optimal",
        "P_Status": "Optimal", 
        "K_Status": "Optimal",
        "Organic_1": "Compost",
        "Organic_2": "Vermicompost",
        "Organic_3": "Neem cake"
    }
    
    # Mock confidences
    confidences = {
        "Primary_Fertilizer": 0.85,
        "Secondary_Fertilizer": 0.80,
        "N_Status": 0.90,
        "P_Status": 0.90,
        "K_Status": 0.90
    }
    
    # Generate the recommendation report
    report = generate_recommendation_report(
        base_inputs=base_inputs,
        predictions=predictions,
        confidences=confidences,
        use_gemini_for_text=False
    )
    
    # Check the results
    primary_fertilizer = report["primary_fertilizer"]
    secondary_fertilizer = report["secondary_fertilizer"]
    
    print("🧪 Testing Balanced NPK (maintenance) + — scenario")
    print(f"Primary Fertilizer Name: {primary_fertilizer['name']}")
    print(f"Primary Fertilizer Amount: {primary_fertilizer['amount_kg']} kg")
    print(f"Secondary Fertilizer Name: {secondary_fertilizer['name']}")
    print(f"Secondary Fertilizer Amount: {secondary_fertilizer['amount_kg']} kg")
    
    # Assertions
    assert primary_fertilizer["name"] == "Balanced NPK (maintenance)", f"Expected 'Balanced NPK (maintenance)', got '{primary_fertilizer['name']}'"
    assert secondary_fertilizer["name"] == "—", f"Expected '—', got '{secondary_fertilizer['name']}'"
    assert primary_fertilizer["amount_kg"] == 0, f"Expected primary amount to be 0, got {primary_fertilizer['amount_kg']}"
    assert secondary_fertilizer["amount_kg"] == 0, f"Expected secondary amount to be 0, got {secondary_fertilizer['amount_kg']}"
    
    print("✅ Test PASSED: Both fertilizer quantities are correctly set to 0!")
    
    return True

def test_normal_case():
    """Test that normal fertilizer predictions still work correctly"""
    
    # Mock base inputs
    base_inputs = {
        "Temperature": 25.0,
        "Humidity": 60.0,
        "Moisture": 45.0,
        "Soil_Type": 1,
        "Crop": 1,
        "Nitrogen": 30.0,  # Low nitrogen
        "Phosphorus": 40.0,
        "Potassium": 60.0,
        "pH": 7.0,
        "Sowing_Date": "2024-01-01",
        "Field_Size": 1.0,
        "Field_Unit": "hectares",
        "Bulk_Density_g_cm3": 1.3,
        "Sampling_Depth_cm": 15.0
    }
    
    # Mock predictions with normal fertilizers
    predictions = {
        "Primary_Fertilizer": "Urea",
        "Secondary_Fertilizer": "DAP",
        "N_Status": "Low",
        "P_Status": "Optimal", 
        "K_Status": "Optimal",
        "Organic_1": "Compost",
        "Organic_2": "Vermicompost",
        "Organic_3": "Neem cake"
    }
    
    # Mock confidences
    confidences = {
        "Primary_Fertilizer": 0.85,
        "Secondary_Fertilizer": 0.80,
        "N_Status": 0.90,
        "P_Status": 0.90,
        "K_Status": 0.90
    }
    
    # Generate the recommendation report
    report = generate_recommendation_report(
        base_inputs=base_inputs,
        predictions=predictions,
        confidences=confidences,
        use_gemini_for_text=False
    )
    
    # Check the results
    primary_fertilizer = report["primary_fertilizer"]
    secondary_fertilizer = report["secondary_fertilizer"]
    
    print("\n🧪 Testing Normal fertilizer scenario")
    print(f"Primary Fertilizer Name: {primary_fertilizer['name']}")
    print(f"Primary Fertilizer Amount: {primary_fertilizer['amount_kg']} kg")
    print(f"Secondary Fertilizer Name: {secondary_fertilizer['name']}")
    print(f"Secondary Fertilizer Amount: {secondary_fertilizer['amount_kg']} kg")
    
    # Assertions - normal case should have non-zero amounts
    assert primary_fertilizer["name"] == "Urea", f"Expected 'Urea', got '{primary_fertilizer['name']}'"
    assert secondary_fertilizer["name"] == "DAP", f"Expected 'DAP', got '{secondary_fertilizer['name']}'"
    assert primary_fertilizer["amount_kg"] > 0, f"Expected primary amount to be > 0, got {primary_fertilizer['amount_kg']}"
    assert secondary_fertilizer["amount_kg"] > 0, f"Expected secondary amount to be > 0, got {secondary_fertilizer['amount_kg']}"
    
    print("✅ Test PASSED: Normal fertilizer quantities are correctly calculated!")
    
    return True

if __name__ == "__main__":
    try:
        print("🚀 Running test for Balanced NPK (maintenance) fix...")
        test_balanced_npk_maintenance_fix()
        test_normal_case()
        print("\n🎉 All tests passed! The fix is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
