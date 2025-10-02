#!/usr/bin/env python3
"""
Test script to verify that "Balanced NPK (maintenance)" and "—" fertilizers 
always show quantity 0, regardless of whether they appear together or separately.
"""

import sys
import os

# Add the new ML model directory to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_ML_MODEL_DIR = os.path.join(BASE_DIR, "My new ml model")
sys.path.insert(0, NEW_ML_MODEL_DIR)

try:
    from llm import generate_recommendation_report
    from predictor import load_default
    print("✅ Successfully imported ML model and LLM modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

def test_scenario(scenario_name, predictions, base_inputs):
    """Test a specific fertilizer prediction scenario"""
    print(f"\n🧪 Testing Scenario: {scenario_name}")
    print(f"   Primary Fertilizer: {predictions.get('Primary_Fertilizer', 'None')}")
    print(f"   Secondary Fertilizer: {predictions.get('Secondary_Fertilizer', 'None')}")
    
    # Mock confidences
    confidences = {key: 0.85 for key in predictions.keys()}
    
    try:
        report = generate_recommendation_report(
            base_inputs=base_inputs,
            predictions=predictions,
            confidences=confidences,
            use_gemini_for_text=False
        )
        
        primary_name = report['primary_fertilizer']['name']
        primary_amount = report['primary_fertilizer']['amount_kg']
        secondary_name = report['secondary_fertilizer']['name']
        secondary_amount = report['secondary_fertilizer']['amount_kg']
        
        print(f"   Result - Primary: {primary_name} ({primary_amount} kg)")
        print(f"   Result - Secondary: {secondary_name} ({secondary_amount} kg)")
        
        # Verify the requirements
        success = True
        
        if primary_name == "Balanced NPK (maintenance)" and primary_amount != 0:
            print(f"   ❌ FAIL: Primary 'Balanced NPK (maintenance)' should have 0 quantity, got {primary_amount}")
            success = False
        
        if secondary_name == "—" and secondary_amount != 0:
            print(f"   ❌ FAIL: Secondary '—' should have 0 quantity, got {secondary_amount}")
            success = False
        
        if success:
            print(f"   ✅ PASS: Quantities are correct for this scenario")
        
        return success
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def main():
    """Run all test scenarios"""
    print("🔬 Testing Balanced NPK and '—' Quantity Fix")
    print("=" * 60)
    
    # Base inputs for all tests
    base_inputs = {
        "Temperature": 25.0,
        "Humidity": 80.0,
        "Moisture": 30.0,
        "Soil_Type": "Clay",
        "Crop": "Rice",
        "Nitrogen": 85.0,
        "Phosphorus": 35.0,
        "Potassium": 45.0,
        "pH": 6.5,
        "Field_Size": 1.0,
        "Field_Unit": "hectares",
        "Sowing_Date": "2024-01-15"
    }
    
    # Test scenarios
    scenarios = [
        {
            "name": "Both Balanced NPK and —",
            "predictions": {
                "Primary_Fertilizer": "Balanced NPK (maintenance)",
                "Secondary_Fertilizer": "—",
                "N_Status": "Optimal",
                "P_Status": "Optimal",
                "K_Status": "Optimal"
            }
        },
        {
            "name": "Only Balanced NPK (maintenance) primary",
            "predictions": {
                "Primary_Fertilizer": "Balanced NPK (maintenance)",
                "Secondary_Fertilizer": "MOP",
                "N_Status": "Optimal",
                "P_Status": "Low",
                "K_Status": "Low"
            }
        },
        {
            "name": "Only — secondary",
            "predictions": {
                "Primary_Fertilizer": "Urea",
                "Secondary_Fertilizer": "—",
                "N_Status": "Low",
                "P_Status": "Optimal",
                "K_Status": "Optimal"
            }
        },
        {
            "name": "Normal fertilizers (no special cases)",
            "predictions": {
                "Primary_Fertilizer": "DAP",
                "Secondary_Fertilizer": "MOP",
                "N_Status": "Low",
                "P_Status": "Low",
                "K_Status": "Low"
            }
        },
        {
            "name": "No fertilizers recommended",
            "predictions": {
                "Primary_Fertilizer": None,
                "Secondary_Fertilizer": None,
                "N_Status": "Optimal",
                "P_Status": "Optimal",
                "K_Status": "Optimal"
            }
        }
    ]
    
    results = []
    for scenario in scenarios:
        success = test_scenario(scenario["name"], scenario["predictions"], base_inputs)
        results.append(success)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    passed = sum(results)
    total = len(results)
    
    print(f"   Passed: {passed}/{total}")
    print(f"   Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("   🎉 ALL TESTS PASSED!")
        return True
    else:
        print("   ⚠️  Some tests failed - please check the implementation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
