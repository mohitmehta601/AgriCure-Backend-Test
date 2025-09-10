#!/usr/bin/env python3
"""
Simple test to verify the balanced NPK fix works directly with the LLM module
"""

import sys
import os

# Add the new ML model directory to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_ML_MODEL_DIR = os.path.join(BASE_DIR, "My new ml model")
sys.path.insert(0, NEW_ML_MODEL_DIR)

from llm import generate_recommendation_report

def test_balanced_npk_fix():
    """Test the specific scenarios mentioned in the requirements"""
    
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
        "Field_Unit": "hectares"
    }
    
    # Test scenarios
    tests = [
        {
            "name": "Balanced NPK (maintenance) as Primary",
            "predictions": {
                "Primary_Fertilizer": "Balanced NPK (maintenance)",
                "Secondary_Fertilizer": "DAP",
                "N_Status": "Optimal",
                "P_Status": "Low",
                "K_Status": "Optimal"
            },
            "expect_primary_qty": 0,
            "expect_secondary_qty": ">0"
        },
        {
            "name": "— as Secondary",
            "predictions": {
                "Primary_Fertilizer": "Urea",
                "Secondary_Fertilizer": "—",
                "N_Status": "Low",
                "P_Status": "Optimal", 
                "K_Status": "Optimal"
            },
            "expect_primary_qty": ">0",
            "expect_secondary_qty": 0
        },
        {
            "name": "Both Balanced NPK and —",
            "predictions": {
                "Primary_Fertilizer": "Balanced NPK (maintenance)",
                "Secondary_Fertilizer": "—",
                "N_Status": "Optimal",
                "P_Status": "Optimal",
                "K_Status": "Optimal"
            },
            "expect_primary_qty": 0,
            "expect_secondary_qty": 0
        }
    ]
    
    print("🧪 Testing Balanced NPK and — Quantity Fix")
    print("=" * 60)
    
    all_passed = True
    
    for test in tests:
        print(f"\n📋 Test: {test['name']}")
        print(f"   Primary: {test['predictions']['Primary_Fertilizer']}")
        print(f"   Secondary: {test['predictions']['Secondary_Fertilizer']}")
        
        # Mock confidences
        confidences = {key: 0.85 for key in test['predictions'].keys()}
        
        try:
            report = generate_recommendation_report(
                base_inputs=base_inputs,
                predictions=test['predictions'],
                confidences=confidences,
                use_gemini_for_text=False
            )
            
            primary_amount = report['primary_fertilizer']['amount_kg']
            secondary_amount = report['secondary_fertilizer']['amount_kg']
            
            print(f"   → Primary quantity: {primary_amount} kg")
            print(f"   → Secondary quantity: {secondary_amount} kg")
            
            # Check expectations
            test_passed = True
            
            if test['expect_primary_qty'] == 0 and primary_amount != 0:
                print(f"   ❌ FAIL: Expected primary quantity 0, got {primary_amount}")
                test_passed = False
            elif test['expect_primary_qty'] == ">0" and primary_amount == 0:
                print(f"   ❌ FAIL: Expected primary quantity >0, got {primary_amount}")
                test_passed = False
                
            if test['expect_secondary_qty'] == 0 and secondary_amount != 0:
                print(f"   ❌ FAIL: Expected secondary quantity 0, got {secondary_amount}")
                test_passed = False
            elif test['expect_secondary_qty'] == ">0" and secondary_amount == 0:
                print(f"   ❌ FAIL: Expected secondary quantity >0, got {secondary_amount}")
                test_passed = False
            
            if test_passed:
                print(f"   ✅ PASS")
            else:
                all_passed = False
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    print("📊 FINAL RESULT")
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ The fix is working correctly:")
        print("   • 'Balanced NPK (maintenance)' → Always 0 kg")
        print("   • '—' → Always 0 kg") 
        print("   • These work independently")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = test_balanced_npk_fix()
    sys.exit(0 if success else 1)
