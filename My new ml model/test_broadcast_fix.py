#!/usr/bin/env python3
"""
Test script to verify the broadcasting error fix in the ensemble predictor.
"""

from predictor import FertilizerRecommender
import numpy as np

def test_multiple_scenarios():
    """Test multiple input scenarios to ensure the broadcasting fix works."""
    recommender = FertilizerRecommender()
    
    test_cases = [
        {
            'name': 'Rice Clay Soil',
            'data': {
                'Temperature': 25, 'Humidity': 60, 'Moisture': 40,
                'Soil_Type': 'Clay', 'Crop': 'Rice',
                'Nitrogen': 20, 'Phosphorus': 15, 'Potassium': 10, 'pH': 6.5
            }
        },
        {
            'name': 'Wheat Sandy Soil',
            'data': {
                'Temperature': 22, 'Humidity': 55, 'Moisture': 35,
                'Soil_Type': 'Sandy', 'Crop': 'Wheat',
                'Nitrogen': 30, 'Phosphorus': 20, 'Potassium': 25, 'pH': 7.0
            }
        },
        {
            'name': 'Cotton Loamy Soil',
            'data': {
                'Temperature': 28, 'Humidity': 65, 'Moisture': 45,
                'Soil_Type': 'Loamy', 'Crop': 'Cotton',
                'Nitrogen': 25, 'Phosphorus': 18, 'Potassium': 20, 'pH': 6.8
            }
        }
    ]
    
    print("=== Broadcasting Error Fix Test ===")
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            print(f"\nTest {i}: {test_case['name']}")
            predictions, confidences = recommender.predict(test_case['data'])
            
            # Check that we got predictions for all targets
            expected_targets = ['N_Status', 'P_Status', 'K_Status', 'Primary_Fertilizer', 
                              'Secondary_Fertilizer', 'Organic_1', 'Organic_2', 'Organic_3', 'pH_Amendment']
            
            for target in expected_targets:
                if target not in predictions:
                    print(f"  ❌ Missing prediction for {target}")
                    all_passed = False
                elif target not in confidences:
                    print(f"  ❌ Missing confidence for {target}")
                    all_passed = False
                else:
                    conf = confidences[target]
                    if not (0 <= conf <= 1):
                        print(f"  ❌ Invalid confidence for {target}: {conf}")
                        all_passed = False
            
            if all_passed:
                print(f"  ✅ All targets predicted successfully")
                print(f"  Primary fertilizer: {predictions['Primary_Fertilizer']} (conf: {confidences['Primary_Fertilizer']:.3f})")
                
        except Exception as e:
            print(f"  ❌ Test failed with error: {e}")
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Broadcasting error is fixed.")
    else:
        print("\n❌ Some tests failed.")
    
    return all_passed

if __name__ == "__main__":
    test_multiple_scenarios()
