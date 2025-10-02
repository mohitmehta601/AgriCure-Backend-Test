#!/usr/bin/env python3
"""
Test the ML → LLM pipeline
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predictor import load_default
from llm import generate_recommendation_report

def test_ml_to_llm_pipeline():
    print("🔍 Testing ML → LLM Pipeline")
    print("=" * 50)
    
    # Load the trained model
    print("1. Loading ML model...")
    model = load_default()
    print("   ✅ Model loaded successfully")
    
    # Test features (sample soil and crop data)
    test_features = {
        'Temperature': 25.0,
        'Humidity': 60.0,
        'Moisture': 40.0,
        'Soil_Type': 'Loamy',
        'Crop': 'Wheat',
        'Nitrogen': 20.0,
        'Phosphorus': 15.0,
        'Potassium': 25.0,
        'pH': 6.5
    }
    
    print("\n2. Input Features:")
    for key, value in test_features.items():
        print(f"   {key}: {value}")
    
    # Get ML predictions
    print("\n3. Running ML prediction...")
    preds, confs = model.predict(test_features)
    print("   ✅ ML predictions generated")
    print(f"   Predictions: {preds}")
    print(f"   Confidences: {confs}")
    
    # Prepare inputs for LLM
    base_inputs = {
        **test_features,
        'Field_Size': 1.0,
        'Sowing_Date': '2024-01-01',
        'Field_Unit': 'hectares'
    }
    
    # Generate LLM-enhanced report
    print("\n4. Generating LLM-enhanced report...")
    try:
        report = generate_recommendation_report(base_inputs, preds, confs)
        print("   ✅ LLM report generated successfully!")
        
        # Display key results
        print("\n5. Final Results:")
        print(f"   Primary Fertilizer: {report['primary_fertilizer']['name']}")
        print(f"   Amount: {report['primary_fertilizer']['amount_kg']} kg")
        print(f"   Confidence: {report['ml_model_prediction']['confidence_percent']}%")
        print(f"   Cost: {report['cost_estimate']['total']}")
        print(f"   Reason: {report['primary_fertilizer']['reason'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"   ❌ LLM generation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_ml_to_llm_pipeline()
    print("\n" + "=" * 50)
    if success:
        print("🎉 ML → LLM Pipeline Test: PASSED")
    else:
        print("❌ ML → LLM Pipeline Test: FAILED")
