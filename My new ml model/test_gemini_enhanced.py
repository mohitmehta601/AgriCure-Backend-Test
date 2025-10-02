#!/usr/bin/env python3
"""
Test ML → Gemini LLM Pipeline with API key
"""
from predictor import load_default
from llm import generate_recommendation_report

def test_with_gemini():
    print("🔍 Testing ML → Gemini LLM Pipeline")
    print("=" * 50)
    
    # Load model and make prediction
    model = load_default()
    test_features = {
        'Temperature': 28.0,
        'Humidity': 65.0, 
        'Moisture': 35.0,
        'Soil_Type': 'Sandy',
        'Crop': 'Rice',
        'Nitrogen': 15.0,
        'Phosphorus': 10.0,
        'Potassium': 20.0,
        'pH': 7.2
    }
    
    print("Input:", test_features)
    preds, confs = model.predict(test_features)
    print(f"ML Predictions: {preds}")
    
    # Generate report with Gemini enhancement
    base_inputs = {**test_features, 'Field_Size': 2.0, 'Sowing_Date': '2024-06-01'}
    
    print("\nGenerating Gemini-enhanced report...")
    report = generate_recommendation_report(base_inputs, preds, confs, use_gemini_for_text=True)
    
    print("\n🎯 Enhanced Report Generated!")
    print(f"Primary: {report['primary_fertilizer']['name']}")
    print(f"Reason: {report['primary_fertilizer']['reason']}")
    print(f"Secondary: {report['secondary_fertilizer']['name']}")
    print(f"Amount: {report['primary_fertilizer']['amount_kg']} kg for {base_inputs['Field_Size']} hectares")
    print(f"Confidence: {report['ml_model_prediction']['confidence_percent']}%")

if __name__ == "__main__":
    test_with_gemini()
