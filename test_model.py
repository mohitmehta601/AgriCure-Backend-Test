#!/usr/bin/env python3

import sys
import os
import asyncio
sys.path.append(os.path.dirname(__file__))

from main import load_and_train_model, predict_fertilizer, FertilizerInput

async def test_model():
    print("Testing Fertilizer Recommendation Model...")
    print("=" * 50)
    
    print("1. Loading and training model...")
    success = load_and_train_model()
    
    if not success:
        print("❌ Failed to load and train model")
        return False
    
    print("✅ Model loaded and trained successfully")
    
    print("\n2. Testing prediction...")
    
    test_input = FertilizerInput(
        Temperature=25.0,
        Humidity=80.0,
        Moisture=30.0,
        Soil_Type="Loamy",
        Crop_Type="rice",
        Nitrogen=85.0,
        Potassium=45.0,
        Phosphorous=35.0
    )
    
    try:
        result = await predict_fertilizer(test_input)
        print(f"✅ Prediction successful!")
        print(f"   Input: Temperature={test_input.Temperature}°C, Humidity={test_input.Humidity}%, Soil={test_input.Soil_Type}, Crop={test_input.Crop_Type}")
        print(f"   Output: Fertilizer={result.fertilizer}, Confidence={result.confidence:.4f}")
        print(f"   Model Accuracy: {result.prediction_info['accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_model())
    if success:
        print("\n🎉 All tests passed! Model is working correctly.")
    else:
        print("\n💥 Some tests failed. Please check the error messages above.")
        sys.exit(1)
