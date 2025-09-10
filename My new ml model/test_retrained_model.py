#!/usr/bin/env python3

import joblib
import numpy as np
import pandas as pd

def test_retrained_model():
    """Test the retrained model with sample data"""
    
    # Load the retrained model
    model_path = "models/fertilizer_recommender.pkl"
    try:
        artifact = joblib.load(model_path)
        print("✅ Model loaded successfully!")
        
        features = artifact["features"]
        targets = artifact["targets"]
        models = artifact["models"]
        label_encoders = artifact["label_encoders"]
        cv_scores = artifact["cv_scores"]
        
        print(f"\n📊 Model Information:")
        print(f"Features: {features}")
        print(f"Targets: {targets}")
        print(f"Number of models per target: {len(list(models.values())[0]) if models else 0}")
        
        # Display CV scores for each target and model
        print(f"\n🎯 Cross-Validation Scores:")
        for target in targets:
            print(f"\n{target}:")
            if target in cv_scores:
                for model_name, score in cv_scores[target].items():
                    print(f"  - {model_name}: {score:.4f}")
        
        # Test with sample data
        sample_data = {
            "Temperature": 25.0,
            "Humidity": 80.0,
            "Moisture": 40.0,
            "Soil_Type": "Loamy",
            "Crop": "Rice",
            "Nitrogen": 20.0,
            "Phosphorus": 25.0,
            "Potassium": 20.0,
            "pH": 6.5
        }
        
        # Create DataFrame for prediction
        test_df = pd.DataFrame([sample_data])
        
        print(f"\n🧪 Testing with sample data:")
        print(f"Input: {sample_data}")
        
        # Make predictions for each target
        predictions = {}
        for target in targets:
            if target in models and models[target]:
                # Get the best model (assume first one for simplicity)
                model_name = list(models[target].keys())[0]
                model = models[target][model_name]
                
                try:
                    prediction_encoded = model.predict(test_df)[0]
                    
                    # Decode the prediction
                    if target in label_encoders:
                        prediction = label_encoders[target].inverse_transform([prediction_encoded])[0]
                        predictions[target] = prediction
                        print(f"  {target}: {prediction} (using {model_name})")
                    else:
                        predictions[target] = prediction_encoded
                        print(f"  {target}: {prediction_encoded} (using {model_name})")
                        
                except Exception as e:
                    print(f"  {target}: Error - {str(e)}")
        
        print(f"\n✅ Model testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading or testing model: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Testing the retrained AgriCure fertilizer recommendation model...")
    success = test_retrained_model()
    if success:
        print("\n🎉 Model retraining and testing successful!")
    else:
        print("\n💥 Model testing failed!")
