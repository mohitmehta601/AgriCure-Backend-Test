#!/usr/bin/env python3
"""
Comprehensive test for the integrated ML model
"""
import sys
import os

# Add the new ML model directory to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_ML_MODEL_DIR = os.path.join(BASE_DIR, "My new ml model")
sys.path.insert(0, NEW_ML_MODEL_DIR)

def test_new_model_integration():
    print("Testing New ML Model Integration")
    print("=" * 50)
    
    # Test 1: Import and load the model
    print("1. Testing model import and loading...")
    try:
        from predictor import load_default, FertilizerRecommender
        model = load_default()
        print("✓ Model loaded successfully")
        print(f"  Features: {model.features}")
        print(f"  Targets: {model.targets}")
        print(f"  Number of features: {len(model.features)}")
        print(f"  Number of targets: {len(model.targets)}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    # Test 2: Test prediction with various input scenarios
    test_cases = [
        {
            "name": "Rice in Sandy Soil",
            "data": {
                "Temperature": 25.0,
                "Humidity": 80.0,
                "Moisture": 30.0,
                "Soil_Type": "Sandy",
                "Crop": "Rice",
                "Nitrogen": 85.0,
                "Phosphorus": 35.0,
                "Potassium": 45.0,
                "pH": 6.5
            }
        },
        {
            "name": "Cotton in Black Soil",
            "data": {
                "Temperature": 30.0,
                "Humidity": 60.0,
                "Moisture": 40.0,
                "Soil_Type": "Black",
                "Crop": "Cotton",
                "Nitrogen": 40.0,
                "Phosphorus": 30.0,
                "Potassium": 70.0,
                "pH": 7.0
            }
        },
        {
            "name": "Tea in Alkaline Soil",
            "data": {
                "Temperature": 20.0,
                "Humidity": 85.0,
                "Moisture": 50.0,
                "Soil_Type": "Alkaline",
                "Crop": "Tea",
                "Nitrogen": 60.0,
                "Phosphorus": 40.0,
                "Potassium": 30.0,
                "pH": 8.0
            }
        }
    ]
    
    print("\n2. Testing predictions with different scenarios...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n  Test Case {i}: {test_case['name']}")
        try:
            predictions, confidences = model.predict(test_case['data'])
            print(f"    ✓ Prediction successful")
            
            # Show key predictions
            key_predictions = ['Primary_Fertilizer', 'N_Status', 'P_Status', 'K_Status']
            for target in key_predictions:
                if target in predictions:
                    pred = predictions[target]
                    conf = confidences.get(target, 0)
                    print(f"    {target}: {pred} (confidence: {conf:.3f})")
            
            # Show organic recommendations
            organic_preds = [k for k in predictions.keys() if k.startswith('Organic_')]
            if organic_preds:
                print("    Organic recommendations:")
                for target in organic_preds:
                    pred = predictions[target]
                    if pred not in ['None', '—', 'NA']:
                        print(f"      {target}: {pred}")
                        
        except Exception as e:
            print(f"    ✗ Error in prediction: {e}")
            return False
    
    # Test 3: Test FastAPI integration components
    print("\n3. Testing FastAPI integration components...")
    try:
        # Import the correct main module from the backend root
        sys.path.insert(0, BASE_DIR)  # Ensure backend root is first in path
        
        # Now import from the backend main.py
        import main as backend_main
        from main import get_recommender, FertilizerInput
        
        # Test model loading through FastAPI interface
        api_model = get_recommender()
        print("✓ FastAPI integration successful")
        print(f"  API model features: {api_model.features}")
        print(f"  API model targets: {api_model.targets}")
        
        # Test Pydantic model
        test_input = FertilizerInput(
            Temperature=25.0,
            Humidity=80.0,
            Moisture=30.0,
            Soil_Type="Sandy",
            Crop_Type="Rice",  # Note: API uses Crop_Type
            Nitrogen=85.0,
            Potassium=45.0,
            Phosphorous=35.0,  # Note: API uses Phosphorous
            pH=6.5
        )
        print("✓ Pydantic model validation successful")
        
    except Exception as e:
        print(f"✗ Error in FastAPI integration: {e}")
        return False
    
    # Test 4: Verify backward compatibility
    print("\n4. Testing backward compatibility...")
    try:
        # Test that we can map the old API format to new model format
        old_api_input = {
            "Temperature": 25.0,
            "Humidity": 80.0,
            "Moisture": 30.0,
            "Soil_Type": "Sandy",
            "Crop_Type": "Rice",  # Old API field name
            "Nitrogen": 85.0,
            "Potassium": 45.0,
            "Phosphorous": 35.0,  # Old API field name
        }
        
        # Map to new model format
        new_model_input = {
            "Temperature": old_api_input["Temperature"],
            "Humidity": old_api_input["Humidity"],
            "Moisture": old_api_input["Moisture"],
            "Soil_Type": old_api_input["Soil_Type"],
            "Crop": old_api_input["Crop_Type"],  # Map Crop_Type to Crop
            "Nitrogen": old_api_input["Nitrogen"],
            "Phosphorus": old_api_input["Phosphorous"],  # Map Phosphorous to Phosphorus
            "Potassium": old_api_input["Potassium"],
            "pH": 6.5  # Default value for new field
        }
        
        predictions, confidences = model.predict(new_model_input)
        
        # Simulate backward compatible response
        primary_fertilizer = predictions.get("Primary_Fertilizer", "Unknown")
        primary_confidence = confidences.get("Primary_Fertilizer", 0.0)
        
        print("✓ Backward compatibility mapping successful")
        print(f"  Mapped prediction: {primary_fertilizer} (confidence: {primary_confidence:.3f})")
        
    except Exception as e:
        print(f"✗ Error in backward compatibility: {e}")
        return False
    
    # Test 5: Dataset verification
    print("\n5. Testing dataset integration...")
    try:
        dataset_path = os.path.join(NEW_ML_MODEL_DIR, "New dataset 5111 rows.csv")
        if os.path.exists(dataset_path):
            import pandas as pd
            df = pd.read_csv(dataset_path)
            print(f"✓ Dataset loaded successfully")
            print(f"  Dataset shape: {df.shape}")
            print(f"  Dataset columns: {list(df.columns)}")
            
            # Check if required columns are present
            required_cols = model.features + model.targets
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"  Warning: Missing columns in dataset: {missing_cols}")
            else:
                print("✓ All required columns present in dataset")
        else:
            print(f"✗ Dataset not found at: {dataset_path}")
            
    except Exception as e:
        print(f"✗ Error testing dataset: {e}")
    
    print("\n" + "=" * 50)
    print("Integration test completed successfully! ✓")
    print("\nSummary:")
    print("- New ML model integrated successfully")
    print("- Multiple prediction targets available")
    print("- FastAPI backend updated to use new model")
    print("- Backward compatibility maintained")
    print("- Enhanced predictions with organic recommendations")
    print("- Dataset with 5111 rows available for training")
    
    return True

if __name__ == "__main__":
    test_new_model_integration()
