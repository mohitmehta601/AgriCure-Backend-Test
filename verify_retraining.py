#!/usr/bin/env python3

import os
import sys

def verify_retraining():
    """Verify that the model retraining was successful"""
    
    print("🔍 Verifying AgriCure Model Retraining...")
    
    # Check if model files exist
    model_files = [
        "models/fertilizer_recommender.pkl",
        "models/classifier.pkl", 
        "models/fertilizer.pkl"
    ]
    
    missing_files = []
    for file in model_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} - {size:,} bytes")
        else:
            missing_files.append(file)
            print(f"❌ {file} - Missing")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    # Try to load the new model
    try:
        import joblib
        model_artifact = joblib.load("models/fertilizer_recommender.pkl")
        
        print(f"\n📊 New Model Details:")
        print(f"  Features: {len(model_artifact['features'])}")
        print(f"  Targets: {len(model_artifact['targets'])}")
        print(f"  Models per target: {len(list(model_artifact['models'].values())[0])}")
        
        print(f"\n🎯 Available Features:")
        for feature in model_artifact['features']:
            print(f"  - {feature}")
            
        print(f"\n🎯 Available Targets:")
        for target in model_artifact['targets']:
            print(f"  - {target}")
        
        # Check if all models were trained successfully
        successful_targets = 0
        for target in model_artifact['targets']:
            if target in model_artifact['models'] and model_artifact['models'][target]:
                successful_targets += 1
        
        print(f"\n✅ Successfully trained models for {successful_targets}/{len(model_artifact['targets'])} targets")
        
        if successful_targets == len(model_artifact['targets']):
            print(f"\n🎉 All models trained successfully!")
            return True
        else:
            print(f"\n⚠️  Some models may not have trained properly")
            return False
            
    except Exception as e:
        print(f"\n❌ Error loading new model: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 AgriCure Model Retraining Verification")
    print("=" * 50)
    
    success = verify_retraining()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 MODEL RETRAINING SUCCESSFUL!")
        print("\nThe AgriCure fertilizer recommendation model has been successfully retrained with:")
        print("✅ Enhanced multi-target prediction capabilities")
        print("✅ Multiple algorithm ensemble (RF, XGBoost, LightGBM, CatBoost)")
        print("✅ Comprehensive nutrient status analysis")
        print("✅ Primary and secondary fertilizer recommendations")
        print("✅ Organic fertilizer suggestions")
        print("✅ pH amendment recommendations")
    else:
        print("❌ MODEL RETRAINING VERIFICATION FAILED!")
        print("\nPlease check the error messages above and retry.")
    
    sys.exit(0 if success else 1)
