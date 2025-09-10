#!/usr/bin/env python3
"""
Startup validation script to check if all required files and dependencies are available
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_required_files():
    """Check if all required files exist"""
    required_files = [
        "main.py",
        "soil_api.py", 
        "start_railway.py",
        "requirements.txt",
        "ml/f2.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    logger.info("✅ All required files found")
    return True

def check_dataset():
    """Check if the dataset is valid"""
    try:
        import pandas as pd
        
        dataset_path = "ml/f2.csv"
        data = pd.read_csv(dataset_path)
        
        logger.info(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        
        required_columns = ['Temparature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type', 
                           'Nitrogen', 'Potassium', 'Phosphorous', 'Fertilizer']
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing columns in dataset: {missing_columns}")
            return False
        
        if len(data) == 0:
            logger.error("Dataset is empty")
            return False
            
        logger.info("✅ Dataset validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return False

def check_dependencies():
    """Check if all dependencies can be imported"""
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        import sklearn
        import requests
        
        logger.info("✅ All dependencies can be imported")
        return True
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def main():
    """Run all validation checks"""
    logger.info("Starting startup validation...")
    
    checks = [
        ("Required files", check_required_files),
        ("Dataset", check_dataset),
        ("Dependencies", check_dependencies)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        logger.info(f"Running {check_name} check...")
        if not check_func():
            logger.error(f"❌ {check_name} check failed")
            all_passed = False
        else:
            logger.info(f"✅ {check_name} check passed")
    
    if all_passed:
        logger.info("🎉 All validation checks passed! App should start successfully.")
        return 0
    else:
        logger.error("❌ Some validation checks failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
