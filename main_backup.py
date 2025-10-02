from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
from typing import Dict, Any
import logging
from datetime import datetime
from soil_api import soil_data_api

# Base directories for models, data, templates, and static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
ML_DIR = os.path.join(BASE_DIR, "ml")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fertilizer Recommendation API",
    description="ML API for predicting fertilizer recommendations based on soil and crop conditions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FertilizerInput(BaseModel):
    Temperature: float
    Humidity: float
    Moisture: float
    Soil_Type: str
    Crop_Type: str
    Nitrogen: float
    Potassium: float
    Phosphorous: float

class LocationInput(BaseModel):
    latitude: float
    longitude: float

class LocationSoilResponse(BaseModel):
    location: Dict[str, Any]
    soil_type: str
    soil_properties: Dict[str, Any]
    confidence: float
    sources: list
    success: bool
    location_info: Dict[str, Any]

class FertilizerResponse(BaseModel):
    fertilizer: str
    confidence: float
    prediction_info: Dict[str, Any]

model = None
soil_encoder = None
crop_encoder = None
fertilizer_encoder = None
model_accuracy = None

def load_and_train_model():
    global model, soil_encoder, crop_encoder, fertilizer_encoder, model_accuracy
    
    try:
        dataset_path = os.path.join(ML_DIR, "f2.csv")
        logger.info(f"Loading dataset from: {dataset_path}")
        
        # Check if file exists
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found: {dataset_path}")
            return False
        
        data = pd.read_csv(dataset_path)
        logger.info(f"Dataset loaded successfully. Shape: {data.shape}")
        
        required_columns = ['Temparature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type', 
                           'Nitrogen', 'Potassium', 'Phosphorous', 'Fertilizer']
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing columns in dataset: {missing_columns}")
            return False
        
        data = data.rename(columns={'Temparature': 'Temperature'})
        
        # Initialize encoders
        soil_encoder = LabelEncoder()
        crop_encoder = LabelEncoder()
        fertilizer_encoder = LabelEncoder()
        
        # Check for empty data
        if len(data) == 0:
            logger.error("Dataset is empty")
            return False
        
        data['Soil_Type_Encoded'] = soil_encoder.fit_transform(data['Soil_Type'])
        data['Crop_Type_Encoded'] = crop_encoder.fit_transform(data['Crop_Type'])
        data['Fertilizer_Encoded'] = fertilizer_encoder.fit_transform(data['Fertilizer'])
        
        feature_columns = ['Temperature', 'Humidity', 'Moisture', 'Soil_Type_Encoded', 
                          'Crop_Type_Encoded', 'Nitrogen', 'Potassium', 'Phosphorous']
        
        X = data[feature_columns]
        y = data['Fertilizer_Encoded']
        
        # Validate data
        if X.isnull().any().any() or y.isnull().any():
            logger.error("Dataset contains null values")
            return False
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        model_accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model trained successfully. Accuracy: {model_accuracy:.4f}")
        logger.info(f"Unique fertilizers: {fertilizer_encoder.classes_.tolist()}")
        logger.info(f"Unique soil types: {soil_encoder.classes_.tolist()}")
        logger.info(f"Unique crop types: {crop_encoder.classes_.tolist()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Fertilizer Recommendation API...")
    try:
        success = load_and_train_model()
        if not success:
            logger.error("Failed to load and train model. API may not function properly.")
            # Don't fail completely, just log the error
        else:
            logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        # Don't fail completely, just log the error

@app.get("/")
async def root():
    try:
        return {
            "message": "Fertilizer Recommendation API",
            "version": "1.0.0",
            "status": "running",
            "model_loaded": model is not None,
            "model_accuracy": model_accuracy,
            "timestamp": datetime.now().isoformat(),
            "service": "fertilizer-recommendation-api"
        }
    except Exception as e:
        logger.error(f"Root endpoint error: {str(e)}")
        return {
            "message": "Fertilizer Recommendation API",
            "version": "1.0.0",
            "status": "running",
            "service": "fertilizer-recommendation-api"
        }

@app.get("/health")
async def health_check():
    """Simple health check endpoint for Railway"""
    return {"status": "healthy"}

@app.get("/status")
async def status_check():
    """Detailed status check with model information"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model is not None,
            "model_accuracy": model_accuracy,
            "service": "fertilizer-recommendation-api",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        return {
            "status": "healthy",
            "service": "fertilizer-recommendation-api",
            "version": "1.0.0"
        }

@app.get("/readiness")
async def readiness_check():
    """Readiness probe - indicates if the service is ready to serve requests"""
    try:
        return {
            "status": "ready" if model is not None else "not_ready",
            "model_loaded": model is not None,
            "model_accuracy": model_accuracy,
            "timestamp": datetime.now().isoformat()
        }
    except Exception:
        return {"status": "ready", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=FertilizerResponse)
async def predict_fertilizer(input_data: FertilizerInput):
    global model, soil_encoder, crop_encoder, fertilizer_encoder
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please try again later.")
    
    try:
        if not (0 <= input_data.Temperature <= 50):
            raise HTTPException(status_code=400, detail="Temperature must be between 0 and 50°C")
        if not (0 <= input_data.Humidity <= 100):
            raise HTTPException(status_code=400, detail="Humidity must be between 0 and 100%")
        if not (0 <= input_data.Moisture <= 100):
            raise HTTPException(status_code=400, detail="Moisture must be between 0 and 100%")
        if not (0 <= input_data.Nitrogen <= 150):
            raise HTTPException(status_code=400, detail="Nitrogen must be between 0 and 150")
        if not (0 <= input_data.Potassium <= 100):
            raise HTTPException(status_code=400, detail="Potassium must be between 0 and 100")
        if not (0 <= input_data.Phosphorous <= 100):
            raise HTTPException(status_code=400, detail="Phosphorous must be between 0 and 100")
        
        try:
            soil_encoded = soil_encoder.transform([input_data.Soil_Type])[0]
            crop_encoded = crop_encoder.transform([input_data.Crop_Type])[0]
        except ValueError as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid categorical value. {str(e)}"
            )
        
        features = np.array([
            input_data.Temperature,
            input_data.Humidity,
            input_data.Moisture,
            soil_encoded,
            crop_encoded,
            input_data.Nitrogen,
            input_data.Potassium,
            input_data.Phosphorous
        ]).reshape(1, -1)
        
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
        fertilizer_name = fertilizer_encoder.inverse_transform([prediction])[0]
        confidence = float(prediction_proba[prediction])
        
        model_info = {
            "accuracy": model_accuracy,
            "n_estimators": model.n_estimators,
            "feature_importance": {
                "Temperature": float(model.feature_importances_[0]),
                "Humidity": float(model.feature_importances_[1]),
                "Moisture": float(model.feature_importances_[2]),
                "Soil_Type": float(model.feature_importances_[3]),
                "Crop_Type": float(model.feature_importances_[4]),
                "Nitrogen": float(model.feature_importances_[5]),
                "Potassium": float(model.feature_importances_[6]),
                "Phosphorous": float(model.feature_importances_[7])
            }
        }
        
        logger.info(f"Prediction made: {fertilizer_name} with confidence {confidence:.4f}")
        
        return FertilizerResponse(
            fertilizer=fertilizer_name,
            confidence=confidence,
            prediction_info=model_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": "RandomForestClassifier",
        "accuracy": model_accuracy,
        "n_estimators": model.n_estimators,
        "feature_importance": {
            "Temperature": float(model.feature_importances_[0]),
            "Humidity": float(model.feature_importances_[1]),
            "Moisture": float(model.feature_importances_[2]),
            "Soil_Type": float(model.feature_importances_[3]),
            "Crop_Type": float(model.feature_importances_[4]),
            "Nitrogen": float(model.feature_importances_[5]),
            "Potassium": float(model.feature_importances_[6]),
            "Phosphorous": float(model.feature_importances_[7])
        },
        "available_fertilizers": fertilizer_encoder.classes_.tolist() if fertilizer_encoder else [],
        "available_soil_types": soil_encoder.classes_.tolist() if soil_encoder else [],
        "available_crop_types": crop_encoder.classes_.tolist() if crop_encoder else []
    }

@app.post("/retrain")
async def retrain_model():
    try:
        success = load_and_train_model()
        if success:
            return {"message": "Model retrained successfully", "accuracy": model_accuracy}
        else:
            raise HTTPException(status_code=500, detail="Failed to retrain model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

@app.post("/soil-data")
async def get_soil_data(location: LocationInput):
    try:
        logger.info(f"Fetching soil data for coordinates: {location.latitude}, {location.longitude}")
        
        soil_data = soil_data_api.get_soil_data_by_location(location.latitude, location.longitude)
        
        location_info = soil_data_api.get_location_info(location.latitude, location.longitude)
        
        response = LocationSoilResponse(
            location=soil_data['location'],
            soil_type=soil_data['soil_type'],
            soil_properties=soil_data['soil_properties'],
            confidence=soil_data['confidence'],
            sources=soil_data['sources'],
            success=soil_data['success'],
            location_info=location_info
        )
        
        logger.info(f"Successfully determined soil type: {soil_data['soil_type']} with confidence: {soil_data['confidence']}")
        return response
        
    except Exception as e:
        logger.error(f"Error fetching soil data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch soil data: {str(e)}")

@app.post("/predict-with-location")
async def predict_fertilizer_with_location(request_data: dict):
    try:
        latitude = request_data.get('latitude')
        longitude = request_data.get('longitude')
        
        if not latitude or not longitude:
            raise HTTPException(status_code=400, detail="Latitude and longitude are required")
        
        soil_data = soil_data_api.get_soil_data_by_location(latitude, longitude)
        
        fertilizer_input = FertilizerInput(
            Temperature=request_data.get('Temperature', 25.0),
            Humidity=request_data.get('Humidity', 80.0),
            Moisture=request_data.get('Moisture', 30.0),
            Soil_Type=soil_data['soil_type'],
            Crop_Type=request_data.get('Crop_Type', 'rice'),
            Nitrogen=request_data.get('Nitrogen', 85.0),
            Potassium=request_data.get('Potassium', 45.0),
            Phosphorous=request_data.get('Phosphorous', 35.0)
        )
        
        prediction_result = await predict_fertilizer(fertilizer_input)
        
        response = {
            **prediction_result.dict(),
            'soil_data': soil_data,
            'location': {'latitude': latitude, 'longitude': longitude}
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in location-based prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)