from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
from typing import Dict, Any
import logging
from datetime import datetime
from soil_api import soil_data_api

# Add the new ML model directory to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_ML_MODEL_DIR = os.path.join(BASE_DIR, "My new ml model")
sys.path.insert(0, NEW_ML_MODEL_DIR)

# Import the new ML model predictor and LLM
try:
    from predictor import load_default, FertilizerRecommender
    ML_MODEL_AVAILABLE = True
    print("✅ New ML model loaded successfully")
except ImportError as e:
    ML_MODEL_AVAILABLE = False
    ML_MODEL_ERROR = str(e)
    print(f"❌ Failed to load new ML model: {e}")

# Import LLM for enhanced recommendations
try:
    from llm import generate_recommendation_report
    LLM_AVAILABLE = True
    print("✅ LLM module loaded successfully")
except ImportError as e:
    LLM_AVAILABLE = False
    LLM_ERROR = str(e)
    print(f"⚠️ LLM module not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fertilizer Recommendation API",
    description="ML API for predicting fertilizer recommendations based on soil and crop conditions",
    version="2.0.0"  # Updated version for new ML model
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
    Crop_Type: str  # Keep original field name for API compatibility
    Nitrogen: float
    Potassium: float
    Phosphorous: float  # Keep original field name for API compatibility
    pH: float = 6.5  # Add pH field with default value

class EnhancedFertilizerInput(BaseModel):
    Temperature: float
    Humidity: float
    Moisture: float
    Soil_Type: str
    Crop_Type: str
    Nitrogen: float
    Potassium: float
    Phosphorous: float
    pH: float = 6.5
    # Additional fields for LLM
    Sowing_Date: str = None  # ISO date format (YYYY-MM-DD)
    Field_Size: float = 1.0  # Default 1 hectare
    Field_Unit: str = "hectares"
    Bulk_Density_g_cm3: float = 1.3  # Default bulk density
    Sampling_Depth_cm: float = 15.0  # Default sampling depth

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
    
class EnhancedFertilizerResponse(BaseModel):
    predictions: Dict[str, str]
    confidences: Dict[str, float]
    prediction_info: Dict[str, Any]

class LLMEnhancedResponse(BaseModel):
    ml_model_prediction: Dict[str, Any]
    soil_condition: Dict[str, Any]
    primary_fertilizer: Dict[str, Any]
    secondary_fertilizer: Dict[str, Any]
    organic_alternatives: list
    application_timing: Dict[str, str]
    cost_estimate: Dict[str, Any]
    meta_info: Dict[str, Any] = None

# Global model instance
_recommender: FertilizerRecommender = None

def get_recommender() -> FertilizerRecommender:
    global _recommender
    if _recommender is None:
        if not ML_MODEL_AVAILABLE:
            raise Exception(f"ML model not available: {ML_MODEL_ERROR}")
        _recommender = load_default()
    return _recommender

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Fertilizer Recommendation API...")
    try:
        if ML_MODEL_AVAILABLE:
            recommender = get_recommender()
            logger.info("New ML model loaded successfully!")
            logger.info(f"Model features: {recommender.features}")
            logger.info(f"Model targets: {recommender.targets}")
        else:
            logger.error(f"Failed to load new ML model: {ML_MODEL_ERROR}")
            
        if LLM_AVAILABLE:
            logger.info("LLM module loaded successfully!")
        else:
            logger.warning(f"LLM module not available: {LLM_ERROR}")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

@app.get("/")
async def root():
    try:
        return {
            "message": "Fertilizer Recommendation API",
            "version": "2.0.0",
            "status": "running",
            "model_loaded": ML_MODEL_AVAILABLE and _recommender is not None,
            "model_type": "Enhanced Ensemble Model" if ML_MODEL_AVAILABLE else "Not Available",
            "llm_available": LLM_AVAILABLE,
            "features": ["ML Predictions", "LLM Enhanced Reports", "Cost Estimation", "Organic Alternatives"],
            "timestamp": datetime.now().isoformat(),
            "service": "fertilizer-recommendation-api"
        }
    except Exception as e:
        logger.error(f"Root endpoint error: {str(e)}")
        return {
            "message": "Fertilizer Recommendation API",
            "version": "2.0.0",
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
            "model_loaded": ML_MODEL_AVAILABLE and _recommender is not None,
            "model_available": ML_MODEL_AVAILABLE,
            "model_type": "Enhanced Ensemble Model" if ML_MODEL_AVAILABLE else "Not Available",
            "service": "fertilizer-recommendation-api",
            "version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        return {
            "status": "healthy",
            "service": "fertilizer-recommendation-api",
            "version": "2.0.0"
        }

@app.get("/readiness")
async def readiness_check():
    """Readiness probe - indicates if the service is ready to serve requests"""
    try:
        return {
            "status": "ready" if (ML_MODEL_AVAILABLE and _recommender is not None) else "not_ready",
            "model_loaded": ML_MODEL_AVAILABLE and _recommender is not None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception:
        return {"status": "ready", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=FertilizerResponse)
async def predict_fertilizer(input_data: FertilizerInput):
    if not ML_MODEL_AVAILABLE:
        raise HTTPException(status_code=500, detail=f"ML model not available: {ML_MODEL_ERROR}")
    
    try:
        # Input validation
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
        if not (4.0 <= input_data.pH <= 9.0):
            raise HTTPException(status_code=400, detail="pH must be between 4.0 and 9.0")
        
        # Prepare features for the new model (map field names)
        features = {
            "Temperature": input_data.Temperature,
            "Humidity": input_data.Humidity,
            "Moisture": input_data.Moisture,
            "Soil_Type": input_data.Soil_Type,
            "Crop": input_data.Crop_Type,  # Map Crop_Type to Crop
            "Nitrogen": input_data.Nitrogen,
            "Phosphorus": input_data.Phosphorous,  # Map Phosphorous to Phosphorus
            "Potassium": input_data.Potassium,
            "pH": input_data.pH
        }
        
        recommender = get_recommender()
        predictions, confidences = recommender.predict(features)
        
        # For backward compatibility, return the primary fertilizer recommendation
        primary_fertilizer = predictions.get("Primary_Fertilizer", "Unknown")
        primary_confidence = confidences.get("Primary_Fertilizer", 0.0)
        
        model_info = {
            "model_type": "Enhanced Ensemble Model",
            "all_predictions": predictions,
            "all_confidences": confidences,
            "features_used": list(features.keys()),
            "targets": recommender.targets
        }
        
        logger.info(f"Prediction made: {primary_fertilizer} with confidence {primary_confidence:.4f}")
        
        return FertilizerResponse(
            fertilizer=primary_fertilizer,
            confidence=primary_confidence,
            prediction_info=model_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-enhanced", response_model=EnhancedFertilizerResponse)
async def predict_fertilizer_enhanced(input_data: FertilizerInput):
    """Enhanced prediction endpoint that returns all predictions from the new model"""
    if not ML_MODEL_AVAILABLE:
        raise HTTPException(status_code=500, detail=f"ML model not available: {ML_MODEL_ERROR}")
    
    try:
        # Input validation (same as above)
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
        if not (4.0 <= input_data.pH <= 9.0):
            raise HTTPException(status_code=400, detail="pH must be between 4.0 and 9.0")
        
        # Prepare features for the new model
        features = {
            "Temperature": input_data.Temperature,
            "Humidity": input_data.Humidity,
            "Moisture": input_data.Moisture,
            "Soil_Type": input_data.Soil_Type,
            "Crop": input_data.Crop_Type,
            "Nitrogen": input_data.Nitrogen,
            "Phosphorus": input_data.Phosphorous,
            "Potassium": input_data.Potassium,
            "pH": input_data.pH
        }
        
        recommender = get_recommender()
        predictions, confidences = recommender.predict(features)
        
        model_info = {
            "model_type": "Enhanced Ensemble Model",
            "features_used": list(features.keys()),
            "targets": recommender.targets,
            "cv_scores": recommender.cv_scores
        }
        
        logger.info(f"Enhanced prediction made for all targets: {list(predictions.keys())}")
        
        return EnhancedFertilizerResponse(
            predictions=predictions,
            confidences=confidences,
            prediction_info=model_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during enhanced prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced prediction error: {str(e)}")

@app.post("/predict-llm-enhanced", response_model=LLMEnhancedResponse)
async def predict_fertilizer_llm_enhanced(input_data: EnhancedFertilizerInput):
    """LLM-enhanced prediction endpoint with comprehensive fertilizer recommendations, cost analysis, and organic alternatives"""
    if not ML_MODEL_AVAILABLE:
        raise HTTPException(status_code=500, detail=f"ML model not available: {ML_MODEL_ERROR}")
    
    if not LLM_AVAILABLE:
        raise HTTPException(status_code=500, detail=f"LLM module not available: {LLM_ERROR}")
    
    try:
        # Input validation
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
        if not (4.0 <= input_data.pH <= 9.0):
            raise HTTPException(status_code=400, detail="pH must be between 4.0 and 9.0")
        if not (0.1 <= input_data.Field_Size <= 1000):
            raise HTTPException(status_code=400, detail="Field size must be between 0.1 and 1000")
        
        # Prepare features for the ML model
        features = {
            "Temperature": input_data.Temperature,
            "Humidity": input_data.Humidity,
            "Moisture": input_data.Moisture,
            "Soil_Type": input_data.Soil_Type,
            "Crop": input_data.Crop_Type,  # Map Crop_Type to Crop
            "Nitrogen": input_data.Nitrogen,
            "Phosphorus": input_data.Phosphorous,  # Map Phosphorous to Phosphorus
            "Potassium": input_data.Potassium,
            "pH": input_data.pH
        }
        
        # Get ML predictions first
        recommender = get_recommender()
        predictions, confidences = recommender.predict(features)
        
        # Prepare inputs for LLM
        base_inputs = {
            "Temperature": input_data.Temperature,
            "Humidity": input_data.Humidity,
            "Moisture": input_data.Moisture,
            "Soil_Type": input_data.Soil_Type,
            "Crop": input_data.Crop_Type,
            "Nitrogen": input_data.Nitrogen,
            "Phosphorus": input_data.Phosphorous,
            "Potassium": input_data.Potassium,
            "pH": input_data.pH,
            "Sowing_Date": input_data.Sowing_Date or "2024-01-01",
            "Field_Size": input_data.Field_Size,
            "Field_Unit": input_data.Field_Unit,
            "Bulk_Density_g_cm3": input_data.Bulk_Density_g_cm3,
            "Sampling_Depth_cm": input_data.Sampling_Depth_cm
        }
        
        # Generate LLM-enhanced report
        llm_report = generate_recommendation_report(
            base_inputs=base_inputs,
            predictions=predictions,
            confidences=confidences,
            use_gemini_for_text=False  # Start with local generation, can be made configurable
        )
        
        logger.info(f"LLM-enhanced prediction generated successfully")
        logger.info(f"Primary fertilizer: {llm_report['primary_fertilizer']['name']}")
        logger.info(f"Total cost estimate: {llm_report['cost_estimate']['total']}")
        
        return LLMEnhancedResponse(
            ml_model_prediction=llm_report["ml_model_prediction"],
            soil_condition=llm_report["soil_condition"],
            primary_fertilizer=llm_report["primary_fertilizer"],
            secondary_fertilizer=llm_report["secondary_fertilizer"],
            organic_alternatives=llm_report["organic_alternatives"],
            application_timing=llm_report["application_timing"],
            cost_estimate=llm_report["cost_estimate"],
            meta_info=llm_report.get("_meta")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during LLM-enhanced prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM prediction error: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    if not ML_MODEL_AVAILABLE:
        raise HTTPException(status_code=500, detail=f"ML model not available: {ML_MODEL_ERROR}")
    
    try:
        recommender = get_recommender()
        return {
            "model_type": "Enhanced Ensemble Model",
            "features": recommender.features,
            "targets": recommender.targets,
            "cv_scores": recommender.cv_scores,
            "available_models": {
                target: list(models.keys()) 
                for target, models in recommender.models.items()
            },
            "label_encoders": {
                target: encoder.classes_.tolist() 
                for target, encoder in recommender.label_encoders.items()
            }
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model info error: {str(e)}")

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
            Crop_Type=request_data.get('Crop_Type', 'Rice'),  # Updated default
            Nitrogen=request_data.get('Nitrogen', 85.0),
            Potassium=request_data.get('Potassium', 45.0),
            Phosphorous=request_data.get('Phosphorous', 35.0),
            pH=request_data.get('pH', 6.5)
        )
        
        prediction_result = await predict_fertilizer_enhanced(fertilizer_input)
        
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