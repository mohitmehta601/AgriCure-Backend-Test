"""
Deployment configuration for Railway
This file handles model loading gracefully for deployment
"""
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_deployment_environment():
    """Check if we're in a deployment environment and handle accordingly"""
    
    # Check if we're on Railway
    is_railway = os.getenv('RAILWAY_ENVIRONMENT_NAME') is not None
    
    # Check if model files exist
    model_path = os.path.join(os.path.dirname(__file__), "My new ml model", "models", "fertilizer_recommender.pkl")
    model_exists = os.path.exists(model_path)
    
    logger.info(f"Railway deployment: {is_railway}")
    logger.info(f"Model file exists: {model_exists}")
    
    return {
        'is_railway': is_railway,
        'model_exists': model_exists,
        'model_path': model_path
    }

def get_fallback_recommendations():
    """Provide fallback recommendations when model is not available"""
    return {
        "N": 120,
        "P": 60,
        "K": 80,
        "fertilizer_type": "NPK 20-10-10",
        "application_rate": "300 kg/ha",
        "note": "Using fallback recommendations - deploy with model for accurate predictions"
    }
