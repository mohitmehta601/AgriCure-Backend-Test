import os
import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 8000))
        logger.info(f"Starting server on port {port}")
        
        # Start the server with proper configuration for Railway
        uvicorn.run(
            "main:app", 
            host="0.0.0.0", 
            port=port,
            log_level="info",
            access_log=True,
            timeout_keep_alive=30
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise
