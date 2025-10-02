#!/usr/bin/env python3

import uvicorn
from main import app

if __name__ == "__main__":
    print("Starting Fertilizer Recommendation API...")
    print("API will be available at: http://localhost:8000")
    print("Interactive docs at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
