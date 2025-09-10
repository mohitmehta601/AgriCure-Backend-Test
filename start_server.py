#!/usr/bin/env python3

import socket
import uvicorn
from main import app

def find_free_port(start_port=8000, max_attempts=10):
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return None

def main():
    print("🚀 Starting Fertilizer Recommendation API...")
    
    port = find_free_port()
    if port is None:
        print("❌ Could not find a free port")
        return
    
    print(f"✅ Server starting on port {port}")
    print(f"🌐 API URL: http://localhost:{port}")
    print(f"📖 Interactive docs: http://localhost:{port}/docs")
    print(f"🔍 Health check: http://localhost:{port}/health")
    print("Press Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=port,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()
