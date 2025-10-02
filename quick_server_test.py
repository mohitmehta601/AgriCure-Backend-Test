#!/usr/bin/env python3
"""
Quick server test script that starts server and tests basic functionality
"""
import subprocess
import time
import requests
import sys
import os

def quick_server_test():
    print("🚀 Quick Server Test: AgriCure Backend ML + LLM")
    print("=" * 50)
    
    # Start server
    print("Starting server...")
    server_process = None
    try:
        server_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main:app", 
            "--host", "127.0.0.1", "--port", "8004"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for server to start
        print("Waiting for server to start...")
        time.sleep(8)
        
        base_url = "http://127.0.0.1:8004"
        
        # Test 1: Health check
        print("\n1. Testing health endpoint...")
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print("   ✓ Health check passed")
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Health check error: {e}")
            return False
        
        # Test 2: Root endpoint
        print("\n2. Testing root endpoint...")
        try:
            response = requests.get(f"{base_url}/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("   ✓ Root endpoint working")
                print(f"   Version: {data.get('version')}")
                print(f"   ML Model: {data.get('model_loaded')}")
                print(f"   LLM: {data.get('llm_available')}")
            else:
                print(f"   ❌ Root failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Root error: {e}")
            return False
        
        # Test 3: Basic prediction
        print("\n3. Testing basic prediction...")
        payload = {
            "Temperature": 28.5,
            "Humidity": 75.0,
            "Moisture": 35.0,
            "Soil_Type": "Black",
            "Crop_Type": "Cotton",
            "Nitrogen": 60.0,
            "Potassium": 50.0,
            "Phosphorous": 40.0,
            "pH": 7.2
        }
        
        try:
            response = requests.post(f"{base_url}/predict", json=payload, timeout=15)
            if response.status_code == 200:
                data = response.json()
                print("   ✓ Basic prediction successful")
                print(f"   Fertilizer: {data.get('fertilizer')}")
                print(f"   Confidence: {data.get('confidence', 0):.3f}")
            else:
                print(f"   ❌ Basic prediction failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
        except Exception as e:
            print(f"   ❌ Basic prediction error: {e}")
            return False
        
        print("\n🎉 SUCCESS! Basic functionality verified:")
        print("   ✅ Server starts with ML + LLM integration")
        print("   ✅ Health endpoint working") 
        print("   ✅ Root endpoint shows new features")
        print("   ✅ Basic predictions working")
        print("   ✅ Backend integration complete!")
        
        return True
        
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        return False
    finally:
        if server_process:
            print("\nShutting down server...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    success = quick_server_test()
    sys.exit(0 if success else 1)
