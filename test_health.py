#!/usr/bin/env python3
"""
Simple test script to verify the health endpoint is working
"""
import requests
import time
import sys

def test_health_endpoint(base_url="http://localhost:8000"):
    """Test the health endpoint"""
    try:
        print(f"Testing health endpoint at {base_url}/health")
        response = requests.get(f"{base_url}/health", timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Health check passed!")
            return True
        else:
            print("❌ Health check failed!")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_root_endpoint(base_url="http://localhost:8000"):
    """Test the root endpoint"""
    try:
        print(f"\nTesting root endpoint at {base_url}/")
        response = requests.get(f"{base_url}/", timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Root endpoint works!")
            return True
        else:
            print("❌ Root endpoint failed!")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print("Starting health check tests...")
    
    # Test health endpoint
    health_ok = test_health_endpoint(base_url)
    
    # Test root endpoint
    root_ok = test_root_endpoint(base_url)
    
    if health_ok and root_ok:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
