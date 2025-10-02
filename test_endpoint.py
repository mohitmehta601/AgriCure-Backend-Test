#!/usr/bin/env python3
"""Test the FastAPI /soil-data endpoint"""

import requests
import json

def test_soil_data_endpoint():
    """Test the /soil-data endpoint"""
    url = "http://localhost:8000/soil-data"
    
    # Test coordinates
    test_data = {
        "latitude": 25.152717,
        "longitude": 75.841536
    }
    
    try:
        print("Testing /soil-data endpoint...")
        print(f"URL: {url}")
        print(f"Data: {test_data}")
        
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=30
        )
        
        print(f"\nResponse Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Success! Response:")
            print(json.dumps(data, indent=2))
            
            # Verify expected fields
            expected_fields = ["location", "soil_type", "confidence", "sources", "success"]
            missing_fields = [field for field in expected_fields if field not in data]
            
            if missing_fields:
                print(f"\n⚠️  Missing fields: {missing_fields}")
            else:
                print("\n✅ All expected fields present")
                
            # Check soil type is one of the allowed values
            allowed_soil_types = [
                "Loamy", "Sandy", "Clayey", "Silty", 
                "Red", "Black", "Laterite", "Peaty", "Saline", "Alkaline"
            ]
            
            if data["soil_type"] in allowed_soil_types:
                print(f"✅ Soil type '{data['soil_type']}' is valid")
            else:
                print(f"❌ Soil type '{data['soil_type']}' is not in allowed list")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_soil_data_endpoint()
