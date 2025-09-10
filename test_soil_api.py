#!/usr/bin/env python3
"""Test script for the soil_api.py module"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from soil_api import soil_data_api

def test_soil_api():
    """Test the soil API with sample coordinates"""
    print("Testing Soil API...")
    
    # Test coordinates for Jaipur, India (known for Black soil)
    lat, lon = 26.9124, 75.7873
    
    try:
        result = soil_data_api.get_soil_data_by_location(lat, lon)
        print(f"\nTest Location: {lat}, {lon}")
        print(f"Soil Type: {result['soil_type']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Sources: {result['sources']}")
        print(f"Success: {result['success']}")
        
        if result['soil_properties']:
            print("\nSoil Properties:")
            for key, value in result['soil_properties'].items():
                print(f"  {key}: {value}")
        
        return result['success']
        
    except Exception as e:
        print(f"Error testing soil API: {e}")
        return False

def test_different_locations():
    """Test multiple locations"""
    test_locations = [
        (26.9124, 75.7873, "Jaipur, India"),
        (40.7128, -74.0060, "New York, USA"),
        (51.5074, -0.1278, "London, UK"),
        (25.1527, 75.8415, "Kota, India")
    ]
    
    print("\n" + "="*50)
    print("Testing Multiple Locations")
    print("="*50)
    
    for lat, lon, name in test_locations:
        print(f"\nTesting {name} ({lat}, {lon}):")
        try:
            result = soil_data_api.get_soil_data_by_location(lat, lon)
            print(f"  Soil Type: {result['soil_type']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Sources: {', '.join(result['sources'])}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    success = test_soil_api()
    if success:
        test_different_locations()
        print("\n✅ Soil API tests completed successfully!")
    else:
        print("\n❌ Soil API test failed!")
