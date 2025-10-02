#!/usr/bin/env python3
"""Test script for the enhanced soil_api.py module"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from soil_api import soil_data_api

def test_enhanced_soil_api():
    """Test the enhanced soil API with sample coordinates"""
    print("Testing Enhanced Soil API...")
    
    # Test coordinates for Jaipur, India
    lat, lon = 26.9124, 75.7873
    
    try:
        result = soil_data_api.get_soil_data_by_location(lat, lon)
        
        print(f"\n=== Test Location: {lat}, {lon} ===")
        print(f"Success: {result['success']}")
        print(f"Soil Type: {result['soil_type']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Sources: {result['sources']}")
        
        print(f"\n=== Location Info ===")
        loc_info = result['location_info']
        for key, value in loc_info.items():
            print(f"  {key}: {value}")
        
        print(f"\n=== Soil Properties ===")
        if result['soil_properties']:
            for key, value in result['soil_properties'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        
        print(f"\n=== Location Data ===")
        location = result['location']
        for key, value in location.items():
            print(f"  {key}: {value}")
        
        # Verify required fields are present
        required_fields = ['location', 'soil_properties', 'soil_type', 'confidence', 'sources', 'success', 'location_info']
        missing_fields = [field for field in required_fields if field not in result]
        
        if missing_fields:
            print(f"\n❌ Missing required fields: {missing_fields}")
            return False
        
        # Verify soil type is one of the valid types
        valid_soil_types = {
            "Sandy", "Silty", "Laterite", "Alkaline", "Black", 
            "Clayey", "Saline", "Loamy", "Red", "Peaty"
        }
        
        if result['soil_type'] not in valid_soil_types:
            print(f"\n❌ Invalid soil type: {result['soil_type']}")
            print(f"Valid types: {valid_soil_types}")
            return False
        
        print(f"\n✅ Enhanced API test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing enhanced soil API: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_regions():
    """Test the API with different geographical regions"""
    test_locations = [
        (26.9124, 75.7873, "Jaipur, India"),
        (40.7128, -74.0060, "New York, USA"),
        (51.5074, -0.1278, "London, UK"),
        (-33.8688, 151.2093, "Sydney, Australia"),
        (1.3521, 103.8198, "Singapore"),
    ]
    
    print("\n" + "="*60)
    print("Testing Enhanced API with Different Regions")
    print("="*60)
    
    for lat, lon, name in test_locations:
        print(f"\n🌍 Testing {name} ({lat}, {lon}):")
        try:
            result = soil_data_api.get_soil_data_by_location(lat, lon)
            
            print(f"  🌱 Soil Type: {result['soil_type']}")
            print(f"  📊 Confidence: {result['confidence']:.3f}")
            print(f"  📡 Sources: {', '.join(result['sources'])}")
            
            loc_info = result['location_info']
            if loc_info.get('formatted_address'):
                print(f"  📍 Location: {loc_info['formatted_address']}")
            elif loc_info.get('city') and loc_info.get('country'):
                print(f"  📍 Location: {loc_info['city']}, {loc_info['country']}")
            
            # Show soil properties summary
            props = result['soil_properties']
            if props:
                clay = props.get('clay', 0)
                sand = props.get('sand', 0)
                silt = props.get('silt', 0)
                ph = props.get('phh2o', 0)
                print(f"  🧪 Properties: Clay={clay:.1f}%, Sand={sand:.1f}%, Silt={silt:.1f}%, pH={ph:.1f}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    success = test_enhanced_soil_api()
    if success:
        test_different_regions()
        print(f"\n🎉 All enhanced soil API tests completed!")
    else:
        print(f"\n💥 Enhanced soil API test failed!")