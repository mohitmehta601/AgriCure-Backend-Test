#!/usr/bin/env python3
"""
Test the Flask web interface to ensure cost estimates are always shown
for all three categories.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from main import app

def test_web_interface_cost_guarantee():
    """Test that the web interface always shows costs for all three categories."""
    
    with app.test_client() as client:
        # Test case 1: Complete form data
        print("=== Testing Web Interface Cost Guarantee ===")
        
        form_data = {
            'Temperature': '25',
            'Humidity': '70', 
            'Moisture': '60',
            'Soil_Type': 'Loamy',
            'Crop': 'Wheat',
            'Nitrogen': '30',
            'Phosphorus': '25',
            'Potassium': '20',
            'pH': '7.0',
            'Sowing_Date': '2024-01-15',
            'Field_Size': '2.5',
            'Field_Unit': 'hectares'
        }
        
        response = client.post('/predict', data=form_data)
        
        if response.status_code == 200:
            print("✓ Web interface responded successfully")
            
            # Check if the response contains cost estimates for all three categories
            html_content = response.get_data(as_text=True)
            
            # Look for cost estimate section in HTML
            if 'Cost Estimate' in html_content:
                print("✓ Cost Estimate section found in response")
                
                # Check for all three categories
                categories_found = {
                    'primary': 'Primary Fertilizer' in html_content,
                    'secondary': 'Secondary Fertilizer' in html_content,
                    'organics': 'Organic Options' in html_content
                }
                
                for category, found in categories_found.items():
                    if found:
                        print(f"✓ {category.capitalize()} category found in HTML")
                    else:
                        print(f"✗ {category.capitalize()} category missing from HTML")
                
                # Check for total cost
                if 'Total' in html_content or 'total' in html_content:
                    print("✓ Total cost section found")
                
                # Check for currency symbols
                if '₹' in html_content:
                    print("✓ Currency formatting found")
                
            else:
                print("✗ Cost Estimate section not found in response")
                
        else:
            print(f"✗ Web interface failed with status code: {response.status_code}")
            print("Response:", response.get_data(as_text=True)[:500])
        
        # Test case 2: API endpoint
        print("\n=== Testing API Endpoint ===")
        
        api_data = {
            'Temperature': 25,
            'Humidity': 70,
            'Moisture': 60,
            'Soil_Type': 'Loamy',
            'Crop': 'Wheat',
            'Nitrogen': 30,
            'Phosphorus': 25,
            'Potassium': 20,
            'pH': 7.0
        }
        
        response = client.post('/api/predict', 
                              data=json.dumps(api_data),
                              content_type='application/json')
        
        if response.status_code == 200:
            print("✓ API endpoint responded successfully")
            api_result = response.get_json()
            
            if 'predictions' in api_result:
                print("✓ API returned predictions")
            
            if 'confidences' in api_result:
                print("✓ API returned confidences")
                
        else:
            print(f"✗ API endpoint failed with status code: {response.status_code}")

if __name__ == "__main__":
    test_web_interface_cost_guarantee()
