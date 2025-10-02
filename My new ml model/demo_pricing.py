# demo_pricing.py
"""
Demo script showing the enhanced pricing system in action.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.price_provider import live_price_provider
from llm import generate_recommendation_report
import json

def demo_live_provider(name, region=None):
    """Demo live price provider with some sample prices."""
    live_prices = {
        "Urea": 45.0,
        "DAP": 160.0, 
        "MOP": 38.0,
        "SOP": 55.0,
        "Vermicompost": 15.0
    }
    return live_prices.get(name)

def main():
    print("🌾 Enhanced Fertilizer Recommendation System - Pricing Demo\n")
    
    # Sample inputs
    base_inputs = {
        "Field_Size": 2.5,
        "Field_Unit": "hectares", 
        "Nitrogen": 85,
        "Phosphorus": 40,
        "Potassium": 113,
        "Bulk_Density_g_cm3": 1.3,
        "Sampling_Depth_cm": 15,
        "Sowing_Date": "2025-10-15"
    }
    
    predictions = {
        "Primary_Fertilizer": "Urea",
        "Secondary_Fertilizer": "MOP", 
        "Organic_1": "Vermicompost",
        "Organic_2": "Neem Cake",
        "N_Status": "low",
        "P_Status": "optimal",
        "K_Status": "low"
    }
    
    confidences = {
        "Primary_Fertilizer": 0.85,
        "Secondary_Fertilizer": 0.78
    }
    
    print("📊 Input Data:")
    print(f"Field Size: {base_inputs['Field_Size']} {base_inputs['Field_Unit']}")
    print(f"N: {base_inputs['Nitrogen']} mg/kg, P: {base_inputs['Phosphorus']} mg/kg, K: {base_inputs['Potassium']} mg/kg")
    print(f"Primary Fertilizer: {predictions['Primary_Fertilizer']}")
    print(f"Secondary Fertilizer: {predictions['Secondary_Fertilizer']}")
    print(f"Organics: {predictions.get('Organic_1', 'None')}, {predictions.get('Organic_2', 'None')}")
    
    # Test with live price provider
    print("\n💰 Testing with Live Price Provider:")
    report = generate_recommendation_report(
        base_inputs,
        predictions, 
        confidences,
        region="UP, India",
        currency="₹",
        price_provider=demo_live_provider,
        local_rate_path="app/rate_table.json"
    )
    
    print(f"Primary Cost: {report['cost_estimate']['primary']}")
    print(f"Secondary Cost: {report['cost_estimate']['secondary']}")
    print(f"Organics Cost: {report['cost_estimate']['organics']}")
    print(f"Total Cost: {report['cost_estimate']['total']}")
    print(f"Price Source: {report['_meta']['price_source']}")
    print(f"Region: {report['_meta']['region']}")
    
    # Test fallback only (no live provider)
    print("\n📁 Testing with Fallback Prices Only:")
    report_fallback = generate_recommendation_report(
        base_inputs,
        predictions,
        confidences,
        region="Maharashtra, India",
        currency="₹",
        price_provider=None,  # No live provider
        local_rate_path="app/rate_table.json"
    )
    
    print(f"Primary Cost: {report_fallback['cost_estimate']['primary']}")
    print(f"Secondary Cost: {report_fallback['cost_estimate']['secondary']}")
    print(f"Organics Cost: {report_fallback['cost_estimate']['organics']}")
    print(f"Total Cost: {report_fallback['cost_estimate']['total']}")
    
    # Show soil test conversions
    print("\n🧪 Soil Test Value Conversions:")
    soil_values = report['soil_condition']['soil_test_values']
    print(f"N: {soil_values['N']['mg_per_kg']} mg/kg → {soil_values['N']['kg_per_ha']} kg/ha")
    print(f"P: {soil_values['P']['mg_per_kg']} mg/kg → {soil_values['P']['kg_per_ha']} kg/ha")
    print(f"K: {soil_values['K']['mg_per_kg']} mg/kg → {soil_values['K']['kg_per_ha']} kg/ha")
    print(f"Bulk Density: {soil_values['bulk_density_g_cm3']} g/cm³")
    print(f"Sampling Depth: {soil_values['sampling_depth_cm']} cm")
    
    print("\n✅ Pricing system working perfectly!")

if __name__ == "__main__":
    main()
