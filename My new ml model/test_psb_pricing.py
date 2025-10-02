# test_psb_pricing.py
"""
Test script specifically for PSB (Phosphate Solubilizing Bacteria) pricing issue.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.price_provider import live_price_provider, normalize_name
from llm import generate_recommendation_report
import json

def main():
    print("🧪 Testing PSB (Phosphate Solubilizing Bacteria) Pricing Fix\n")
    
    # Test normalization first
    print("🔍 Testing name normalization:")
    test_names = ["PSB", "psb", "Phosphate Solubilizing Bacteria", "phosphate solubilizing bacteria"]
    for name in test_names:
        normalized = normalize_name(name)
        print(f"  '{name}' → '{normalized}'")
    
    # Sample inputs with PSB as secondary fertilizer
    base_inputs = {
        "Field_Size": 5.5,  # Same as in the attachment
        "Field_Unit": "acres",
        "Nitrogen": 85,
        "Phosphorus": 40,
        "Potassium": 113,
        "Bulk_Density_g_cm3": 1.3,
        "Sampling_Depth_cm": 15,
        "Sowing_Date": "2025-10-15"
    }
    
    predictions = {
        "Primary_Fertilizer": "DAP",
        "Secondary_Fertilizer": "PSB",  # This was showing ₹0 before
        "Organic_1": "Vermicompost",
        "Organic_2": "Neem Cake", 
        "Organic_3": "Compost",
        "N_Status": "low",
        "P_Status": "optimal", 
        "K_Status": "low"
    }
    
    confidences = {
        "Primary_Fertilizer": 0.92,
        "Secondary_Fertilizer": 0.78
    }
    
    print(f"\n📊 Input Data:")
    print(f"Field Size: {base_inputs['Field_Size']} {base_inputs['Field_Unit']}")
    print(f"Primary Fertilizer: {predictions['Primary_Fertilizer']}")
    print(f"Secondary Fertilizer: {predictions['Secondary_Fertilizer']}")
    print(f"Organics: {predictions.get('Organic_1')}, {predictions.get('Organic_2')}, {predictions.get('Organic_3')}")
    
    print(f"\n💰 Generating Report with PSB as Secondary:")
    report = generate_recommendation_report(
        base_inputs,
        predictions,
        confidences,
        region="India",
        currency="₹",
        price_provider=None,  # Use fallback pricing
        local_rate_path="app/rate_table.json"
    )
    
    print(f"\n🎯 Cost Breakdown:")
    print(f"Primary Fertilizer (DAP): {report['cost_estimate']['primary']}")
    print(f"Secondary Fertilizer (PSB): {report['cost_estimate']['secondary']}")
    print(f"Organic Options: {report['cost_estimate']['organics']}")
    print(f"Total Cost: {report['cost_estimate']['total']}")
    
    # Show detailed breakdown
    breakdown = report['cost_estimate']['breakdown']
    print(f"\n📋 Detailed Breakdown:")
    print(f"Primary: {breakdown['primary_details']['fertilizer']} - {breakdown['primary_details']['amount_kg']}kg × {breakdown['primary_details']['price_per_kg']} = {breakdown['primary_details']['cost']}")
    print(f"Secondary: {breakdown['secondary_details']['fertilizer']} - {breakdown['secondary_details']['amount_kg']}kg × {breakdown['secondary_details']['price_per_kg']} = {breakdown['secondary_details']['cost']}")
    print(f"Organics: {breakdown['organics_details']['options_count']} options, {breakdown['organics_details']['total_amount_kg']}kg total = {breakdown['organics_details']['cost']}")
    
    # Check if PSB cost is no longer ₹0
    psb_cost = report['cost_estimate']['secondary']
    if psb_cost != "₹0":
        print(f"\n✅ SUCCESS: PSB pricing is now working! Cost: {psb_cost}")
    else:
        print(f"\n❌ ISSUE: PSB is still showing ₹0")
        
    print(f"\n📝 Notes: {report['cost_estimate']['notes']}")

if __name__ == "__main__":
    main()
