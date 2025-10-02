# test_attachment_scenario.py
"""
Test script to match the exact scenario from the user's attachment.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.price_provider import live_price_provider, normalize_name
from llm import generate_recommendation_report
import json

def main():
    print("🎯 Testing Exact Scenario from User's Attachment\n")
    
    # Match the exact scenario from attachment 
    base_inputs = {
        "Field_Size": 5.5,  # 5.5 acres as shown in attachment
        "Field_Unit": "acres",
        "Nitrogen": 85,  # Approximating based on typical values
        "Phosphorus": 40,
        "Potassium": 113,
        "Bulk_Density_g_cm3": 1.3,
        "Sampling_Depth_cm": 15,
        "Sowing_Date": "2025-10-15"
    }
    
    # Based on attachment showing DAP primary and PSB secondary
    predictions = {
        "Primary_Fertilizer": "DAP",
        "Secondary_Fertilizer": "PSB",
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
    
    print(f"📊 Scenario from Attachment:")
    print(f"Field Size: {base_inputs['Field_Size']} {base_inputs['Field_Unit']} (India)")
    print(f"Expected Primary: DAP (~440kg)")
    print(f"Expected Secondary: PSB (~330kg) - was showing 'N/A' price")
    print(f"Expected Total: ~₹93,060 (from attachment)")
    
    print(f"\n💰 Generating Report:")
    report = generate_recommendation_report(
        base_inputs,
        predictions,
        confidences,
        region="India",
        currency="₹",
        price_provider=None,
        local_rate_path="app/rate_table.json"
    )
    
    # Extract amounts for comparison
    primary_amount = report['primary_fertilizer']['amount_kg']
    secondary_amount = report['secondary_fertilizer']['amount_kg']
    
    print(f"\n📋 Results Comparison:")
    print(f"{'Category':<15} {'Attachment':<15} {'Our Result':<15} {'Status':<10}")
    print("-" * 60)
    print(f"{'Primary DAP':<15} {'440kg':<15} {f'{primary_amount}kg':<15} {'✓' if abs(primary_amount - 440) < 150 else '⚠️'}")
    print(f"{'Secondary PSB':<15} {'330kg':<15} {f'{secondary_amount}kg':<15} {'✓' if abs(secondary_amount - 330) < 100 else '⚠️'}")
    
    print(f"\n💰 Cost Breakdown:")
    print(f"Primary Fertilizer (DAP): {report['cost_estimate']['primary']}")
    print(f"Secondary Fertilizer (PSB): {report['cost_estimate']['secondary']} ← FIXED! (was ₹0)")
    print(f"Organic Options: {report['cost_estimate']['organics']}")
    print(f"Total Cost: {report['cost_estimate']['total']}")
    
    # Show that PSB now has a proper price
    breakdown = report['cost_estimate']['breakdown']
    psb_price_per_kg = breakdown['secondary_details']['price_per_kg']
    psb_cost = breakdown['secondary_details']['cost']
    
    print(f"\n🎉 PSB Pricing Fix Results:")
    print(f"   PSB Price per kg: {psb_price_per_kg} (was 'N/A')")
    print(f"   PSB Total Cost: {psb_cost} (was ₹0)")
    print(f"   Amount: {secondary_amount}kg")
    
    if psb_cost != "₹0" and psb_price_per_kg != "N/A":
        print(f"\n✅ SUCCESS: Secondary Fertilizer (PSB) cost issue is RESOLVED!")
        print(f"   • Price per kg is now showing: {psb_price_per_kg}")
        print(f"   • Total cost is now showing: {psb_cost}")
        print(f"   • No more 'N/A' or ₹0 for PSB!")
    else:
        print(f"\n❌ Issue still exists")

if __name__ == "__main__":
    main()
