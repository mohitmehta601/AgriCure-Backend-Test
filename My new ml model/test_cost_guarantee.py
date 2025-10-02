#!/usr/bin/env python3
"""
Test script to verify that LLM cost estimation always provides
costs for all three categories: Primary, Secondary, and Organic.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm import generate_recommendation_report

def test_complete_cost_estimation():
    """Test that all three cost categories are always provided."""
    
    # Test case 1: Normal predictions with all fertilizers
    print("=== Test Case 1: Complete ML Predictions ===")
    base_inputs = {
        "Temperature": 25.0,
        "Humidity": 70.0,
        "Moisture": 60.0,
        "Soil_Type": "Loamy",
        "Crop": "Wheat",
        "Nitrogen": 30.0,
        "Phosphorus": 25.0,
        "Potassium": 20.0,
        "pH": 7.0,
        "Sowing_Date": "2024-01-15",
        "Field_Size": 2.5,
        "Field_Unit": "hectares"
    }
    
    predictions = {
        "Primary_Fertilizer": "Urea",
        "Secondary_Fertilizer": "MOP",
        "Organic_1": "Vermicompost",
        "Organic_2": "Neem Cake",
        "N_Status": "Low",
        "P_Status": "Optimal",
        "K_Status": "Low"
    }
    
    confidences = {
        "Primary_Fertilizer": 0.85,
        "Secondary_Fertilizer": 0.78,
        "N_Status": 0.82
    }
    
    report = generate_recommendation_report(base_inputs, predictions, confidences)
    
    # Verify all three categories have cost estimates
    cost_estimate = report["cost_estimate"]
    assert cost_estimate["primary"] is not None, "Primary cost should never be None"
    assert cost_estimate["secondary"] is not None, "Secondary cost should never be None"  
    assert cost_estimate["organics"] is not None, "Organics cost should never be None"
    assert cost_estimate["total"] is not None, "Total cost should never be None"
    
    print(f"✓ Primary cost: {cost_estimate['primary']}")
    print(f"✓ Secondary cost: {cost_estimate['secondary']}")
    print(f"✓ Organics cost: {cost_estimate['organics']}")
    print(f"✓ Total cost: {cost_estimate['total']}")
    
    # Test case 2: Empty predictions (worst case)
    print("\n=== Test Case 2: Empty ML Predictions ===")
    empty_predictions = {}
    empty_confidences = {}
    
    report2 = generate_recommendation_report(base_inputs, empty_predictions, empty_confidences)
    
    cost_estimate2 = report2["cost_estimate"]
    assert cost_estimate2["primary"] is not None, "Primary cost should never be None even with empty predictions"
    assert cost_estimate2["secondary"] is not None, "Secondary cost should never be None even with empty predictions"
    assert cost_estimate2["organics"] is not None, "Organics cost should never be None even with empty predictions"
    assert cost_estimate2["total"] is not None, "Total cost should never be None even with empty predictions"
    
    print(f"✓ Primary cost: {cost_estimate2['primary']}")
    print(f"✓ Secondary cost: {cost_estimate2['secondary']}")
    print(f"✓ Organics cost: {cost_estimate2['organics']}")
    print(f"✓ Total cost: {cost_estimate2['total']}")
    
    # Test case 3: Partial predictions
    print("\n=== Test Case 3: Partial ML Predictions ===")
    partial_predictions = {
        "Primary_Fertilizer": "DAP",
        "N_Status": "High"
        # No secondary or organics
    }
    
    partial_confidences = {"Primary_Fertilizer": 0.90}
    
    report3 = generate_recommendation_report(base_inputs, partial_predictions, partial_confidences)
    
    cost_estimate3 = report3["cost_estimate"]
    assert cost_estimate3["primary"] is not None, "Primary cost should never be None"
    assert cost_estimate3["secondary"] is not None, "Secondary cost should never be None even when not predicted"
    assert cost_estimate3["organics"] is not None, "Organics cost should never be None even when not predicted"
    assert cost_estimate3["total"] is not None, "Total cost should never be None"
    
    print(f"✓ Primary cost: {cost_estimate3['primary']}")
    print(f"✓ Secondary cost: {cost_estimate3['secondary']}")
    print(f"✓ Organics cost: {cost_estimate3['organics']}")
    print(f"✓ Total cost: {cost_estimate3['total']}")
    
    # Verify breakdown exists
    if "breakdown" in cost_estimate3:
        print(f"✓ Cost breakdown available with detailed info")
        breakdown = cost_estimate3["breakdown"]
        assert "primary_details" in breakdown, "Primary details should be in breakdown"
        assert "secondary_details" in breakdown, "Secondary details should be in breakdown"
        assert "organics_details" in breakdown, "Organics details should be in breakdown"
    
    print("\n🎉 All tests passed! LLM will always provide cost estimates for all three categories.")
    return True

if __name__ == "__main__":
    test_complete_cost_estimation()
