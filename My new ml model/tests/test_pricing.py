# tests/test_pricing.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.price_provider import live_price_provider
from llm import generate_recommendation_report

def dummy_provider(name, region=None):
    """Test price provider that returns known prices for testing."""
    return {"Urea": 45.0, "DAP": 160.0, "MOP": 38.0}.get(name)

def test_pricing_paths():
    """Test that pricing system works with live provider and fallbacks."""
    base_inputs = {
        "Field_Size": 2, 
        "Field_Unit": "hectares",
        "Nitrogen": 85,
        "Phosphorus": 40, 
        "Potassium": 113
    }
    preds = {
        "Primary_Fertilizer": "Urea", 
        "Secondary_Fertilizer": "SOP", 
        "Organic_1": "Vermicompost",
        "N_Status": "low",
        "P_Status": "optimal", 
        "K_Status": "low"
    }
    conf = {"Primary_Fertilizer": 0.82}
    
    data = generate_recommendation_report(
        base_inputs, 
        preds, 
        conf, 
        region="UP, India", 
        price_provider=dummy_provider
    )
    
    assert data["cost_estimate"]["primary"] is not None
    assert data["_meta"]["price_source"] == "live->fallback"
    assert data["_meta"]["region"] == "UP, India"
    print("✓ Pricing test passed!")

if __name__ == "__main__":
    test_pricing_paths()
