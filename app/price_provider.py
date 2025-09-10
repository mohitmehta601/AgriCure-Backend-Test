# app/price_provider.py
from typing import Optional

# Normalize common names to a canonical key used across sources/local tables
FERT_ALIAS = {
    "mop": "MOP", "murate of potash": "MOP", "muriate of potash": "MOP", "potassium chloride": "MOP",
    "sop": "SOP", "potassium sulfate": "SOP", "potassium sulphate": "SOP",
    "urea": "Urea", "dap": "DAP", "diammonium phosphate": "DAP",
    "can": "Calcium Ammonium Nitrate", "calcium ammonium nitrate": "Calcium Ammonium Nitrate",
    "ammonium sulphate": "Ammonium Sulphate", "ammonium sulfate": "Ammonium Sulphate",
    "ammonium nitrate": "Ammonium Nitrate", "ammonium chloride": "Ammonium Chloride",
    "ssp": "SSP", "single super phosphate": "SSP",
    "tsp": "TSP", "triple super phosphate": "TSP",
    "rock phosphate": "Rock Phosphate",
    "vermicompost": "Vermicompost", "neem cake": "Neem cake", "bone meal": "Bone meal",
    "compost": "Compost", "poultry manure": "Poultry manure", "wood ash": "Wood Ash",
    "fym": "FYM", "farmyard manure": "FYM",
    "green manure": "Green manure", "mustard cake": "Mustard cake",
    "banana wastes": "Banana wastes", "banana peel compost": "Banana peel compost",
    "mulch": "Mulch", "azolla": "Azolla",
    # Secondary/Biofertilizers
    "psb": "PSB", "phosphate solubilizing bacteria": "PSB",
    "rhizobium": "Rhizobium", "azospirillum": "Azospirillum", "azotobacter": "Azotobacter",
    # Advisory terms
    "balanced npk (maintenance)": "Balanced NPK (maintenance)",
    "split n doses": "Split N doses", "stop p": "Stop P", "stop k": "Stop K",
    "reduce n": "Reduce N", "avoid n": "Avoid N", "avoid potash": "Avoid Potash",
    "avoid phosphate application": "Avoid Phosphate application",
    "none": "None", "—": "—"
}

def normalize_name(name: Optional[str]) -> Optional[str]:
    """Normalize fertilizer name to canonical form for consistent lookup."""
    if not name: 
        return None
    key = name.strip().lower()
    return FERT_ALIAS.get(key, name.strip())

def live_price_provider(name: str, region: Optional[str] = None) -> Optional[float]:
    """
    Return latest market price in ₹/kg for `name`, or None if unavailable.
    Replace this stub with your real API/DB/scraper.
    
    Args:
        name: Fertilizer name (will be normalized)
        region: Geographic region for regional pricing
        
    Returns:
        Price per kg in rupees, or None if unavailable
    """
    canon = normalize_name(name)
    # TODO: wire to your live data source. For now, return None to trigger fallback.
    return None
