import json
import os
from datetime import datetime, date
from typing import Dict, Any, Callable, Optional, List

from dotenv import load_dotenv
from app.price_provider import live_price_provider, normalize_name

# ---------- (0) Helper function for unit conversion ----------
def mgkg_to_kg_ha(value_mgkg: float, bulk_density_g_cm3: float = 1.3, depth_cm: float = 15) -> float:
    """
    Convert soil test values from mg/kg (ppm) to kg/ha.
    
    Formula: kg/ha = (mg/kg) × bulk_density(g/cm³) × sampling_depth(cm) × 0.1
    
    Args:
        value_mgkg: Soil test value in mg/kg (ppm)
        bulk_density_g_cm3: Soil bulk density in g/cm³ (default: 1.3)
        depth_cm: Soil sampling depth in cm (default: 15)
    
    Returns:
        Converted value in kg/ha, rounded to 2 decimal places
    """
    return round(float(value_mgkg) * bulk_density_g_cm3 * depth_cm * 0.1, 2)

# ---------- (A) OPTIONAL: Gemini only for phrasing; prices/amounts are computed locally ----------
def _get_gemini_client():
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "google-generativeai package is required. Install dependencies from requirements.txt"
        ) from e

    load_dotenv(override=False)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Export it or add to a .env file.")
    genai.configure(api_key=api_key)
    return genai


# ---------- (B) Simple NPK registry ----------
FERTILIZER_NPK: Dict[str, str] = {
    # Primary/Inorganic (examples)
    "Urea": "46-0-0",
    "DAP": "18-46-0",
    "MOP": "0-0-60",
    "SOP": "0-0-50",
    "Ammonium Sulphate": "21-0-0",
    "Potassium sulfate": "0-0-50",
    "Calcium Ammonium Nitrate": "26-0-0",
    # Organics (for display only)
    "Vermicompost": "NPK varies",
    "Neem Cake": "NPK varies (~5-1-1)",
    "Bone Meal": "NPK varies (~3-15-0)",
    "Compost": "NPK varies",
    "Poultry manure": "NPK varies",
    "Wood Ash": "K-rich",
}

# If rate_table.json not present, use these ₹/kg example prices as a fallback.
# Replace with your region's baseline; live prices override these.
from typing import Dict

RATE_TABLE_DEFAULT: Dict[str, float] = {
    # --- Primary Fertilizers ---
    "Urea": 40.0,                       # ₹/kg
    "DAP": 150.0,
    "MOP": 33.0,
    "SOP": 50.0,
    "SSP": 25.0,
    "TSP": 45.0,
    "Ammonium Sulphate": 24.0,
    "Ammonium Nitrate": 32.0,
    "Ammonium Chloride": 28.0,
    "Calcium Ammonium Nitrate": 30.0,
    "Rock Phosphate": 20.0,
    "Potassium sulfate": 50.0,

    # --- Secondary / Biofertilizers ---
    "PSB": 15.0,
    "Rhizobium": 15.0,
    "Azospirillum": 15.0,
    "Azotobacter": 15.0,
    "Azolla": 10.0,

    # --- Organic Fertilizers ---
    "Vermicompost": 12.0,
    "Compost": 6.0,
    "FYM": 5.0,                        # Farmyard manure
    "Green manure": 7.0,
    "Neem cake": 25.0,
    "Mustard cake": 20.0,
    "Bone meal": 18.0,
    "Poultry manure": 8.0,
    "Banana wastes": 5.0,
    "Banana peel compost": 6.0,
    "Mulch": 2.0,
    "Wood Ash": 3.0,

    # --- Advisory Placeholders (non-materials, keep 0) ---
    "Balanced NPK (maintenance)": 0.0,
    "Split N doses": 0.0,
    "Stop P": 0.0,
    "Stop K": 0.0,
    "Reduce N": 0.0,
    "Avoid N": 0.0,
    "Avoid Potash": 0.0,
    "Avoid Phosphate application": 0.0,
    "None": 0.0,
    "—": 0.0
}

def _load_local_rate_table(path: str = "app/rate_table.json") -> Dict[str, float]:
    """Load local fallback prices if available."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Expect {"prices":{"Urea":40, ...}} or {"Urea":40,...}
            return data.get("prices", data)
    except Exception:
        return RATE_TABLE_DEFAULT


# ---------- (C) Price provider plumbing ----------
PriceProvider = Callable[[str, Optional[str]], Optional[float]]
# Signature: (fertilizer_name, region) -> price_per_kg or None

# Default fallback rates if file is missing
DEFAULT_FALLBACK_RATES = {
    "Urea": 40, "DAP": 150, "MOP": 33, "SOP": 50, "Calcium Ammonium Nitrate": 30, "Ammonium Sulphate": 24,
    "Vermicompost": 12, "Neem Cake": 25, "Bone Meal": 18, "Compost": 6, "Poultry manure": 8, "Wood Ash": 3
}

def _load_local_rate_table(path: str = "app/rate_table.json") -> Dict[str, Any]:
    """Load local fallback prices and metadata if available."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            prices = data.get("prices") or data
            return {"prices": prices, "currency": data.get("currency", "₹"), "region": data.get("region")}
    except Exception:
        return {"prices": DEFAULT_FALLBACK_RATES, "currency": "₹", "region": None}

def _resolve_price(
    name: Optional[str], 
    region: Optional[str], 
    price_provider: Optional[PriceProvider], 
    local_table: Dict[str, Any]
) -> Optional[float]:
    """Resolve price using live provider first, then local table, then defaults."""
    if not name: 
        return None
    canon = normalize_name(name)
    
    # 1) Try live provider
    if price_provider:
        try:
            p = price_provider(canon, region)
            if p and p > 0: 
                return float(p)
        except Exception:
            pass
    
    # 2) Try local file
    prices = local_table.get("prices", {})
    if canon in prices:
        try: 
            return float(prices[canon])
        except Exception: 
            pass
    
    # 3) Last-ditch default
    if canon in DEFAULT_FALLBACK_RATES:
        return float(DEFAULT_FALLBACK_RATES[canon])
    
    return None

def _fmt_money(val: Optional[float], currency: str = "₹", show_zero: bool = True) -> str:
    """Format monetary value with currency symbol, always return a string."""
    if val is None:
        return f"{currency}0" if show_zero else "N/A"
    if val == 0.0:
        return f"{currency}0" if show_zero else "N/A"
    return f"{currency}{int(round(val)):,}"


# ---------- (D) Dose logic ----------
def _dose_factor_from_status(status: Optional[str]) -> float:
    """
    Simple, transparent scaling based on ML status:
    - Low  => +25%
    - High => -20%
    - Optimal/None/Unknown => 0%
    """
    if not status:
        return 0.0
    s = status.strip().lower()
    if s == "low":
        return 0.25
    if s == "high":
        return -0.20
    return 0.0  # optimal / unknown


def _base_reco_kg_per_hectare(kind: str) -> float:
    """
    Conservative base dose per hectare for the *chosen* item.
    These are starting points; status deltas are applied on top.
    Tweak to your agronomy table if you have crop-specific rates.
    """
    k = kind.lower()
    if k in {"urea", "calcium ammonium nitrate", "ammonium sulphate"}:
        return 100.0  # kg/ha (Nitrogen sources)
    if k in {"mop", "potassium sulfate", "sop"}:
        return 60.0   # kg/ha (Potassium sources)
    if k in {"dap"}:
        return 80.0   # kg/ha (Phosphorus + some N)
    # Organics (lighter, supplementary)
    if k in {"vermicompost"}:
        return 200.0
    if k in {"neem cake"}:
        return 40.0
    if k in {"bone meal"}:
        return 30.0
    if k in {"compost"}:
        return 400.0
    if k in {"poultry manure"}:
        return 150.0
    if k in {"wood ash"}:
        return 60.0
    return 60.0


def _scaled_amount_kg(name: str, field_size: float, status_delta: float) -> float:
    per_ha = _base_reco_kg_per_hectare(name)
    per_ha = max(per_ha * (1.0 + status_delta), 0.0)
    return round(per_ha * field_size)


# ---------- (E) Human tips (optionally refined by Gemini later) ----------
def _method_hint(name: str) -> str:
    n = name.lower()
    if n in {"urea", "calcium ammonium nitrate", "ammonium sulphate"}:
        return "Apply 2–3 split doses as top-dressing during vegetative growth."
    if n in {"mop", "sop", "potassium sulfate"}:
        return "Broadcast or band place; avoid waterlogging; apply around fruiting stage for quality."
    if n in {"dap"}:
        return "Apply at basal dose during sowing/land prep; keep seeds away from direct contact."
    # organics
    if n in {"vermicompost", "compost", "poultry manure"}:
        return "Incorporate into topsoil 2–3 weeks before sowing for mineralization."
    if n in {"neem cake"}:
        return "Mix into soil at land preparation; slow-release nitrogen + pest deterrence."
    if n in {"bone meal"}:
        return "Use as basal before sowing/transplanting; phosphorus source."
    if n in {"wood ash"}:
        return "Apply lightly; provides K and raises pH—avoid overuse on alkaline soils."
    return "Follow label guidance and local agronomy recommendations."


def _application_timing_text(sowing_date_str: Optional[str]) -> Dict[str, str]:
    primary = "Give main fertilizer before sowing and again in 2–3 small doses as crop grows."
    secondary = "Use during flowering or fruiting stage when the crop needs extra boost."
    organics = "Mix into soil 2–3 weeks before sowing so it breaks down in time."

    if sowing_date_str:
        try:
            sd = date.fromisoformat(str(sowing_date_str))
            primary = f"Give main fertilizer before {sd} and again in 2–3 small doses as crop grows."
            secondary = f"Use during flowering or fruiting stage after {sd} when the crop needs extra boost."
            organics = f"Mix into soil 2–3 weeks before {sd} so it breaks down in time."
        except Exception:
            pass
    return {"primary": primary, "secondary": secondary, "organics": organics}


# ---------- (F) Main function ----------
def generate_recommendation_report(
    base_inputs: Dict[str, Any],
    predictions: Dict[str, Any],
    confidences: Dict[str, float],
    *,
    region: Optional[str] = None,
    currency: str = "₹",
    price_provider: Optional[PriceProvider] = None,
    local_rate_path: str = "app/rate_table.json",
    use_gemini_for_text: bool = False,
) -> Dict[str, Any]:
    """
    Build a JSON-ready recommendation shaped exactly for your UI.

    - Amounts (kg) are computed deterministically from ML statuses + field size.
    - Prices come from `price_provider(name, region)` (if given), else local rate table.
    - Text phrasing can optionally be refined by Gemini (set use_gemini_for_text=True).
    """
    print(f"🔬 Generating LLM report from ML predictions:")
    print(f"   Predictions: {predictions}")
    print(f"   Confidences: {confidences}")
    print(f"   Base inputs: {base_inputs}")

    # ---------- Inputs ----------
    sowing_date = base_inputs.get("Sowing_Date")  # ISO date preferred
    field_size = float(base_inputs.get("Field_Size", 1.0) or 1.0)
    field_unit = base_inputs.get("Field_Unit", "hectares")

    # Get bulk density and sampling depth with defaults
    bulk_density = float(base_inputs.get("Bulk_Density_g_cm3", 1.3))
    sampling_depth = float(base_inputs.get("Sampling_Depth_cm", 15))

    # Get soil test values and convert from mg/kg to kg/ha
    nitrogen_mgkg = float(base_inputs.get("Nitrogen", 0))
    phosphorus_mgkg = float(base_inputs.get("Phosphorus", 0))
    potassium_mgkg = float(base_inputs.get("Potassium", 0))

    nitrogen_kgha = mgkg_to_kg_ha(nitrogen_mgkg, bulk_density, sampling_depth)
    phosphorus_kgha = mgkg_to_kg_ha(phosphorus_mgkg, bulk_density, sampling_depth)
    potassium_kgha = mgkg_to_kg_ha(potassium_mgkg, bulk_density, sampling_depth)

    n_status = predictions.get("N_Status")
    p_status = predictions.get("P_Status")
    k_status = predictions.get("K_Status")

    primary_name = predictions.get("Primary_Fertilizer")
    secondary_name = predictions.get("Secondary_Fertilizer")

    print(f"🌱 ML Model Analysis:")
    print(f"   Primary Fertilizer: {primary_name}")
    print(f"   Secondary Fertilizer: {secondary_name}")
    print(f"   Nutrient Status - N: {n_status}, P: {p_status}, K: {k_status}")

    organics: List[str] = []
    for key in ("Organic_1", "Organic_2", "Organic_3"):
        v = predictions.get(key)
        if v and isinstance(v, str):
            organics.append(v)

    ph_amend = predictions.get("pH_Amendment") or predictions.get("PH_Amendment")

    # ---------- Price book ----------
    local_table = _load_local_rate_table(local_rate_path)
    currency = currency or local_table.get("currency", "₹")
    effective_region = region or local_table.get("region")

    # ---------- Amounts ----------
    # Primary is tied most to N_status if it's an N source; otherwise use matching status
    if primary_name:
        if primary_name.lower() in {"urea", "calcium ammonium nitrate", "ammonium sulphate", "dap"}:
            primary_delta = _dose_factor_from_status(n_status)
        elif primary_name.lower() in {"mop", "sop", "potassium sulfate"}:
            primary_delta = _dose_factor_from_status(k_status)
        else:
            primary_delta = 0.0
        primary_amount = _scaled_amount_kg(primary_name, field_size, primary_delta)
    else:
        primary_amount = 0

    if secondary_name:
        if secondary_name.lower() in {"mop", "sop", "potassium sulfate"}:
            secondary_delta = _dose_factor_from_status(k_status)
        elif secondary_name.lower() in {"dap"}:
            secondary_delta = _dose_factor_from_status(p_status)
        else:
            secondary_delta = 0.0
        secondary_amount = _scaled_amount_kg(secondary_name, field_size, secondary_delta)
    else:
        secondary_amount = 0

    organics_blocks = []
    for o in organics:
        amt = _scaled_amount_kg(o, field_size, 0.0)
        organics_blocks.append(
            {
                "name": o,
                "amount_kg": amt,
                "reason": _method_hint(o).split(".")[0],  # short line
                "timing": "Apply near land preparation or as basal before sowing.",
            }
        )

    # ---------- Prices & totals ----------
    primary_price = _resolve_price(primary_name, effective_region, price_provider, local_table) if primary_name else None
    secondary_price = _resolve_price(secondary_name, effective_region, price_provider, local_table) if secondary_name else None

    # Handle organics pricing
    organics_cost_values = []
    organics_all_priced = True
    for block in organics_blocks:
        p = _resolve_price(block["name"], effective_region, price_provider, local_table)
        cost = (block["amount_kg"] * p) if (p is not None) else None
        organics_cost_values.append(cost)
        if cost is None:
            organics_all_priced = False

    # Ensure all three categories always have cost values (never None)
    primary_cost = (primary_amount * primary_price) if (primary_price is not None and primary_amount > 0) else 0.0
    secondary_cost = (secondary_amount * secondary_price) if (secondary_price is not None and secondary_amount > 0) else 0.0
    
    # For organics, if no organics are recommended, set cost to 0.0
    if not organics_blocks:
        organics_cost = 0.0
    else:
        # If some organic costs are available, use partial sum; if none available, use 0.0
        available_costs = [c for c in organics_cost_values if c is not None]
        organics_cost = sum(available_costs, 0.0) if available_costs else 0.0

    # Total cost is always calculable since all components are now numeric
    total_cost = primary_cost + secondary_cost + organics_cost

    # ---------- Soil analysis & notes ----------
    nutrient_deficiencies = []
    if (n_status or "").lower() == "low":
        nutrient_deficiencies.append("Nitrogen")
    if (p_status or "").lower() == "low":
        nutrient_deficiencies.append("Phosphorus")
    if (k_status or "").lower() == "low":
        nutrient_deficiencies.append("Potassium")

    recs = [
        "Maintain current pH levels" if not ph_amend or str(ph_amend).lower() in {"none", "na", "optimal"} else f"Apply pH amendment: {ph_amend}",
        "Maintain current moisture levels",
    ]
    if nutrient_deficiencies:
        recs.append("Address " + ", ".join(nutrient_deficiencies) + " deficiency")
    recs += [
        "Regular soil testing every 6 months is recommended",
        "Consider crop rotation to maintain soil health",
    ]

    # ---------- Application timing ----------
    timing = _application_timing_text(sowing_date)

    # ---------- Validation: Ensure all categories have meaningful content ----------
    # If ML model didn't predict any fertilizers, provide default recommendations with zero costs
    if not primary_name:
        primary_name = "No primary fertilizer recommended"
        primary_amount = 0
        primary_cost = 0.0
        primary_reason = "Based on soil analysis, current nutrient levels appear sufficient for primary fertilization."
    
    if not secondary_name:
        secondary_name = "No secondary fertilizer recommended"
        secondary_amount = 0
        secondary_cost = 0.0
        secondary_reason = "Soil nutrient levels indicate no additional secondary fertilization needed at this time."
    
    if not organics_blocks:
        # Add a default organic recommendation even if ML didn't suggest any
        organics_blocks = [{
            "name": "Compost (optional)",
            "amount_kg": int(_scaled_amount_kg("Compost", field_size, 0.0)),
            "reason": "General soil health improvement",
            "timing": "Apply as needed for long-term soil health benefits."
        }]
        # Recalculate organics cost with the default option
        default_organic_price = _resolve_price("Compost", effective_region, price_provider, local_table)
        organics_cost = (organics_blocks[0]["amount_kg"] * default_organic_price) if default_organic_price else 0.0
        # Recalculate total
        total_cost = primary_cost + secondary_cost + organics_cost

    # ---------- Confidence ----------
    # Use Primary_Fertilizer confidence if provided else average of all
    pri_conf = confidences.get("Primary_Fertilizer")
    if pri_conf is None and confidences:
        pri_conf = sum(confidences.values()) / max(len(confidences), 1)
    confidence_percent = round(float(pri_conf) * 100) if pri_conf is not None else None

    # ---------- Friendly reasons (with ML-aware explanations) ----------
    def generate_smart_reason(fertilizer_name: str, nutrient_status: str, is_primary: bool = True) -> str:
        """Generate intelligent explanations based on ML predictions."""
        if not fertilizer_name:
            return "No fertilizer recommended by the model."
        
        role = "primary" if is_primary else "secondary"
        status_info = ""
        
        # Add nutrient status context
        if nutrient_status:
            if nutrient_status.lower() == "low":
                status_info = f"ML model detected low {nutrient_status.split('_')[0]} levels, "
            elif nutrient_status.lower() == "high":
                status_info = f"ML model detected high {nutrient_status.split('_')[0]} levels, "
            else:
                status_info = f"ML model detected optimal {nutrient_status.split('_')[0]} levels, "
        
        base_reason = f"Selected as {role} fertilizer based on soil analysis and crop requirements. "
        
        # Add fertilizer-specific guidance
        fert_lower = fertilizer_name.lower()
        if fert_lower in {"urea", "calcium ammonium nitrate", "ammonium sulphate"}:
            specific = "Provides essential nitrogen for vegetative growth and protein synthesis."
        elif fert_lower in {"dap"}:
            specific = "Supplies both nitrogen and phosphorus for root development and early growth."
        elif fert_lower in {"mop", "sop", "potassium sulfate"}:
            specific = "Enhances fruit quality, disease resistance, and water use efficiency."
        else:
            specific = "Provides balanced nutrition according to soil test recommendations."
        
        return status_info + base_reason + specific

    primary_reason = generate_smart_reason(primary_name, n_status, True)
    secondary_reason = generate_smart_reason(secondary_name, k_status, False)

    # Optionally let Gemini polish the one-liners (never prices/amounts).
    if use_gemini_for_text:
        try:
            genai = _get_gemini_client()
            model = genai.GenerativeModel("gemini-1.5-flash")
            txt = model.generate_content(
                "Rewrite in 1–2 short farmer-friendly lines each (no numbers/doses/prices):\n"
                f"Primary: {_method_hint(primary_name) if primary_name else ''}\n"
                f"Secondary: {_method_hint(secondary_name) if secondary_name else ''}"
            ).text or ""
            parts = [p.strip("• ").strip() for p in txt.splitlines() if p.strip()]
            if parts:
                primary_reason = parts[0][:180]
                if len(parts) > 1:
                    secondary_reason = parts[1][:180]
        except Exception:
            pass

    # ---------- Compose UI-ready JSON ----------
    data: Dict[str, Any] = {
        "ml_model_prediction": {
            "name": (primary_name or "Fertilizer"),
            "confidence_percent": confidence_percent,
            "npk": f"NPK: {FERTILIZER_NPK.get(primary_name, '—')}" if primary_name else "—",
        },
        "soil_condition": {
            "ph_status": "Optimal" if not ph_amend or str(ph_amend).lower() in {"none", "na", "optimal"} else "Needs amendment",
            "moisture_status": "Optimal",
            "nutrient_deficiencies": nutrient_deficiencies,
            "recommendations": recs,
            "soil_test_values": {
                "units_input": "mg/kg",
                "units_converted": "kg/ha",
                "bulk_density_g_cm3": bulk_density,
                "sampling_depth_cm": sampling_depth,
                "N": {"mg_per_kg": nitrogen_mgkg, "kg_per_ha": nitrogen_kgha},
                "P": {"mg_per_kg": phosphorus_mgkg, "kg_per_ha": phosphorus_kgha},
                "K": {"mg_per_kg": potassium_mgkg, "kg_per_ha": potassium_kgha}
            },
        },
        "primary_fertilizer": {
            "name": primary_name or "—",
            "amount_kg": primary_amount,
            "reason": primary_reason,
            "application_method": _method_hint(primary_name) if primary_name else "—",
        },
        "secondary_fertilizer": {
            "name": secondary_name or "—",
            "amount_kg": secondary_amount,
            "reason": secondary_reason if secondary_name else "—",
            "application_method": _method_hint(secondary_name) if secondary_name else "—",
        },
        "organic_alternatives": organics_blocks,
        "application_timing": timing,
        "cost_estimate": {
            "primary": _fmt_money(primary_cost, currency),
            "secondary": _fmt_money(secondary_cost, currency),
            "organics": _fmt_money(organics_cost, currency),
            "total": _fmt_money(total_cost, currency),
            "notes": f"For {field_size} {field_unit}"
                     + (f" in {effective_region}" if effective_region else "")
                     + ". Prices fetched from live provider when available; fallback to local rate table."
                     + " All three categories (Primary, Secondary, Organic) are always included for complete cost analysis.",
            "breakdown": {
                "primary_details": {
                    "fertilizer": primary_name or "Not recommended",
                    "amount_kg": primary_amount,
                    "price_per_kg": _fmt_money(primary_price, currency) if primary_price else "N/A",
                    "cost": _fmt_money(primary_cost, currency)
                },
                "secondary_details": {
                    "fertilizer": secondary_name or "Not recommended", 
                    "amount_kg": secondary_amount,
                    "price_per_kg": _fmt_money(secondary_price, currency) if secondary_price else "N/A",
                    "cost": _fmt_money(secondary_cost, currency)
                },
                "organics_details": {
                    "options_count": len(organics_blocks),
                    "total_amount_kg": sum(block["amount_kg"] for block in organics_blocks),
                    "cost": _fmt_money(organics_cost, currency)
                }
            }
        },
        "_meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "inputs": base_inputs,
            "predictions": predictions,
            "confidences": confidences,
            "region": effective_region,
            "currency": currency,
            "price_source": "live->fallback",
        },
    }
    
    print(f"🎯 Generated LLM-Enhanced Report:")
    print(f"   Primary: {data['primary_fertilizer']['name']} ({data['primary_fertilizer']['amount_kg']}kg) - {data['cost_estimate']['primary']}")
    print(f"   Secondary: {data['secondary_fertilizer']['name']} ({data['secondary_fertilizer']['amount_kg']}kg) - {data['cost_estimate']['secondary']}")
    print(f"   Organics: {len(organics_blocks)} option(s) - {data['cost_estimate']['organics']}")
    print(f"   Confidence: {confidence_percent}%")
    print(f"   Total Cost: {data['cost_estimate']['total']}")
    print(f"   All three categories guaranteed: ✓")
    
    return data
