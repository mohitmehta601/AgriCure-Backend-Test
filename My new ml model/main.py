from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
from typing import Any, Dict

from predictor import load_default, FertilizerRecommender
from llm import generate_recommendation_report


app = Flask(__name__)


# Lazy-load trained model to keep startup fast and avoid import errors if not trained yet
_recommender: FertilizerRecommender | None = None


def get_recommender() -> FertilizerRecommender:
    global _recommender
    if _recommender is None:
        _recommender = load_default()
    return _recommender


@app.route("/")
def home():
    return render_template("plantindex.html")


@app.route("/Model1")
def Model1():
    return render_template("Model1.html")


@app.route("/Detail")
def Detail():
    return render_template("Detail.html")


def _parse_float(val: Any) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


@app.route("/predict", methods=["POST"])
def predict():
    # Read inputs from form (supporting both new and legacy field names)
    temp = _parse_float(request.form.get("Temperature") or request.form.get("temp"))
    humi = _parse_float(request.form.get("Humidity") or request.form.get("humid"))
    mois = _parse_float(request.form.get("Moisture") or request.form.get("mois"))
    soil = request.form.get("Soil_Type") or request.form.get("soil")
    crop = request.form.get("Crop") or request.form.get("crop")
    nitro = _parse_float(request.form.get("Nitrogen") or request.form.get("nitro"))
    phosp = _parse_float(request.form.get("Phosphorus") or request.form.get("phos"))
    pota = _parse_float(request.form.get("Potassium") or request.form.get("pota"))
    ph_val = _parse_float(request.form.get("pH"))
    sowing_date = request.form.get("Sowing_Date")
    field_size = _parse_float(request.form.get("Field_Size"))
    field_unit = request.form.get("Field_Unit", "hectares")
    skip_llm = request.form.get("skip_llm") == "true"

    # Handle legacy soil type mapping (if numeric values are sent)
    if soil and soil.isdigit():
        soil_map = {
            "0": "Black",
            "1": "Clayey", 
            "2": "Loamy",
            "3": "Red",
            "4": "Sandy"
        }
        soil = soil_map.get(soil, soil)

    # Handle legacy crop type mapping (if numeric values are sent)
    if crop and crop.isdigit():
        crop_map = {
            "0": "Wheat",      # Updated to match dataset
            "1": "Cotton", 
            "2": "Groundnut",  # Updated to match dataset
            "3": "Maize",
            "4": "Millets", 
            "5": "Soybean",    # Updated to match dataset
            "6": "Rice",       # Updated to match dataset
            "7": "Pulses",
            "8": "Sugarcane", 
            "9": "Tea",        # Updated to match dataset
            "10": "Coffee"     # Updated to match dataset
        }
        crop = crop_map.get(crop, crop)

    missing = [
        k
        for k, v in {
            "Temperature": temp,
            "Humidity": humi,
            "Moisture": mois,
            "Soil_Type": soil,
            "Crop": crop,
            "Nitrogen": nitro,
            "Phosphorus": phosp,
            "Potassium": pota,
            "pH": ph_val,
        }.items()
        if v in (None, "")
    ]
    if missing:
        return render_template(
            "Model1.html",
            error=f"Missing/invalid inputs: {', '.join(missing)}",
        )

    features: Dict[str, Any] = {
        "Temperature": temp,
        "Humidity": humi,
        "Moisture": mois,
        "Soil_Type": soil,
        "Crop": crop,
        "Nitrogen": nitro,
        "Phosphorus": phosp,
        "Potassium": pota,
        "pH": ph_val,
    }

    # Run ML predictions
    try:
        recommender = get_recommender()
        preds, confs = recommender.predict(features)
    except Exception as e:
        return render_template("Model1.html", error=str(e))

    # Prepare base inputs for LLM (use defaults if optional fields not provided)
    base_inputs = {
        **features,
        "Sowing_Date": sowing_date or "2024-01-01",  # Default sowing date
        "Field_Size": field_size if field_size is not None else 1.0,  # Default 1 hectare
        "Field_Unit": field_unit or "hectares",
    }
    
    # Check if LLM should be disabled via environment variable or user choice
    disable_llm = os.getenv("DISABLE_LLM", "false").lower() == "true" or skip_llm
    
    # Try to generate LLM-enhanced report (always attempt unless explicitly disabled)
    if not disable_llm:
        try:
            print(f"Generating LLM report with ML predictions: {preds}")
            report = generate_recommendation_report(base_inputs, preds, confs)
            return render_template(
                "result.html",
                report=report,
                raw_predictions=preds,
                raw_confidences=confs,
            )
        except Exception as e:
            # Log the error but continue with fallback
            error_str = str(e)
            print(f"LLM generation failed: {error_str}")
            
            # Provide specific error messages for common issues
            if "insufficient_quota" in error_str or "429" in error_str or "quota" in error_str.lower():
                user_error = "Gemini API quota exceeded. Please check your billing or upgrade your plan."
            elif "GEMINI_API_KEY" in error_str:
                user_error = "Gemini API key not configured properly. Set GEMINI_API_KEY environment variable."
            elif "google-generativeai package" in error_str:
                user_error = "Google Generative AI package not installed properly."
            else:
                user_error = f"Advanced report temporarily unavailable: {error_str}"
            
            # Fall back to enhanced predictions view with extra info
            return render_template(
                "Model1.html",
                predictions=preds,
                confidences=confs,
                base_inputs=base_inputs,
                llm_error=user_error,
                show_basic_report=True
            )
    else:
        # LLM disabled, show enhanced predictions
        return render_template(
            "Model1.html",
            predictions=preds,
            confidences=confs,
            base_inputs=base_inputs,
            llm_error="AI report skipped by user choice." if skip_llm else "AI report disabled in configuration.",
            show_basic_report=True
        )

    # Otherwise render predictions only
    return render_template("Model1.html", predictions=preds, confidences=confs)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    try:
        recommender = get_recommender()
        preds, confs = recommender.predict(data)
        return jsonify({"predictions": preds, "confidences": confs})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
