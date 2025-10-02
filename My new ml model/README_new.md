# Fertilizer Recommendation App

Overview

- Train ML models on `New dataset 5111 rows.csv` to predict:
  N_Status, P_Status, K_Status, Primary_Fertilizer, Secondary_Fertilizer,
  Organic_1, Organic_2, Organic_3, pH_Amendment.
- Algorithms considered per target: XGBoost, LightGBM, CatBoost, RandomForest. The best model per target is chosen by CV.
- Flask app exposes a form to enter inputs and get predictions.
- Optionally, the app calls a Gemini LLM to turn the predictions + sowing date + field size into a structured, human‑friendly report.

Setup

1. Python env

   - `python -m venv .venv && .\.venv\Scripts\Activate.ps1`
   - `pip install -r requirements.txt`

2. Train models (produces `models/fertilizer_recommender.pkl`)

   - `python train.py`

3. Configure Gemini (optional but required for full report)

   - Set `GEMINI_API_KEY` in your env (don't hardcode keys):
     - PowerShell: `$env:GEMINI_API_KEY="AIzaSy..."`
     - or create a `.env` file with `GEMINI_API_KEY=AIzaSy...`

4. Run app
   - `python main.py`
   - Open `http://127.0.0.1:5000/Model1`

Files

- `train.py` – trains per‑target pipelines and saves artifact.
- `predictor.py` – loads artifact and predicts for a single record.
- `llm.py` – calls Gemini to generate a structured recommendation JSON.
- `templates/Model1.html` – input form and basic prediction display.
- `templates/result.html` – styled report produced by LLM.

Notes

- Dataset column headers must match exactly. The training script validates columns and drops rows with missing values in used fields.
- If you change feature names or add categories, re‑train.
- The app uses the primary fertilizer's probability as the "confidence" badge when available.
