import os
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Optional heavy learners
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


DATA_FILE = "New dataset 5111 rows.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fertilizer_recommender.pkl")

FEATURES: List[str] = [
    "Temperature","Humidity","Moisture","Soil_Type","Crop",
    "Nitrogen","Phosphorus","Potassium","pH",
]

TARGETS: List[str] = [
    "N_Status","P_Status","K_Status","Primary_Fertilizer","Secondary_Fertilizer",
    "Organic_1","Organic_2","Organic_3","pH_Amendment",
]


def build_preprocessor() -> ColumnTransformer:
    categorical = ["Soil_Type", "Crop"]
    numeric = [c for c in FEATURES if c not in categorical]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", "passthrough", numeric),
        ]
    )


def get_candidates() -> Dict[str, object]:
    models: Dict[str, object] = {}
    models["rf"] = RandomForestClassifier(n_estimators=400, random_state=42)

    if XGBClassifier is not None:
        models["xgb"] = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8,
            objective="multi:softprob", eval_metric="mlogloss",
            tree_method="hist", random_state=42,
        )
    if LGBMClassifier is not None:
        models["lgbm"] = LGBMClassifier(
            n_estimators=800, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, random_state=42,
        )
    if CatBoostClassifier is not None:
        models["catboost"] = CatBoostClassifier(
            depth=8, learning_rate=0.05, iterations=800, verbose=False, random_state=42
        )
    return models


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    # type fixes for numerics
    for c in ["Temperature","Humidity","Moisture","Nitrogen","Phosphorus","Potassium","pH"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # basic hygiene
    df = df.dropna(subset=FEATURES + TARGETS).reset_index(drop=True)

    preprocessor = build_preprocessor()

    X = df[FEATURES]
    models_package: Dict[str, Dict[str, Pipeline]] = {}
    label_encoders: Dict[str, LabelEncoder] = {}
    cv_scores: Dict[str, Dict[str, float]] = {}

    for target in TARGETS:
        print(f"\nTraining target: {target}")
        y = df[target].astype(str)

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        label_encoders[target] = le

        candidates = get_candidates()
        models_package[target] = {}
        cv_scores[target] = {}

        # Safe StratifiedCV (fallback to simple 3-fold when rare classes)
        unique, counts = np.unique(y_enc, return_counts=True)
        min_count = counts.min() if len(counts) else 0
        cv = 3 if min_count < 2 else StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Train & score all candidates, keep all
        for name, model in candidates.items():
            pipe = Pipeline([("preprocess", preprocessor), ("clf", model)])
            try:
                scores = cross_val_score(pipe, X, y_enc, cv=cv, scoring="accuracy")
                mean_acc = float(np.mean(scores))
                print(f" - {name} CV acc: {mean_acc:.4f}")
                cv_scores[target][name] = mean_acc
                pipe.fit(X, y_enc)
                models_package[target][name] = pipe
            except Exception as e:
                print(f" - {name} failed: {e}")

        if not models_package[target]:
            # absolute fallback
            print("   (all candidates failed; training fallback RF)")
            fallback = Pipeline([("preprocess", preprocessor),
                                 ("clf", RandomForestClassifier(n_estimators=400, random_state=42))])
            fallback.fit(X, y_enc)
            models_package[target]["rf"] = fallback
            cv_scores[target]["rf"] = 0.0

    os.makedirs(MODEL_DIR, exist_ok=True)
    artifact = {
        "features": FEATURES,
        "targets": TARGETS,
        "models": models_package,     # dict[target][model_name] -> pipeline
        "label_encoders": label_encoders,
        "cv_scores": cv_scores,       # dict[target][model_name] -> cv_acc
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
