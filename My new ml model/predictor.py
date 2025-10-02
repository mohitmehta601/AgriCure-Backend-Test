import os
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "fertilizer_recommender.pkl")


class FertilizerRecommender:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}. Run train.py first.")
        artifact = joblib.load(model_path)
        self.features = artifact["features"]
        self.targets = artifact["targets"]
        self.models = artifact["models"]          # dict[target][model_name] -> pipeline
        self.label_encoders = artifact["label_encoders"]
        self.cv_scores = artifact.get("cv_scores", {})

    def _soft_vote(self, target: str, X: pd.DataFrame) -> Tuple[int, float]:
        """Return (y_enc_pred, confidence) using weighted soft voting across all available models."""
        model_dict: Dict[str, Any] = self.models[target]
        weights = self.cv_scores.get(target, {})

        # Get the total number of classes for this target from the label encoder
        n_classes = len(self.label_encoders[target].classes_)
        proba_sum = None
        total_w = 0.0

        for name, pipe in model_dict.items():
            w = float(weights.get(name, 1.0))
            try:
                proba = pipe.predict_proba(X)[0]  # shape: [n_classes_model]
                
                # Ensure proba has the correct shape by padding with zeros if necessary
                if len(proba) < n_classes:
                    # Pad with small values for missing classes
                    full_proba = np.full(n_classes, 1e-6)
                    full_proba[:len(proba)] = proba
                    proba = full_proba
                elif len(proba) > n_classes:
                    # Truncate if somehow we have more classes than expected
                    proba = proba[:n_classes]
                    
            except Exception:
                # fallback to hard prediction as degenerate proba
                yhat = pipe.predict(X)[0]
                # convert hard pred into one-hot with small smoothing
                proba = np.full(n_classes, 1e-6)
                if int(yhat) < n_classes:  # Ensure the predicted class is within bounds
                    proba[int(yhat)] = 1.0

            if proba_sum is None:
                proba_sum = w * proba
            else:
                proba_sum += w * proba
            total_w += w

        avg_proba = proba_sum / max(total_w, 1e-9)
        y_enc = int(np.argmax(avg_proba))
        conf = float(np.max(avg_proba))
        return y_enc, conf

    def predict(self, record: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Predict all targets with weighted soft-vote ensemble."""
        row = {k: record.get(k) for k in self.features}
        X = pd.DataFrame([row], columns=self.features)  # Ensure column order matches training

        predictions: Dict[str, str] = {}
        confidences: Dict[str, float] = {}

        for target in self.targets:
            le = self.label_encoders[target]
            y_enc, conf = self._soft_vote(target, X)
            label = le.inverse_transform(np.array([y_enc]))[0]
            predictions[target] = str(label)
            confidences[target] = conf

        return predictions, confidences


def load_default() -> FertilizerRecommender:
    return FertilizerRecommender(MODEL_PATH)

