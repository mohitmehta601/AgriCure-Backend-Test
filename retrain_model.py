#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# Base directories for models, data, templates, and static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
ML_DIR = os.path.join(BASE_DIR, "ml")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

def retrain_model():
    try:
        data_path = os.path.join(ML_DIR, "f2.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError("Dataset f2.csv not found")
        
        print(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        X = df[['Temparature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
        y = df['Fertilizer']
        
        soil_encoder = LabelEncoder()
        crop_encoder = LabelEncoder()
        fertilizer_encoder = LabelEncoder()
        
        X_encoded = X.copy()
        X_encoded['Soil_Type'] = soil_encoder.fit_transform(X['Soil_Type'])
        X_encoded['Crop_Type'] = crop_encoder.fit_transform(X['Crop_Type'])
        y_encoded = fertilizer_encoder.fit_transform(y)
        
        print(f"Unique soil types: {soil_encoder.classes_}")
        print(f"Unique crop types: {crop_encoder.classes_}")
        print(f"Unique fertilizers: {fertilizer_encoder.classes_}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        print("Training Random Forest model...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully with accuracy: {accuracy:.4f}")
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        with open(os.path.join(MODEL_DIR, "classifier.pkl"), "wb") as f:
            pickle.dump(model, f)
        
        with open(os.path.join(MODEL_DIR, "fertilizer.pkl"), "wb") as f:
            pickle.dump(fertilizer_encoder, f)
        
        print("Model and encoder saved successfully")
        
        test_input = np.array([[25, 78, 43, 4, 1, 22, 26, 38]])
        prediction = model.predict(test_input)
        fertilizer = fertilizer_encoder.classes_[prediction[0]]
        print(f"Test prediction: {fertilizer}")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Retraining fertilizer recommendation model...")
    success = retrain_model()
    if success:
        print("\n🎉 Model retrained successfully!")
    else:
        print("\n💥 Model retraining failed!")
