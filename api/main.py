from fastapi import FastAPI, Body
from pydantic import create_model
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="Grocery Customer Churn Prediction API")

# === Load model ===
model_path = os.path.join("models", "churn_model.pkl")
model = joblib.load(model_path)

# === Load features from CSV ===
features_path = os.path.join("models", "model_features.csv")
feature_names = pd.read_csv(features_path, header=None).iloc[:, 0].tolist()


#Dynamic pydantic model creation
feature_schema = {name: (float, ...) for name in feature_names}
CustomerFeatures = create_model("CustomerFeatures", **feature_schema)

print(f"Expected features: {len(feature_names)} → {feature_names}")


# === FastAPI prediction endpoint ===
@app.post("/predict")
def predict_churn(data: CustomerFeatures = Body(...)):  # type: ignore
    features = np.array([list(data.dict().values())])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    incoming = list(data.dict().keys())
    print(f"Received features: {len(incoming)} → {incoming}")


    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(round(probability, 4))
    }
