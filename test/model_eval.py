import shap
import joblib
import pandas as pd

# === Load trained model ===
model = joblib.load("models/churn_model.pkl")  # Adjust if needed

# === Load test features ===
X = pd.read_csv("data/cleaned_grocery_churn_data.csv")
if "churn" in X.columns:
    X = X.drop(columns=["churn"])  # Remove label if present

# === Ensure feature order and fill any missing expected features ===
expected_features = pd.read_csv("models/model_features.csv", header=None)[0].tolist()

# Fill missing columns with 0s before reordering
for col in expected_features:
    if col not in X.columns:
        X[col] = 0

# Reorder to match training feature order
X = X[expected_features]

# === Run SHAP explanations ===
explainer = shap.Explainer(model)
shap_values = explainer(X)

# === Plot SHAP summary ===
shap.summary_plot(shap_values, X)
