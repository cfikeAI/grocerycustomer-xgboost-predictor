# Grocery Customer Churn Predictor (MLOps Pipeline)

### Overview
End-to-end MLOps project to predict grocery customer churn using **XGBoost**, **MLflow**, **DVC**, and **FastAPI**, deployed with **Docker**, **Azure AKS**, and CI/CD via **GitHub Actions**.

---

## Features
✔ Train & track experiments with **MLflow**  
✔ Data & model versioning with **DVC (Azure Blob Storage)**  
✔ CI/CD pipeline: Auto-retrain → Push to remote → Deploy  
✔ Serve predictions with **FastAPI** (Dockerized)  
✔ Ready for **AKS deployment** (Terraform)

## **Architecture**

Data → DVC → Train (XGBoost) → MLflow Logging → Model Registry
↓
FastAPI + Docker → Azure Container Registry → Azure Kubernetes Service
↓
Monitoring: Prometheus + Grafana


## **Setup Instructions**

### 1. Clone Repo
git clone https://github.com/cfikeAI/grocerycustomer-xgboost-predictor.git
cd grocerycustomer-xgboost-predictor


### 2. Restore Data and Model
pip install -r requirements.txt
dvc pull  # Downloads dataset & model from Azure Blob

### 3. Train locally 
python train_model.py

### 4. Run FastAPI locally
uvicorn api.main:app --reload --port 8000
