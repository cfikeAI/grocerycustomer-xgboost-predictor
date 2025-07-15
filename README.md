# Grocery Customer Churn Predictor (MLOps Pipeline)

### Overview
End-to-end MLOps project to predict grocery customer churn using **XGBoost**, **MLflow**, **DVC**, and **FastAPI**, deployed with **Docker**, **Azure AKS**, and CI/CD via **GitHub Actions**.

---

## Features
Train & track experiments with **MLflow**  
Data & model versioning with **DVC (Azure Blob Storage)**  
CI/CD pipeline: Auto-retrain → Push to remote → Deploy  
Serve predictions with **FastAPI** (Dockerized)  
Ready for **AKS deployment** (Terraform)

### Visual Diagram:
**Data Pipeline**

DVC for data versioning, using Azure Blob Storage as remote

**Model Training**

XGBoost for churn prediction

MLflow for experiment tracking and artifact logging

**CI/CD Pipeline**

GitHub Actions automates retraining and pushes updated artifacts to DVC remote

**Model Serving**

FastAPI application containerized with Docker

Image pushed to Azure ACR/AKS

**Deployment**

Deployed on Azure Kubernetes Service (AKS) for scalability

**Monitoring**

Prometheus for metrics collection

Grafana for real-time visualization


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
