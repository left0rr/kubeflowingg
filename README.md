# GPON Router Failure Prediction — End-to-End MLOps Platform

An end-to-end MLOps system that predicts **GPON router failures within the next 7 days** using telemetry data.
The platform demonstrates a **production-style ML lifecycle** including data processing, model training, experiment tracking, orchestration, deployment, monitoring, and automated retraining.

This project simulates a **telecom network operations environment** where routers emit telemetry such as optical power, temperature, voltage, and error counts.
A machine learning model predicts imminent failure so operations teams can proactively intervene.

---

# Project Goals

* Build a **realistic telecom predictive maintenance system**
* Demonstrate **modern MLOps architecture**
* Implement **automated model retraining**
* Provide **monitoring and drift detection**
* Deploy a **scalable inference service**

The platform runs locally using a lightweight Kubernetes environment.

---

# System Architecture

```
                         GitHub Repository
                               │
                               ▼
                        CI/CD Pipeline
                     (lint, test, build)
                               │
                               ▼
                     Docker Container Images
                               │
                               ▼
                         KIND Cluster
                 ┌─────────────────────────┐
                 │   Kubeflow Pipelines    │
                 │                         │
                 │  Ingest → Train → Eval  │
                 │                         │
                 └──────────┬──────────────┘
                            │
                            ▼
                        MLflow Server
                (Experiment Tracking + Registry)
                            │
                            ▼
                          MinIO
               (Datasets + Model Artifacts)
                            │
                            ▼
                         KServe
                 (Model Inference Service)
                            │
                            ▼
                      Monitoring Stack
            Evidently + Prometheus + Grafana
                            │
                            ▼
                     Drift Detection
                            │
                            ▼
                     Automated Retraining
                     (Kubeflow Pipeline)
```

---

# MLOps Lifecycle

```
Data → Feature Engineering → Training → Evaluation
     → Model Registry → Deployment → Monitoring
     → Drift Detection → Retraining → Redeploy
```

This **closed loop** ensures the system remains accurate as network conditions change.

---

# Technology Stack

## Infrastructure

* Docker
* KIND Kubernetes cluster
* Docker Compose

## MLOps Platform

* Kubeflow Pipelines
* MLflow
* MinIO
* KServe

## Machine Learning

* XGBoost
* Scikit-learn
* Pandas
* NumPy

## Monitoring

* Evidently AI
* Prometheus
* Grafana

## Backend / APIs

* FastAPI
* Uvicorn
* Pydantic

## DevOps

* GitHub Actions
* Makefile automation

---

# Repository Structure

```
repo-root/
│
├── src/
│   ├── data/
│   │   ├── ingest.py
│   │   ├── validation.py
│   │   └── feature_engineering.py
│   │
│   └── training/
│       ├── train_xgboost.py
│       ├── evaluate.py
│       └── register_model.py
│
├── pipelines/
│   ├── pipeline_components.py
│   ├── kubeflow_pipeline.py
│   └── retraining_pipeline.py
│
├── deployment/
│   └── kserve/
│       └── inference_service.yaml
│
├── monitoring/
│   ├── drift_detection.py
│   ├── metrics_exporter.py
│   └── retraining_trigger.py
│
├── infrastructure/
│   └── docker-compose.yml
│
├── scripts/
│   └── scheduler.py
│
├── .github/workflows/
│   └── ci-cd.yml
│
├── requirements.txt
├── Makefile
└── README.md
```

---

# Dataset Description

The dataset contains simulated **GPON router telemetry metrics**.

## Features

| Feature                    | Description                      |
| -------------------------- | -------------------------------- |
| Optical_RX_Power_dBm       | Received optical signal strength |
| Optical_TX_Power_dBm       | Transmitted optical power        |
| Temperature_C              | Device temperature               |
| Voltage_mV                 | Device voltage                   |
| Bias_Current_mA            | Laser diode bias current         |
| Interface_Error_Count      | Network interface errors         |
| Reboot_Count_Last_7D       | Reboots in last week             |
| Connected_Devices          | Number of clients connected      |
| Device_Age_Days            | Router age                       |
| Maintenance_Count_Last_30D | Maintenance operations           |

## Target

```
Failure_In_7_Days
```

Binary classification:

```
1 = router failure predicted
0 = normal operation
```

---

# Machine Learning Model

The primary model is:

```
XGBoost Gradient Boosted Trees
```

Why XGBoost:

* Robust to tabular data
* Handles nonlinear relationships
* High performance for structured telemetry
* Widely used in industry

Evaluation metrics include:

* AUC
* Precision
* Recall
* F1-Score

---

# Data Processing Pipeline

1. Load dataset
2. Validate schema with Pydantic
3. Feature engineering
4. Normalize voltage
5. Compute rolling optical power statistics
6. Save processed dataset

---

# Training Pipeline

The Kubeflow pipeline performs:

```
Data Ingestion
     ↓
Feature Engineering
     ↓
Train XGBoost Model
     ↓
Evaluate Model
     ↓
Log Experiments
     ↓
Register Model
```

Artifacts and metrics are stored in MLflow.

---

# Model Deployment

The trained model is deployed using **KServe**.

Inference endpoint example:

```
POST /v1/models/router-failure:predict
```

Example request:

```json
{
  "Optical_RX_Power_dBm": -19.2,
  "Temperature_C": 70,
  "Voltage_V": 3.3,
  "Reboot_Count_Last_7D": 3
}
```

Example response:

```json
{
  "failure_probability": 0.82
}
```

---

# Monitoring and Observability

Monitoring is handled by three systems.

## Evidently

Detects:

* data drift
* feature drift
* distribution shifts

Generates HTML drift reports.

---

## Prometheus

Collects metrics:

* inference latency
* prediction counts
* failure prediction ratio

---

## Grafana

Provides dashboards showing:

* prediction trends
* system performance
* model metrics

---

# Automated Retraining

When drift is detected:

1. Monitoring job runs drift analysis
2. If drift exceeds threshold
3. Kubeflow retraining pipeline is triggered
4. New model is trained
5. Model performance is evaluated
6. Model is promoted only if performance improves

Example policy:

```
if new_auc > production_auc:
    promote_model
```

---

# CI/CD Pipeline

Continuous integration runs on every push.

Steps:

```
Install dependencies
Run linting
Run tests
Build containers
```

This ensures code quality before deployment.

---

# Local Setup Guide

## 1 Install Dependencies

```
make install
```

---

## 2 Start Infrastructure

```
docker-compose up -d
```

This launches:

* MLflow
* PostgreSQL
* MinIO

---

## 3 Start Kubernetes Cluster

```
kind create cluster
```

---

## 4 Deploy Kubeflow Pipelines

Apply pipeline configuration.

---

## 5 Run Training Pipeline

Compile and upload pipeline:

```
python pipelines/kubeflow_pipeline.py
```

---

## 6 Deploy Model

Apply KServe service:

```
kubectl apply -f deployment/kserve/inference_service.yaml
```

---

# Example Workflow

```
Router Telemetry
        │
        ▼
Dataset Stored in MinIO
        │
        ▼
Kubeflow Training Pipeline
        │
        ▼
MLflow Model Registry
        │
        ▼
KServe Deployment
        │
        ▼
Predictions for Router Failure
        │
        ▼
Monitoring + Drift Detection
        │
        ▼
Automatic Retraining
```

---

# Key MLOps Concepts Demonstrated

* End-to-End ML lifecycle
* Experiment tracking
* Model versioning
* Data drift monitoring
* Automated retraining
* Containerized infrastructure
* Kubernetes ML deployment

---

# Future Improvements

Possible extensions:

* Feature Store integration
* Canary model deployment
* Online learning
* Real telemetry ingestion via Kafka
* Distributed hyperparameter tuning

---

# Author

MLOps internship project focused on **telecom network reliability prediction**.

---

# License

MIT License
