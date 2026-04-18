# GPON Router Failure Prediction — End-to-End MLOps Platform

An end-to-end MLOps system that predicts **GPON router failures within the next 7 days** using telemetry data.
The platform demonstrates a **production-style ML lifecycle** including data processing, model training, experiment tracking, orchestration, deployment, monitoring, and automated retraining.

This project simulates a **telecom network operations environment** where routers emit telemetry such as optical power, temperature, voltage, and error counts.
A machine learning model predicts imminent failure so operations teams can proactively intervene.

---

## Project Status

| Phase | Component | Status |
|-------|-----------|--------|
| Data | Ingestion + Pydantic validation | ✅ Done |
| Data | Feature engineering | ✅ Done |
| Training | XGBoost classifier | ✅ Done |
| Training | MLflow experiment tracking + registry | ✅ Done |
| Orchestration | Kubeflow Pipelines (4-stage) | ✅ Done |
| Deployment | KServe RawDeployment on KIND | ✅ Done |
| Monitoring | Prometheus metrics exporter | ✅ Done |
| Monitoring | Traffic simulator → KServe | ✅ Done |
| Monitoring | Evidently drift detection | ✅ Done |
| Monitoring | Grafana dashboards | ✅ Done |
| Retraining | Drift-triggered retraining trigger foundation | 🔄 In Progress |
| Retraining | Champion promotion helper | ✅ Done |
| Security | FastAPI inference gateway foundation | 🔄 In Progress |
| Security | OPA / Kyverno policy enforcement | 📋 Planned |
| Security | SPIFFE / SPIRE workload identity | 📋 Future |
| Features | Feast feature store | 📋 Future |
| Quality | Great Expectations data contracts | 📋 Future |

---

## System Architecture

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
                 │  Ingest → Train →       │
                 │  Evaluate → Register    │
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
                         FastAPI
             (Auth + Guardrails + Audit Logs)
                            │
                            ▼
                         KServe
                 (Model Inference Service)
                            │
                            ▼
                      Monitoring Stack
      Evidently + Prometheus + Grafana + Node Exporter
                            │
                            ▼
                     Drift Detection
                            │
                            ▼
                 Retraining Trigger Decision
                            │
                            ▼
               Optional Kubeflow Retraining Run
```

---

## MLOps Lifecycle

```
Data → Feature Engineering → Training → Evaluation
     → Model Registry → Deployment → Monitoring
     → Drift Detection → Retraining → Redeploy
```

This **closed loop** ensures the system remains accurate as network conditions change.

---

## Technology Stack

### Infrastructure

| Tool | Purpose |
|------|---------|
| Docker | Containerisation |
| KIND | Local Kubernetes cluster |
| Docker Compose | MLflow + MinIO + PostgreSQL + Prometheus |

### MLOps Platform

| Tool | Purpose |
|------|---------|
| Kubeflow Pipelines v2 | Pipeline orchestration (4 stages) |
| MLflow 2.12 | Experiment tracking + model registry |
| MinIO | S3-compatible artifact store |
| KServe v0.14 | Model inference serving (RawDeployment) |

### Machine Learning

| Tool | Purpose |
|------|---------|
| XGBoost 2.0 | Binary failure classifier |
| Scikit-learn | Evaluation metrics |
| Pandas | Data processing |
| Evidently AI | Data + prediction drift detection |

### Monitoring

| Tool | Purpose |
|------|---------|
| Prometheus | Metrics collection (scrapes port 8000) |
| Grafana | Dashboards for predictions, drift, and system health |
| Node Exporter | Host CPU, memory, and filesystem metrics |
| prometheus_client | Python metrics exporter |

### Backend / APIs

| Tool | Purpose |
|------|---------|
| FastAPI | Inference gateway (planned) |
| Pydantic | Input validation schema |

### DevOps

| Tool | Purpose |
|------|---------|
| GitHub Actions | CI/CD (lint + test) |
| flake8 | Linting |
| pytest | Testing |
| Makefile | Automation |

---

## Repository Structure

```
repo-root/
│
├── src/
│   ├── api/
│   │   └── inference_gateway.py   # FastAPI gateway in front of KServe
│   │
│   ├── data/
│   │   ├── ingest.py               # Data loading + Pydantic validation
│   │   ├── validation.py           # TelemetryRecord schema
│   │   └── feature_engineering.py # Voltage normalisation + rolling features
│   │
│   └── training/
│       ├── train_xgboost.py        # Chronological split + XGBoost training
│       ├── evaluate.py             # AUC, F1, precision, recall
│       └── register_model.py       # MLflow experiment + model registration
│
├── pipelines/
│   ├── pipeline_components.py      # KFP v2 components (ingest/train/eval/register)
│   ├── kubeflow_pipeline.py        # Pipeline definition + ConfigMap injection
│   └── pipeline.yaml              # Compiled pipeline (auto-generated)
│
├── deployment/
│   └── kserve/
│       └── inference_service.yaml  # KServe InferenceService (RawDeployment)
│
├── monitoring/
│   ├── drift_detection.py          # Evidently DataDriftPreset report
│   ├── metrics_exporter.py         # Prometheus gauge (prediction_failure_ratio)
│   ├── retraining_trigger.py       # Drift threshold → KFP retraining trigger
│   └── prometheus.yml              # Prometheus scrape config
│
├── retraining/
│   ├── README.md                   # Retraining foundations and workflow docs
│   └── retraining_config.example.yaml
│
├── security/
│   └── fastapi/
│       └── README.md               # Gateway architecture and local run guide
│
├── scripts/
│   ├── generate_data.py            # Synthetic GPON telemetry generator
│   └── scheduler.py               # Periodic drift check scheduler (planned)
│
├── infrastructure/
│   └── docker-compose.yml          # MLflow + MinIO + PostgreSQL + Prometheus
│
├── .github/workflows/
│   └── ci-cd.yml                  # Lint + test on push to main
│
├── Dockerfile.kfp-base             # KFP component base image (all deps baked in)
├── start_mlops.sh                  # Session startup script (IPs + ConfigMap)
├── simulate_trafic.py              # Traffic simulator → KServe + CSV logging
├── requirements.txt
├── Makefile
└── README.md
```

---

## Dataset Description

Simulated **GPON router telemetry** at 3-hour intervals across 500 devices over 30 days.

### Features

| Feature | Description | Range |
|---------|-------------|-------|
| Optical_RX_Power_dBm | Received optical signal strength | -40 to 0 dBm |
| Optical_TX_Power_dBm | Transmitted optical power | -10 to 10 dBm |
| Temperature_C | Device temperature | -40 to 125 °C |
| Voltage_V | Device voltage (normalised from mV) | 0 to 5 V |
| Bias_Current_mA | Laser diode bias current | 0 to 200 mA |
| Interface_Error_Count | Cumulative interface errors | ≥ 0 |
| Reboot_Count_Last_7D | Reboots in last week | ≥ 0 |
| Connected_Devices | Number of connected clients | ≥ 0 |
| Device_Age_Days | Router age since deployment | ≥ 0 |
| Maintenance_Count_Last_30D | Maintenance operations | ≥ 0 |

### Target

```
Failure_In_7_Days  →  1 = failure predicted,  0 = normal operation
```

Class distribution: ~15% positive (failure), ~85% negative.

---

## Model Performance

Evaluated on a chronological 20% test split (24,000 rows):

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9980 |
| Precision | 0.9048 |
| Recall | 0.8487 |
| F1 Score | 0.8759 |
| Positive rate | 3.7% |

---

## Kubeflow Pipeline

Four sequential stages compiled to `pipelines/pipeline.yaml`:

```
Ingest & Validate Telemetry
        ↓
Train XGBoost Classifier
        ↓
Evaluate Model & Quality Gate  (AUC ≥ 0.75 required)
        ↓
Register Model in MLflow
```

All environment variables (MLflow URI, MinIO endpoint, credentials) are
injected from a Kubernetes ConfigMap `mlops-endpoints` — no hardcoded IPs.

---

## Deployment

The trained model is served via **KServe v0.14 in RawDeployment mode** (no Istio/Knative required).
For operator-facing traffic, a FastAPI gateway now sits in front of KServe so
we can preserve router metadata, apply auth, and centralise logging without
polluting the model feature vector.

Key design decisions for local KIND setup:

- `protocolVersion` and `runtimeVersion` removed — causes MLServer routing if present
- Sidecar `model-downloader` container (aws-cli) fetches `model.bst` from MinIO
  into a shared `emptyDir` volume instead of relying on KServe storage initializer
- `imagePullPolicy: Never` on all containers — images pre-loaded into KIND via `docker save`
- Namespace-scoped `ServingRuntime` overrides cluster-scoped one for pull policy control

### Inference endpoint

```
POST /v1/models/gpon-failure-predictor:predict
```

Example request (feature order must match training):

```json
{
  "instances": [[
    -18.5,   // Optical_RX_Power_dBm
    2.3,     // Optical_TX_Power_dBm
    42.1,    // Temperature_C
    35.2,    // Bias_Current_mA
    12,      // Interface_Error_Count
    1,       // Reboot_Count_Last_7D
    4,       // Connected_Devices
    730,     // Device_Age_Days
    0,       // Maintenance_Count_Last_30D
    3.3      // Voltage_V
  ]]
}
```

Example response:

```json
{"predictions": [0.0007]}
```

---

## Monitoring

### Prometheus metrics

`metrics_exporter.py` exposes `prediction_failure_ratio` as a Prometheus Gauge on `:8000/metrics`.
Prometheus scrapes it every 10 seconds via `host.docker.internal:8000`.

### Traffic simulation

`simulate_trafic.py` now targets the FastAPI gateway by default and sends:

- router metadata (`device_id`, serial number, Tunisian telecom number)
- model features
- simulation context (`source_mode`, drift profile, drift status)

The FastAPI gateway forwards only the feature vector to KServe, then writes:

- `data/predictions/latest.csv` for metrics exporter + drift detection
- `data/predictions/prediction_events.csv` for alerting/audit workflows
- `data/feedback/labeled_feedback.csv` when simulated ground-truth labels are available

### Drift detection

`drift_detection.py` uses Evidently AI `DataDriftPreset` to compare
`data/processed/processed.csv` (baseline) against `data/predictions/latest.csv` (production).

```bash
python -m monitoring.drift_detection \
    --baseline data/processed/processed.csv \
    --current  data/predictions/latest.csv \
    --output   monitoring/reports/drift_report.html
```

Exits with code 1 when dataset-level drift is detected — designed to gate CI/CD pipelines.

### Grafana dashboards

Grafana is provisioned from `monitoring/grafana/` and reads from Prometheus.
The current dashboard covers:

- `prediction_failure_ratio` over time
- current sample window size
- drift alert status and drifted feature count
- host CPU, memory, and root filesystem usage
- exporter health and drift-check freshness

---

## Automated Retraining Foundation

```
drift_detection.py writes HTML report + Prometheus metrics
        ↓
retraining_trigger.py reads drift metrics
        ↓
trigger rules are evaluated in dry-run mode by default
        ↓
if rules pass and --submit is enabled:
        ↓
Kubeflow pipeline run is submitted through the KFP SDK
        ↓
quality gate still decides whether the candidate can move forward
        ↓
promote_champion.py can be used after successful validation
```

This is intentionally a foundation, not a fully hands-off closed loop yet.

The current retraining trigger adds:

- transparent drift thresholds
- cooldown protection to avoid repeated submissions
- dry-run mode for safe tuning
- Prometheus textfile metrics for future Grafana panels

Starter commands:

```bash
python -m monitoring.retraining_trigger \
  --config retraining/retraining_config.example.yaml
```

```bash
python -m monitoring.retraining_trigger \
  --config retraining/retraining_config.example.yaml \
  --submit
```

---

## Planned Security Additions

### FastAPI Inference Gateway

A lightweight FastAPI service sits in front of KServe adding:

- API key authentication via `X-API-Key` header
- Request rate limiting
- Input schema validation for router metadata + model features
- Request logging for audit trail and future alerting
- Single stable endpoint regardless of KServe pod IP changes
- Preservation of router identity fields outside the model feature vector

### OPA / Kyverno — Kubernetes Policy Enforcement

Enforce cluster-wide policies:

- All pods must declare resource limits
- Only images from approved registries (`kserve/`, `kfp-base`)
- No privileged containers in `kserve` namespace
- ConfigMaps containing credentials must be namespaced

### SPIFFE / SPIRE — Workload Identity (future)

Cryptographic workload identity for pod-to-pod and pod-to-service authentication.
Each pod receives an X.509 SVID (SPIFFE Verifiable Identity Document).
Eliminates the need for static credentials in ConfigMaps for inter-service communication.

---

## Planned Feature Store — Feast (future)

Centralise feature definitions to eliminate the current duplication between:

- `src/data/feature_engineering.py` (local training)
- `pipelines/pipeline_components.py` ingestion component (KFP)
- `simulate_trafic.py` (inference time)

With Feast, a single `FeatureView` definition drives all three paths,
guaranteeing training/serving feature parity.

---

## Local Setup Guide

### 1. Every session — start infrastructure

```bash
./start_mlops.sh
```

This starts Docker Compose services, connects them to the KIND network,
updates the Kubernetes ConfigMap with current IPs, and reloads the kfp-base image.

### 2. First time only — generate and process data

```bash
mkdir -p data/raw data/processed
python scripts/generate_data.py
python -m src.data.ingest --input data/raw/telemetry.csv
```

### 3. Train and register model locally

```bash
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123

python -m src.training.register_model \
  --input data/processed/processed.csv \
  --tracking-uri http://localhost:5000
```

### 4. Run the Kubeflow pipeline

```bash
# Compile
python -m pipelines.kubeflow_pipeline

# Upload pipeline.yaml to KFP UI at http://localhost:8080
# Create a run with input_csv_path = s3://gpon-telemetry/raw/telemetry.csv
```

### 5. Port-forward KServe for inference

```bash
kubectl port-forward -n kserve \
  svc/gpon-failure-predictor-predictor 8085:80 &
```

### 6. Start the FastAPI gateway

```bash
export FASTAPI_GATEWAY_API_KEY=gpon-dev-key
uvicorn src.api.inference_gateway:app --host 0.0.0.0 --port 8010
```

### 7. Run traffic simulator

```bash
python simulate_trafic.py \
  --target fastapi \
  --gateway-url http://localhost:8010/predict \
  --api-key gpon-dev-key
```

### 8. Run metrics exporter

```bash
python -m monitoring.metrics_exporter \
  --predictions data/predictions/latest.csv \
  --port 8000 \
  --interval 30
```

### 9. Run drift detection

```bash
python -m monitoring.drift_detection \
    --baseline data/processed/processed.csv \
    --current  data/predictions/latest.csv \
    --output   monitoring/reports/drift_report.html
```

### 10. Evaluate retraining trigger in dry-run mode

```bash
python -m monitoring.retraining_trigger \
  --config retraining/retraining_config.example.yaml
```

### 11. Submit a retraining run when ready

```bash
python -m monitoring.retraining_trigger \
  --config retraining/retraining_config.example.yaml \
  --submit
```

---

## CI/CD Pipeline

Runs on every push to `main`:

```
Install dependencies → flake8 lint → pytest
```

---

## Known Issues and Design Notes

- MinIO IP on KIND network changes after laptop restart — `start_mlops.sh` handles this automatically
- KServe storage initializer not injected in RawDeployment with custom ServingRuntime — workaround uses aws-cli sidecar
- `kind load docker-image` fails for multi-platform OCI images — use `docker save` + `ctr import` instead
- `protocolVersion: v2` in InferenceService routes to MLServer (Seldon) not native xgbserver — always omit it

---

## Author

MLOps internship project — telecom network reliability prediction using GPON telemetry.

---

## License

MIT License
