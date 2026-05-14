# GCP VM Deployment Guide

This guide covers deploying the GPON MLOps Platform on a fresh GCP VM.

Target VM specs: `4 vCPU · 16 GB RAM · 60 GB SSD · Debian 12 or Ubuntu 22.04 LTS`

This version mirrors the flow you already validated manually:

- repo-local `venv`
- `make install`
- KIND `v0.31.0`
- Kubeflow Pipelines `2.15.0`
- `ingress-nginx` before KServe
- KServe `RawDeployment`

## Quick Start

```bash
git clone https://github.com/<your-org>/gpon-mlops.git
cd gpon-mlops

chmod +x setup_gcp.sh start_mlops.sh
./setup_gcp.sh
```

The setup script now handles:

- system prerequisites
- swap
- KIND cluster
- Kubeflow Pipelines
- cert-manager + ingress-nginx + KServe
- Kyverno plus starter `Audit` policies
- Docker Compose stack
- KIND/network ConfigMaps
- image loading
- MinIO buckets
- data generation, ingestion, local training, MLflow registration
- champion export to the stable MinIO deployment path
- KServe `InferenceService` apply

## What Gets Installed

| Component | Where | Version |
|-----------|-------|---------|
| Docker Engine | VM system | latest |
| kubectl | VM system | latest stable |
| KIND | VM system | `v0.31.0` |
| Helm | VM system | latest |
| Kubeflow Pipelines | KIND cluster | `2.15.0` |
| KServe | KIND cluster | `v0.14.0` |
| cert-manager | KIND cluster | `v1.14.4` |
| ingress-nginx | KIND cluster | latest KIND manifest |
| Kyverno | KIND cluster | `v1.16.2` |
| MLflow | Docker Compose | repo-managed |
| MinIO | Docker Compose | repo-managed |
| PostgreSQL | Docker Compose | repo-managed |
| Prometheus | Docker Compose | repo-managed |
| Grafana | Docker Compose | repo-managed |

## Important KServe Note

This project currently deploys the model with an `aws-cli` sidecar in
`deployment/kserve/inference_service.yaml`.

That means:

- you **do need** the `amazon/aws-cli:latest` image loaded into KIND
- you **do need** the `mlops-endpoints` ConfigMap populated in `kserve`
- you **do need** the stable model artifact at:
  - `s3://deployment-models/gpon-failure-predictor/champion/model.bst`
- you **do not need** KServe's storage initializer for this manifest

`setup_gcp.sh` now handles the champion export and InferenceService apply for you.

## Python Environment

The setup script creates and uses a repo-local virtual environment at `venv/`.
That matches your usual flow and avoids Debian system Python packaging issues.

If you want to work manually after setup:

```bash
source venv/bin/activate
make install
```

## Accessing the UIs

All services listen on `localhost` on the VM. Use an SSH tunnel from your laptop:

```bash
ssh -N \
  -L 5000:localhost:5000 \
  -L 8080:localhost:8080 \
  -L 9001:localhost:9001 \
  -L 3000:localhost:3000 \
  -L 9090:localhost:9090 \
  -L 8010:localhost:8010 \
  <YOUR_GCP_USER>@<VM_EXTERNAL_IP>
```

Then open:

- MLflow: [http://localhost:5000](http://localhost:5000)
- KFP UI: [http://localhost:8080](http://localhost:8080)
- MinIO Console: [http://localhost:9001](http://localhost:9001)
- Grafana: [http://localhost:3000](http://localhost:3000)
- Prometheus: [http://localhost:9090](http://localhost:9090)
- FastAPI docs when started: [http://localhost:8010/docs](http://localhost:8010/docs)

## Every Session

After a VM reboot or a fresh SSH session:

```bash
./start_mlops.sh
```

Optional flags:

```bash
./start_mlops.sh --no-rebuild
./start_mlops.sh --no-compose
```

The startup script now also restarts the KServe predictor deployment when it
exists, so the predictor picks up the refreshed MinIO/MLflow endpoint env vars
after ConfigMap updates.

## Verification

After setup:

```bash
kubectl get pods -n kubeflow
kubectl get inferenceservice -n kserve
kubectl get cpol
```

You should see:

- Kubeflow pods running
- `gpon-failure-predictor` InferenceService created
- Kyverno starter policies applied

## FastAPI Gateway

The setup script does not start the gateway as a background service yet.
Run it manually for now:

```bash
export FASTAPI_GATEWAY_API_KEY=gpon-dev-key
make gateway-run
```

## Monitoring Workflow

After KServe and FastAPI are up:

```bash
python simulate_trafic.py \
  --target fastapi \
  --gateway-url http://localhost:8010/predict \
  --api-key gpon-dev-key
```

```bash
python -m monitoring.metrics_exporter \
  --predictions data/predictions/latest.csv \
  --port 8000 \
  --interval 30 \
  --window-rows 300
```

```bash
python -m monitoring.drift_detection \
  --baseline data/processed/processed.csv \
  --current data/predictions/latest.csv \
  --output monitoring/reports/drift_report.html \
  --current-window-rows 500
```

## Partial Re-runs

```bash
SKIP_PREREQS=true SKIP_K8S=true SKIP_COMPOSE=true ./setup_gcp.sh
SKIP_PREREQS=true SKIP_K8S=true SKIP_COMPOSE=true SKIP_IMAGES=true ./setup_gcp.sh
SKIP_PREREQS=true SKIP_K8S=true SKIP_COMPOSE=true SKIP_IMAGES=true SKIP_DATA=true ./setup_gcp.sh
SKIP_DEPLOY=true ./setup_gcp.sh
```

## Manual Reference

If you want to stay close to the exact order you already tested, the automation
now follows this same sequence:

1. create the repo `venv`
2. run `make install`
3. create the KIND cluster
4. install KFP `2.15.0`
5. install `cert-manager`
6. install `ingress-nginx`
7. install KServe and switch to `RawDeployment`
8. run `make docker-up`
9. run `./start_mlops.sh`
10. generate data, ingest it, register the model, upload telemetry, compile the pipeline
11. promote the champion and apply the InferenceService

## What Is Still Manual

These are still intentionally manual for now:

- starting the FastAPI gateway
- port-forwarding KServe for direct local testing
- traffic simulation
- metrics exporter
- drift detection / retraining trigger execution

## Troubleshooting

- If `kubectl apply -f deployment/kserve/inference_service.yaml` succeeds but the predictor pod does not become ready, check:
  - `kubectl describe inferenceservice -n kserve gpon-failure-predictor`
  - `kubectl get pods -n kserve`
  - `kubectl logs -n kserve deploy/gpon-failure-predictor-predictor -c model-downloader`
- If the downloader fails with S3 errors, verify the stable champion object exists:
  - `s3://deployment-models/gpon-failure-predictor/champion/model.bst`
- If Kyverno policy reports are empty at first, give the reports controller a little time, then run:
  - `kubectl get clusterpolicyreport`
  - `kubectl logs -n kyverno deploy/kyverno-reports-controller --tail=200`
