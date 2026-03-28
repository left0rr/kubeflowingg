#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Starting Docker Compose services ==="
cd "$SCRIPT_DIR"
make docker-up
sleep 10

echo "=== Connecting to KIND network ==="
docker network connect kind mlflow-server 2>/dev/null || true
docker network connect kind mlflow-minio 2>/dev/null || true
sleep 3

echo "=== Detecting IPs ==="
MLFLOW_IP=$(docker inspect mlflow-server \
  --format '{{range $k,$v := .NetworkSettings.Networks}}{{if eq $k "kind"}}{{$v.IPAddress}}{{end}}{{end}}')
MINIO_IP=$(docker inspect mlflow-minio \
  --format '{{range $k,$v := .NetworkSettings.Networks}}{{if eq $k "kind"}}{{$v.IPAddress}}{{end}}{{end}}')

echo "MLflow IP : $MLFLOW_IP"
echo "MinIO  IP : $MINIO_IP"

echo "=== Updating Kubernetes ConfigMap ==="
apply_endpoints_configmap() {
  local namespace="$1"

  if ! kubectl get namespace "$namespace" > /dev/null 2>&1; then
    echo "Namespace '$namespace' not found; skipping ConfigMap update"
    return
  fi

  kubectl create configmap mlops-endpoints \
    --from-literal=MLFLOW_TRACKING_URI=http://${MLFLOW_IP}:5000 \
    --from-literal=MLFLOW_S3_ENDPOINT_URL=http://${MINIO_IP}:9000 \
    --from-literal=MINIO_ENDPOINT_URL=http://${MINIO_IP}:9000 \
    --from-literal=AWS_ACCESS_KEY_ID=minio \
    --from-literal=AWS_SECRET_ACCESS_KEY=minio123 \
    -n "$namespace" \
    --dry-run=client -o yaml | kubectl apply -f -
}

apply_endpoints_configmap kubeflow
apply_endpoints_configmap kserve

echo "=== Reloading kfp-base image into KIND ==="
if docker image inspect kfp-base:latest > /dev/null 2>&1; then
  kind load docker-image kfp-base:latest --name mlops-cluster
  echo "kfp-base:latest loaded into KIND"
else
  echo "kfp-base:latest not found locally — building it now..."
  docker build -f Dockerfile.kfp-base -t kfp-base:latest .
  kind load docker-image kfp-base:latest --name mlops-cluster
  echo "kfp-base:latest built and loaded into KIND"
fi

if kubectl get deployment -n kserve gpon-failure-predictor-predictor > /dev/null 2>&1; then
  echo "=== Restarting KServe predictor to pick up updated endpoints ==="
  kubectl rollout restart deployment/gpon-failure-predictor-predictor -n kserve
  kubectl rollout status deployment/gpon-failure-predictor-predictor -n kserve --timeout=180s || true
fi

echo ""
echo "=== Done! ==="
echo "MLflow UI : http://localhost:5000"
echo "MinIO  UI : http://localhost:9001"
echo "KFP    UI : http://localhost:8080"
echo ""
echo "MLflow tracking URI for pods : http://${MLFLOW_IP}:5000"
echo "MinIO endpoint for pods      : http://${MINIO_IP}:9000"
