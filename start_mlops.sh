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

load_image_into_kind() {
  local image_name="$1"
  local cluster_name="$2"
  local kind_nodes=()

  mapfile -t kind_nodes < <(
    docker ps \
      --filter "label=io.x-k8s.kind.cluster=${cluster_name}" \
      --format '{{.Names}}'
  )

  if [ "${#kind_nodes[@]}" -eq 0 ]; then
    echo "No kind nodes found for cluster '${cluster_name}'"
    return 1
  fi

  local archive_file
  archive_file="$(mktemp "${TMPDIR:-/tmp}/kind-image-XXXXXX.tar")"
  docker save "$image_name" -o "$archive_file"

  for node in "${kind_nodes[@]}"; do
    echo "Importing ${image_name} into ${node}"
    docker exec -i "$node" ctr -n k8s.io images import < "$archive_file"
  done

  rm -f "$archive_file"
}

upload_raw_telemetry_to_minio() {
  local raw_dataset_path="data/raw/telemetry.csv"

  if [ ! -f "$raw_dataset_path" ]; then
    echo "Raw dataset not found at ${raw_dataset_path}; skipping MinIO upload"
    return
  fi

  echo "=== Uploading raw telemetry dataset to MinIO ==="
  docker cp "$raw_dataset_path" mlflow-minio:/tmp/telemetry.csv
  docker exec mlflow-minio /bin/sh -c \
    "mc alias set local http://127.0.0.1:9000 minio minio123 >/dev/null 2>&1 && \
     mc cp /tmp/telemetry.csv local/gpon-telemetry/raw/telemetry.csv >/dev/null"
  docker exec mlflow-minio rm -f /tmp/telemetry.csv
  echo "Uploaded ${raw_dataset_path} to s3://gpon-telemetry/raw/telemetry.csv"
}

echo "=== Reloading kfp-base image into KIND ==="
if docker image inspect kfp-base:latest > /dev/null 2>&1; then
  load_image_into_kind kfp-base:latest mlops-cluster
  echo "kfp-base:latest loaded into KIND via docker save + ctr import"
else
  echo "kfp-base:latest not found locally — building it now..."
  docker build -f Dockerfile.kfp-base -t kfp-base:latest .
  load_image_into_kind kfp-base:latest mlops-cluster
  echo "kfp-base:latest built and loaded into KIND via docker save + ctr import"
fi

upload_raw_telemetry_to_minio

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
