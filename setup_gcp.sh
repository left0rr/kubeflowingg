#!/usr/bin/env bash
# =============================================================================
# setup_gcp.sh — One-shot GCP VM setup for the GPON MLOps Platform
# =============================================================================
# Tested on: Debian 12 / Ubuntu 22.04 LTS
# Target VM:  4 vCPU · 16 GB RAM · 60 GB SSD
#
# What this script does (in order):
#   Phase 1  — System prerequisites  (Docker, kubectl, KIND, pip)
#   Phase 2  — Swap space            (4 GB swap guard-rail)
#   Phase 3  — KIND cluster          (uses kind-config.yaml)
#   Phase 4  — Kubeflow Pipelines    (backend 2.3.0)
#   Phase 5  — KServe + cert-manager (v0.14 / RawDeployment)
#   Phase 6  — Kyverno               (Audit-mode starter policies)
#   Phase 7  — Docker Compose stack  (MLflow, MinIO, Prometheus, Grafana)
#   Phase 8  — Network bridging      (compose ↔ KIND) + ConfigMaps
#   Phase 9  — Container images      (build kfp-base, pull + load KServe)
#   Phase 10 — MinIO buckets
#   Phase 11 — Data pipeline         (generate → ingest → train → register)
#   Phase 12 — Promote champion + deploy KServe model
#   Phase 13 — Summary
#
# Skip flags (set env var to "true" before running):
#   SKIP_PREREQS=true   ./setup_gcp.sh   # skip apt + binary installs
#   SKIP_K8S=true       ./setup_gcp.sh   # skip KIND + KFP + KServe + Kyverno
#   SKIP_COMPOSE=true   ./setup_gcp.sh   # skip docker-compose startup
#   SKIP_IMAGES=true    ./setup_gcp.sh   # skip image build + KIND load
#   SKIP_DATA=true      ./setup_gcp.sh   # skip data generation + training
#   SKIP_DEPLOY=true    ./setup_gcp.sh   # skip champion export + KServe deploy
#
# Usage:
#   chmod +x setup_gcp.sh
#   ./setup_gcp.sh
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()   { echo -e "${GREEN}[$(date +'%H:%M:%S')] ✔  $*${NC}"; }
info()  { echo -e "${CYAN}[$(date +'%H:%M:%S')] ℹ  $*${NC}"; }
warn()  { echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠  $*${NC}"; }
error() { echo -e "${RED}[$(date +'%H:%M:%S')] ✘  $*${NC}"; exit 1; }
step()  {
    echo -e "\n${BOLD}${CYAN}══════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $*${NC}"
    echo -e "${BOLD}${CYAN}══════════════════════════════════════${NC}\n"
}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

SKIP_PREREQS="${SKIP_PREREQS:-false}"
SKIP_K8S="${SKIP_K8S:-false}"
SKIP_COMPOSE="${SKIP_COMPOSE:-false}"
SKIP_IMAGES="${SKIP_IMAGES:-false}"
SKIP_DATA="${SKIP_DATA:-false}"
SKIP_DEPLOY="${SKIP_DEPLOY:-false}"

KIND_VERSION="v0.23.0"
KFP_VERSION="2.3.0"
KSERVE_VERSION="v0.14.0"
CERTMANAGER_VERSION="v1.15.0"
KYVERNO_VERSION="v1.16.2"

cmd_exists() { command -v "$1" &>/dev/null; }

wait_for_pods() {
    local ns="$1"; local label="$2"; local timeout="${3:-300}"
    info "Waiting for pods (ns=$ns, label=$label, timeout=${timeout}s)…"
    kubectl wait pod --for=condition=ready \
        -n "$ns" -l "$label" \
        --timeout="${timeout}s" \
        || error "Pods did not become ready within ${timeout}s (ns=$ns, label=$label)"
}

wait_for_deployment() {
    local ns="$1"; local dep="$2"; local timeout="${3:-300}"
    info "Waiting for deployment $ns/$dep…"
    kubectl rollout status deployment/"$dep" -n "$ns" --timeout="${timeout}s" \
        || error "Deployment $dep did not roll out within ${timeout}s"
}

phase_prereqs() {
    step "Phase 1 — System Prerequisites"

    sudo apt-get update -qq

    if ! cmd_exists docker; then
        info "Installing Docker Engine…"
        sudo apt-get install -y ca-certificates curl gnupg lsb-release
        sudo install -m 0755 -d /etc/apt/keyrings
        curl -fsSL "https://download.docker.com/linux/$(. /etc/os-release && echo "$ID")/gpg" \
            | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        sudo chmod a+r /etc/apt/keyrings/docker.gpg
        echo \
          "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
          https://download.docker.com/linux/$(. /etc/os-release && echo "$ID") \
          $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
          | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt-get update -qq
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
            docker-buildx-plugin docker-compose-plugin
        sudo systemctl enable --now docker
        log "Docker installed."
    else
        log "Docker already installed: $(docker --version)"
    fi

    if ! groups | grep -q docker; then
        sudo usermod -aG docker "$USER"
        warn "Added $USER to docker group. Re-executing script with new group…"
        exec sg docker -- bash "$0" "$@"
    fi

    if ! cmd_exists kubectl; then
        info "Installing kubectl…"
        KUBECTL_VERSION="$(curl -sL https://dl.k8s.io/release/stable.txt)"
        curl -sLO "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl"
        sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
        rm kubectl
        log "kubectl installed."
    else
        log "kubectl already installed."
    fi

    if ! cmd_exists kind; then
        info "Installing KIND ${KIND_VERSION}…"
        curl -sLo ./kind "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-amd64"
        chmod +x ./kind
        sudo mv ./kind /usr/local/bin/kind
        log "KIND installed: $(kind version)"
    else
        log "KIND already installed: $(kind version)"
    fi

    if ! cmd_exists helm; then
        info "Installing Helm…"
        curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
        log "Helm installed."
    else
        log "Helm already installed."
    fi

    info "Installing Python requirements…"
    sudo apt-get install -y python3-pip python3-dev build-essential -qq
    pip3 install --quiet -r "$REPO_DIR/requirements.txt"
    log "Python requirements installed."
}

phase_swap() {
    step "Phase 2 — Swap Space (4 GB guard-rail)"
    local swap_file="/swapfile"
    if swapon --show | grep -q "$swap_file"; then
        log "Swap already active."
        return
    fi
    if [ -f "$swap_file" ]; then
        log "Swap file exists, enabling…"
    else
        info "Creating 4 GB swap file at $swap_file…"
        sudo fallocate -l 4G "$swap_file"
        sudo chmod 600 "$swap_file"
        sudo mkswap "$swap_file"
    fi
    sudo swapon "$swap_file"
    grep -q "$swap_file" /etc/fstab \
        || echo "${swap_file} none swap sw 0 0" | sudo tee -a /etc/fstab
    log "Swap enabled: $(free -h | awk '/Swap/{print $2}')"
}

phase_kind_cluster() {
    step "Phase 3 — KIND Cluster"

    if kind get clusters 2>/dev/null | grep -q "mlops-cluster"; then
        log "KIND cluster 'mlops-cluster' already exists."
    else
        info "Creating KIND cluster from kind-config.yaml…"
        kind create cluster --config "$REPO_DIR/kind-config.yaml" --wait 120s
        log "KIND cluster created."
    fi

    kubectl cluster-info --context kind-mlops-cluster \
        || error "Cannot reach KIND cluster. Check Docker status."
    log "Cluster accessible."

    for ns in kubeflow kserve kyverno; do
        kubectl get namespace "$ns" &>/dev/null || kubectl create namespace "$ns"
    done
    log "Namespaces ensured: kubeflow, kserve, kyverno."
}

phase_kfp() {
    step "Phase 4 — Kubeflow Pipelines (backend ${KFP_VERSION})"

    if kubectl get deployment ml-pipeline -n kubeflow &>/dev/null; then
        log "KFP already installed — skipping."
        return
    fi

    info "Applying KFP cluster-scoped resources…"
    kubectl apply -k \
        "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${KFP_VERSION}" \
        || error "Failed to apply KFP cluster-scoped resources. Check internet access."

    kubectl wait --for condition=established --timeout=120s crd/applications.app.k8s.io \
        || warn "CRD wait timed out — continuing anyway."

    info "Applying KFP env/dev manifests…"
    kubectl apply -k \
        "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=${KFP_VERSION}"

    info "Waiting for KFP core deployments (this may take 5-10 minutes)…"
    for dep in ml-pipeline ml-pipeline-ui ml-pipeline-persistenceagent ml-pipeline-scheduledworkflow; do
        wait_for_deployment kubeflow "$dep" 600
    done

    info "Patching KFP UI service to NodePort 30080…"
    kubectl patch svc ml-pipeline-ui -n kubeflow \
        -p '{"spec":{"type":"NodePort","ports":[{"port":80,"targetPort":3000,"nodePort":30080}]}}' \
        || warn "Service patch failed — you may need to forward the port manually."

    log "KFP installed and accessible at http://localhost:8080"
}

phase_kserve() {
    step "Phase 5 — cert-manager + KServe (${KSERVE_VERSION})"

    if ! kubectl get namespace cert-manager &>/dev/null; then
        info "Installing cert-manager ${CERTMANAGER_VERSION}…"
        kubectl apply -f \
            "https://github.com/cert-manager/cert-manager/releases/download/${CERTMANAGER_VERSION}/cert-manager.yaml"
        info "Waiting for cert-manager webhooks to be ready…"
        wait_for_pods cert-manager "app=cert-manager" 180
        wait_for_pods cert-manager "app=webhook" 180
        sleep 15
        log "cert-manager ready."
    else
        log "cert-manager already installed."
    fi

    if kubectl get deployment kserve-controller-manager -n kserve &>/dev/null; then
        log "KServe already installed — skipping."
    else
        info "Installing KServe ${KSERVE_VERSION}…"
        kubectl apply -f \
            "https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve.yaml"
        kubectl apply -f \
            "https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve-cluster-resources.yaml" \
            || warn "kserve-cluster-resources apply returned non-zero — may be partial."

        wait_for_deployment kserve kserve-controller-manager 300
        log "KServe installed."
    fi

    info "Patching KServe to use RawDeployment as default mode…"
    kubectl patch configmap/inferenceservice-config -n kserve \
        --type=merge \
        -p '{"data":{"deploy":"{\"defaultDeploymentMode\":\"RawDeployment\"}"}}' \
        2>/dev/null || warn "Could not patch inferenceservice-config — may not exist yet."

    log "KServe configured for RawDeployment."
}

phase_kyverno() {
    step "Phase 6 — Kyverno (Audit-mode starter policies)"

    if kubectl get deployment kyverno-admission-controller -n kyverno &>/dev/null; then
        log "Kyverno already installed — skipping install."
    else
        info "Installing Kyverno ${KYVERNO_VERSION}…"
        kubectl apply -f \
            "https://github.com/kyverno/kyverno/releases/download/${KYVERNO_VERSION}/install.yaml"
        for dep in kyverno-admission-controller kyverno-background-controller kyverno-reports-controller kyverno-cleanup-controller; do
            if kubectl get deployment "$dep" -n kyverno &>/dev/null; then
                wait_for_deployment kyverno "$dep" 300
            fi
        done
        log "Kyverno installed."
    fi

    info "Applying starter Kyverno policies in Audit mode…"
    kubectl apply -f "$REPO_DIR/security/kyverno/policies/disallow-privileged-kserve.yaml"
    kubectl apply -f "$REPO_DIR/security/kyverno/policies/require-pod-resources-kserve.yaml"
    log "Kyverno policies applied."
}

phase_compose() {
    step "Phase 7 — Docker Compose MLflow Stack"

    info "Starting docker compose services…"
    docker compose -f "$REPO_DIR/infrastructure/docker-compose.yml" up -d --build
    info "Waiting 20 s for services to initialise…"
    sleep 20

    for svc in mlflow-server mlflow-minio mlflow-postgres grafana prometheus; do
        if docker ps --format '{{.Names}}' | grep -q "$svc"; then
            log "  ✔ $svc running"
        else
            warn "  $svc does not appear to be running — check: docker ps"
        fi
    done
}

phase_network() {
    step "Phase 8 — Network Bridging and ConfigMaps"

    info "Connecting compose containers to KIND network…"
    for container in mlflow-server mlflow-minio; do
        docker network connect kind "$container" 2>/dev/null \
            && log "  Connected $container to KIND network" \
            || log "  $container already on KIND network"
    done
    sleep 5

    info "Detecting container IPs on KIND network…"
    MLFLOW_IP=$(docker inspect mlflow-server \
        --format '{{range $k,$v := .NetworkSettings.Networks}}{{if eq $k "kind"}}{{$v.IPAddress}}{{end}}{{end}}')
    MINIO_IP=$(docker inspect mlflow-minio \
        --format '{{range $k,$v := .NetworkSettings.Networks}}{{if eq $k "kind"}}{{$v.IPAddress}}{{end}}{{end}}')

    if [ -z "$MLFLOW_IP" ] || [ -z "$MINIO_IP" ]; then
        error "Could not detect MLflow/MinIO IPs on KIND network. Check: docker network inspect kind."
    fi

    info "MLflow IP  : $MLFLOW_IP"
    info "MinIO  IP  : $MINIO_IP"

    for ns in kubeflow kserve; do
        kubectl get namespace "$ns" &>/dev/null || continue
        kubectl create configmap mlops-endpoints \
            --from-literal=MLFLOW_TRACKING_URI="http://${MLFLOW_IP}:5000" \
            --from-literal=MLFLOW_S3_ENDPOINT_URL="http://${MINIO_IP}:9000" \
            --from-literal=MINIO_ENDPOINT_URL="http://${MINIO_IP}:9000" \
            --from-literal=AWS_ACCESS_KEY_ID="minio" \
            --from-literal=AWS_SECRET_ACCESS_KEY="minio123" \
            -n "$ns" --dry-run=client -o yaml | kubectl apply -f -
        log "  ConfigMap applied to namespace: $ns"
    done

    export MLFLOW_IP MINIO_IP
    export MLFLOW_TRACKING_URI="http://localhost:5000"
    export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
    export MINIO_ENDPOINT_URL="http://localhost:9000"
    export AWS_ACCESS_KEY_ID="minio"
    export AWS_SECRET_ACCESS_KEY="minio123"
    log "Network bridging and ConfigMaps complete."
}

phase_images() {
    step "Phase 9 — Container Images"

    info "Building kfp-base:latest from Dockerfile.kfp-base…"
    docker build -f "$REPO_DIR/Dockerfile.kfp-base" -t kfp-base:latest "$REPO_DIR"
    log "kfp-base:latest built."

    info "Loading kfp-base:latest into KIND…"
    kind load docker-image kfp-base:latest --name mlops-cluster
    log "kfp-base:latest loaded into KIND."

    for image in "kserve/xgbserver:latest" "amazon/aws-cli:latest"; do
        info "Pulling $image…"
        docker pull "$image"
        info "Loading $image into KIND…"
        kind load docker-image "$image" --name mlops-cluster
        log "  $image loaded."
    done

    log "All required images loaded into KIND."
}

phase_minio_buckets() {
    step "Phase 10 — MinIO Buckets"

    info "Waiting for MinIO to be healthy…"
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:9000/minio/health/live" > /dev/null 2>&1; then
            log "MinIO is healthy."
            break
        fi
        sleep 3
        [ "$i" -eq 30 ] && error "MinIO did not become healthy in 90 s."
    done

    python3 - <<'PYEOF'
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123",
    region_name="us-east-1",
    config=Config(signature_version="s3v4"),
)

for bucket in ["mlflow-artifacts", "gpon-telemetry", "deployment-models"]:
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"  Bucket already exists: {bucket}")
    except ClientError:
        s3.create_bucket(Bucket=bucket)
        print(f"  Created bucket: {bucket}")
PYEOF
    log "MinIO buckets ready."
}

phase_data() {
    step "Phase 11 — Data Pipeline (generate → ingest → train → register)"

    export MLFLOW_TRACKING_URI="http://localhost:5000"
    export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
    export MINIO_ENDPOINT_URL="http://localhost:9000"
    export AWS_ACCESS_KEY_ID="minio"
    export AWS_SECRET_ACCESS_KEY="minio123"

    if [ -f "$REPO_DIR/data/raw/telemetry.csv" ]; then
        log "Raw telemetry already exists — skipping generation."
    else
        info "Generating synthetic GPON telemetry…"
        mkdir -p "$REPO_DIR/data/raw"
        python3 "$REPO_DIR/scripts/generate_data.py"
        log "Raw data generated."
    fi

    if [ -f "$REPO_DIR/data/processed/processed.csv" ]; then
        log "Processed data already exists — skipping ingestion."
    else
        info "Running ingestion pipeline…"
        python3 -m src.data.ingest \
            --input "$REPO_DIR/data/raw/telemetry.csv" \
            --output "$REPO_DIR/data/processed/processed.csv"
        log "Ingestion complete."
    fi

    info "Waiting for MLflow server to be ready…"
    for i in $(seq 1 40); do
        if curl -sf "http://localhost:5000/health" > /dev/null 2>&1; then
            log "MLflow is ready."
            break
        fi
        sleep 3
        [ "$i" -eq 40 ] && error "MLflow did not become ready in 120 s."
    done

    info "Training XGBoost model and registering in MLflow…"
    python3 -m src.training.register_model \
        --input "$REPO_DIR/data/processed/processed.csv" \
        --tracking-uri http://localhost:5000 \
        --experiment-name "gpon-failure-prediction" \
        --model-name "gpon-xgboost-classifier"
    log "Model trained and registered."

    info "Uploading raw telemetry to s3://gpon-telemetry/raw/telemetry.csv…"
    python3 - <<'PYEOF'
import boto3
from botocore.config import Config

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123",
    region_name="us-east-1",
    config=Config(signature_version="s3v4"),
)
s3.upload_file("data/raw/telemetry.csv", "gpon-telemetry", "raw/telemetry.csv")
print("  Uploaded telemetry.csv to s3://gpon-telemetry/raw/telemetry.csv")
PYEOF
    log "Raw data uploaded to MinIO."

    info "Compiling Kubeflow pipeline…"
    python3 -m pipelines.kubeflow_pipeline
    log "Pipeline compiled → pipelines/pipeline.yaml"
}

phase_deploy_model() {
    step "Phase 12 — Promote champion and deploy KServe model"

    export MLFLOW_TRACKING_URI="http://localhost:5000"
    export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
    export MINIO_ENDPOINT_URL="http://localhost:9000"
    export AWS_ACCESS_KEY_ID="minio"
    export AWS_SECRET_ACCESS_KEY="minio123"

    info "Promoting the current best registered model to the stable champion path…"
    python3 "$REPO_DIR/monitoring/promote_champion.py" --skip-rollout-restart
    log "Champion artifact synced to deployment-models."

    info "Applying the KServe InferenceService…"
    kubectl apply -f "$REPO_DIR/deployment/kserve/inference_service.yaml"

    info "Waiting for the KServe InferenceService to become Ready…"
    kubectl wait --for=condition=Ready inferenceservice/gpon-failure-predictor -n kserve --timeout=420s \
        && log "KServe InferenceService is Ready." \
        || warn "InferenceService did not report Ready within the timeout; inspect with: kubectl describe inferenceservice -n kserve gpon-failure-predictor"
}

phase_summary() {
    step "Setup Complete — Service Endpoints"

    MLFLOW_IP=$(docker inspect mlflow-server \
        --format '{{range $k,$v := .NetworkSettings.Networks}}{{if eq $k "kind"}}{{$v.IPAddress}}{{end}}{{end}}' \
        2>/dev/null || echo "unknown")
    MINIO_IP=$(docker inspect mlflow-minio \
        --format '{{range $k,$v := .NetworkSettings.Networks}}{{if eq $k "kind"}}{{$v.IPAddress}}{{end}}{{end}}' \
        2>/dev/null || echo "unknown")

    echo ""
    echo -e "${BOLD}  Local (on this VM)${NC}"
    echo "  ─────────────────────────────────────────────"
    echo "  MLflow UI        →  http://localhost:5000"
    echo "  MinIO Console    →  http://localhost:9001  (minio / minio123)"
    echo "  Grafana          →  http://localhost:3000  (admin / admin123)"
    echo "  Prometheus       →  http://localhost:9090"
    echo "  KFP UI           →  http://localhost:8080"
    echo ""
    echo -e "${BOLD}  From your local machine (SSH tunnel)${NC}"
    echo "  ─────────────────────────────────────────────"
    echo "  ssh -N -L 5000:localhost:5000 \\"
    echo "         -L 8080:localhost:8080 \\"
    echo "         -L 9001:localhost:9001 \\"
    echo "         -L 3000:localhost:3000 \\"
    echo "         -L 9090:localhost:9090 \\"
    echo "         <YOUR_GCP_USER>@<VM_EXTERNAL_IP>"
    echo ""
    echo -e "${BOLD}  KIND network IPs (used by Kubernetes pods)${NC}"
    echo "  ─────────────────────────────────────────────"
    echo "  MLflow  →  http://${MLFLOW_IP}:5000"
    echo "  MinIO   →  http://${MINIO_IP}:9000"
    echo ""
    echo -e "${BOLD}  Verification${NC}"
    echo "  ─────────────────────────────────────────────"
    echo "  kubectl get inferenceservice -n kserve"
    echo "  kubectl get cpol"
    echo ""
    echo -e "${BOLD}  Next steps${NC}"
    echo "  ─────────────────────────────────────────────"
    echo "  1. Run ./start_mlops.sh on every new session"
    echo "  2. Port-forward KServe: kubectl port-forward -n kserve \\"
    echo "       svc/gpon-failure-predictor-predictor 8085:80"
    echo "  3. Start the FastAPI gateway: make gateway-run"
    echo "  4. Simulate traffic: python simulate_trafic.py --target fastapi --api-key gpon-dev-key"
    echo "  5. Run drift detection: python -m monitoring.drift_detection --baseline data/processed/processed.csv --current data/predictions/latest.csv"
    echo ""
    log "GPON MLOps Platform is ready."
}

main() {
    echo ""
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║   GPON MLOps — GCP VM Setup Script       ║${NC}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════╝${NC}"
    echo ""
    info "Repo root  : $REPO_DIR"
    info "User       : $USER"
    info "Skips      : prereqs=$SKIP_PREREQS  k8s=$SKIP_K8S  compose=$SKIP_COMPOSE  images=$SKIP_IMAGES  data=$SKIP_DATA  deploy=$SKIP_DEPLOY"
    echo ""

    [[ "$SKIP_PREREQS" != "true" ]] && phase_prereqs
    phase_swap
    [[ "$SKIP_K8S"     != "true" ]] && phase_kind_cluster
    [[ "$SKIP_K8S"     != "true" ]] && phase_kfp
    [[ "$SKIP_K8S"     != "true" ]] && phase_kserve
    [[ "$SKIP_K8S"     != "true" ]] && phase_kyverno
    [[ "$SKIP_COMPOSE" != "true" ]] && phase_compose
    phase_network
    [[ "$SKIP_IMAGES"  != "true" ]] && phase_images
    phase_minio_buckets
    [[ "$SKIP_DATA"    != "true" ]] && phase_data
    [[ "$SKIP_DEPLOY"  != "true" ]] && phase_deploy_model
    phase_summary
}

main "$@"
