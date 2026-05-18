#!/usr/bin/env bash
# =============================================================================
# setup_gcp.sh — One-shot GCP VM setup for the GPON MLOps Platform
# =============================================================================
# Tested on: Debian 12 / Ubuntu 22.04 LTS  (GCP e2-standard-4)
# Target VM:  4 vCPU · 16 GB RAM · 60 GB SSD
#
# Phases:
#   1  — System prerequisites  (Docker, kubectl, KIND, python venv)
#   2  — Swap space            (4 GB guard-rail)
#   3  — KIND cluster
#   4  — Kubeflow Pipelines    (2.15.0)
#   5  — ingress-nginx + cert-manager + KServe (v0.14.0)
#   6  — Kyverno               (v1.16.2, Audit mode)
#   7  — Docker Compose stack  (MLflow · MinIO · Prometheus · Grafana)
#   8  — Network bridging      (compose ↔ KIND) + ConfigMaps
#   9  — Container images      (build kfp-base, pull + load KServe images)
#   10 — MinIO buckets
#   11 — Data pipeline         (generate → ingest → train → MLflow register)
#   12 — Promote champion + deploy KServe InferenceService
#   13 — Summary
#
# Skip flags (export before running to skip completed phases):
#   SKIP_PREREQS=true   ./setup_gcp.sh
#   SKIP_K8S=true       ./setup_gcp.sh
#   SKIP_COMPOSE=true   ./setup_gcp.sh
#   SKIP_IMAGES=true    ./setup_gcp.sh
#   SKIP_DATA=true      ./setup_gcp.sh
#   SKIP_DEPLOY=true    ./setup_gcp.sh
#
# Usage:
#   chmod +x setup_gcp.sh && ./setup_gcp.sh
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
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

# ── Repo root ─────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# ── Skip flags — exported so child processes and any re-exec inherit them ─────
export SKIP_PREREQS="${SKIP_PREREQS:-false}"
export SKIP_K8S="${SKIP_K8S:-false}"
export SKIP_COMPOSE="${SKIP_COMPOSE:-false}"
export SKIP_IMAGES="${SKIP_IMAGES:-false}"
export SKIP_DATA="${SKIP_DATA:-false}"
export SKIP_DEPLOY="${SKIP_DEPLOY:-false}"

# ── Version pins ──────────────────────────────────────────────────────────────
KIND_VERSION="v0.31.0"
KFP_VERSION="2.15.0"
KSERVE_VERSION="v0.14.0"
CERTMANAGER_VERSION="v1.14.4"
KYVERNO_VERSION="v1.12.6"

# ── Python venv — avoids Debian 12 PEP 668 "externally managed" errors ────────
VENV_DIR="$REPO_DIR/venv"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

ensure_venv_active() {
    if [ ! -x "$PYTHON_BIN" ]; then
        info "Creating Python virtual environment at $VENV_DIR …"
        python3 -m venv "$VENV_DIR"
    fi
    export PATH="$VENV_DIR/bin:$PATH"
}

# ── swapon / mkswap — /sbin may not be in $PATH on GCP user shells ────────────
SWAPON="$(command -v swapon 2>/dev/null || echo /sbin/swapon)"
MKSWAP="$(command -v mkswap  2>/dev/null || echo /sbin/mkswap)"

# ── Misc helpers ──────────────────────────────────────────────────────────────
cmd_exists() { command -v "$1" &>/dev/null; }

wait_for_pods() {
    local ns="$1" label="$2" timeout="${3:-300}"
    info "Waiting for pods  ns=$ns  label=$label  timeout=${timeout}s …"
    kubectl wait pod --for=condition=ready -n "$ns" -l "$label" \
        --timeout="${timeout}s" \
        || error "Pods not ready after ${timeout}s (ns=$ns  label=$label)"
}

wait_for_deployment() {
    local ns="$1" dep="$2" timeout="${3:-300}"
    info "Waiting for deployment $ns/$dep …"
    kubectl rollout status deployment/"$dep" -n "$ns" --timeout="${timeout}s" \
        || error "Deployment $dep did not roll out within ${timeout}s"
}

start_mlflow_server() {
    if ! docker image inspect infrastructure-mlflow:latest >/dev/null 2>&1; then
        info "Building infrastructure-mlflow:latest …"
        docker compose -f "$REPO_DIR/infrastructure/docker-compose.yml" build mlflow
    fi

    info "Restarting mlflow-server with the stable startup path…"
    docker rm -f mlflow-server 2>/dev/null || true

    docker run -d \
        --name mlflow-server \
        --network infrastructure_mlops-network \
        --restart unless-stopped \
        -p 5000:5000 \
        -e MLFLOW_S3_ENDPOINT_URL=http://mlflow-minio:9000 \
        -e AWS_ACCESS_KEY_ID=minio \
        -e AWS_SECRET_ACCESS_KEY=minio123 \
        -e MLFLOW_SQLALCHEMYSTORE_POOL_PRE_PING=true \
        infrastructure-mlflow \
        mlflow server \
            --host 0.0.0.0 \
            --port 5000 \
            --backend-store-uri postgresql://mlflow:mlflow123@mlflow-postgres:5432/mlflow_db \
            --default-artifact-root s3://mlflow-artifacts/ \
            --gunicorn-opts "--timeout 120 --workers 2 --keep-alive 10"

    info "Waiting for mlflow-server health endpoint…"
    for i in $(seq 1 20); do
        if curl -sf --max-time 5 http://localhost:5000/health &>/dev/null; then
            log "mlflow-server is healthy."
            return
        fi
        sleep 3
    done

    error "mlflow-server did not become healthy."
}

# =============================================================================
# PHASE 1 — System Prerequisites
# =============================================================================
phase_prereqs() {
    step "Phase 1 — System Prerequisites"

    sudo apt-get update -qq
    sudo apt-get install -y \
        git ca-certificates curl gnupg lsb-release \
        python3-venv python3-pip python3-dev build-essential -qq

    # ── Docker ────────────────────────────────────────────────────────────────
    if ! cmd_exists docker; then
        info "Installing Docker Engine …"
        sudo install -m 0755 -d /etc/apt/keyrings
        curl -fsSL \
            "https://download.docker.com/linux/$(. /etc/os-release && echo "$ID")/gpg" \
            | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        sudo chmod a+r /etc/apt/keyrings/docker.gpg
        echo \
          "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
          https://download.docker.com/linux/$(. /etc/os-release && echo "$ID") \
          $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
          | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt-get update -qq
        sudo apt-get install -y \
            docker-ce docker-ce-cli containerd.io \
            docker-buildx-plugin docker-compose-plugin
        sudo systemctl enable --now docker
        log "Docker installed."
    else
        log "Docker already installed: $(docker --version)"
    fi

    # ── Docker group — GCP-safe check ─────────────────────────────────────────
    # GCP OS-Login VMs often have a primary GID with no name in /etc/group, so
    # 'groups | grep docker' is unreliable.  We test whether docker actually works
    # without sudo instead. If it fails we add the user and exit cleanly with
    # instructions — no exec/sg re-launch which breaks on GCP.
    if ! docker ps &>/dev/null; then
        sudo usermod -aG docker "$USER"
        echo ""
        echo -e "${YELLOW}══════════════════════════════════════════════════════${NC}"
        echo -e "${YELLOW}  Docker group updated for $USER.${NC}"
        echo -e "${YELLOW}  You need a new shell session before Docker works.${NC}"
        echo ""
        echo -e "${YELLOW}  Run this now:${NC}"
        echo -e "${YELLOW}    newgrp docker${NC}"
        echo -e "${YELLOW}  Then re-run:${NC}"
        echo -e "${YELLOW}    ./setup_gcp.sh${NC}"
        echo -e "${YELLOW}══════════════════════════════════════════════════════${NC}"
        echo ""
        exit 0   # clean exit — not an error, user just needs to re-run
    fi
    log "Docker daemon accessible without sudo."

    # ── kubectl ───────────────────────────────────────────────────────────────
    if ! cmd_exists kubectl; then
        info "Installing kubectl …"
        KUBECTL_VERSION="$(curl -sL https://dl.k8s.io/release/stable.txt)"
        curl -sLO "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl"
        sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
        rm kubectl
        log "kubectl installed."
    else
        log "kubectl already installed."
    fi

    # ── KIND ──────────────────────────────────────────────────────────────────
    if ! cmd_exists kind; then
        info "Installing KIND ${KIND_VERSION} …"
        curl -sLo ./kind \
            "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-amd64"
        chmod +x ./kind
        sudo mv ./kind /usr/local/bin/kind
        log "KIND installed: $(kind version)"
    else
        log "KIND already installed: $(kind version)"
    fi

    # ── Helm ──────────────────────────────────────────────────────────────────
    if ! cmd_exists helm; then
        info "Installing Helm …"
        curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 \
            | bash
        log "Helm installed."
    else
        log "Helm already installed."
    fi

    # ── Python venv + repo deps ───────────────────────────────────────────────
    # We install into a local venv, NOT system Python, to avoid PEP 668 errors
    # on Debian 12 / Ubuntu 22.04+.  Do NOT use 'make install' here — that
    # calls the system pip.
    ensure_venv_active
    info "Installing Python requirements into venv …"
    "$PIP_BIN" install --quiet --upgrade pip
    "$PIP_BIN" install --quiet -r "$REPO_DIR/requirements.txt"
    log "Python requirements installed."
}

# =============================================================================
# PHASE 2 — Swap Space
# =============================================================================
phase_swap() {
    step "Phase 2 — Swap Space (4 GB guard-rail)"
    local swap_file="/swapfile"

    # Use full /sbin paths — GCP user $PATH may omit /sbin
    if "$SWAPON" --show 2>/dev/null | grep -q "$swap_file"; then
        log "Swap already active."
        return
    fi

    if [ ! -f "$swap_file" ]; then
        info "Creating 4 GB swap file …"
        sudo fallocate -l 4G "$swap_file"
        sudo chmod 600 "$swap_file"
        sudo "$MKSWAP" "$swap_file"
    else
        info "Swap file already exists — enabling …"
    fi

    # 'swapon' on an already-active swap returns "device busy"; suppress that
    sudo "$SWAPON" "$swap_file" 2>/dev/null || true

    grep -q "$swap_file" /etc/fstab \
        || echo "${swap_file} none swap sw 0 0" | sudo tee -a /etc/fstab

    log "Swap: $(free -h | awk '/Swap/{print $2}')"
}

# =============================================================================
# PHASE 3 — KIND Cluster
# =============================================================================
phase_kind_cluster() {
    step "Phase 3 — KIND Cluster"

    if kind get clusters 2>/dev/null | grep -q "mlops-cluster"; then
        log "KIND cluster 'mlops-cluster' already exists."
    else
        info "Creating KIND cluster from kind-config.yaml …"
        kind create cluster --config "$REPO_DIR/kind-config.yaml" --wait 120s
        log "KIND cluster created."
    fi

    kubectl cluster-info --context kind-mlops-cluster \
        || error "Cannot reach KIND cluster. Is Docker running?"
    log "Cluster accessible."

    for ns in kubeflow kserve kyverno; do
        kubectl get namespace "$ns" &>/dev/null \
            || kubectl create namespace "$ns"
    done
    log "Namespaces ensured: kubeflow, kserve, kyverno."
}

# =============================================================================
# PHASE 4 — Kubeflow Pipelines
# =============================================================================
phase_kfp() {
    step "Phase 4 — Kubeflow Pipelines (${KFP_VERSION})"

    if kubectl get deployment ml-pipeline -n kubeflow &>/dev/null; then
        log "KFP already installed — skipping."
        return
    fi

    local tmp_dir
    tmp_dir="$(mktemp -d)"
    # shellcheck disable=SC2064
    trap "rm -rf '$tmp_dir'" RETURN

    info "Cloning kubeflow/pipelines tag ${KFP_VERSION} …"
    git clone --depth 1 --branch "${KFP_VERSION}" \
        https://github.com/kubeflow/pipelines.git \
        "$tmp_dir/pipelines" \
        || error "Failed to clone kubeflow/pipelines."

    info "Applying KFP cluster-scoped resources …"
    kubectl apply -k "$tmp_dir/pipelines/manifests/kustomize/cluster-scoped-resources"

    kubectl wait --for=condition=established --timeout=120s \
        crd/applications.app.k8s.io || warn "CRD wait timed out — continuing."

    info "Applying KFP platform-agnostic manifests …"
    kubectl apply -k "$tmp_dir/pipelines/manifests/kustomize/env/platform-agnostic"

    info "Waiting for KFP core deployments (may take 5–10 min) …"
    for dep in ml-pipeline ml-pipeline-ui \
                ml-pipeline-persistenceagent ml-pipeline-scheduledworkflow; do
        wait_for_deployment kubeflow "$dep" 600
    done

    # Patch to NodePort 30080 which kind-config.yaml maps to host port 8080.
    # Use json-patch to target only port index 0 — avoids replacing all ports.
    info "Patching KFP UI service to NodePort 30080 …"
    kubectl patch svc ml-pipeline-ui -n kubeflow --type=json \
        -p '[
              {"op":"replace","path":"/spec/type","value":"NodePort"},
              {"op":"replace","path":"/spec/ports/0/nodePort","value":30080}
            ]' \
        || warn "NodePort patch failed — use port-forward as fallback: kubectl port-forward svc/ml-pipeline-ui 8080:80 -n kubeflow"

    log "KFP installed → http://localhost:8080"
}

# =============================================================================
# PHASE 5 — ingress-nginx + cert-manager + KServe
# =============================================================================
phase_kserve() {
    step "Phase 5 — ingress-nginx + cert-manager + KServe (${KSERVE_VERSION})"

    # ── cert-manager ──────────────────────────────────────────────────────────
    if ! kubectl get namespace cert-manager &>/dev/null; then
        info "Installing cert-manager ${CERTMANAGER_VERSION} …"
        kubectl apply -f \
            "https://github.com/cert-manager/cert-manager/releases/download/${CERTMANAGER_VERSION}/cert-manager.yaml"
        wait_for_pods cert-manager "app=cert-manager" 180
        wait_for_pods cert-manager "app=webhook" 180
        sleep 15
        log "cert-manager ready."
    else
        log "cert-manager already installed."
    fi

    # ── ingress-nginx ─────────────────────────────────────────────────────────
    if ! kubectl get namespace ingress-nginx &>/dev/null; then
        info "Installing ingress-nginx (KIND flavour) …"
        kubectl apply -f \
            "https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml"
        kubectl wait --namespace ingress-nginx \
            --for=condition=ready pod \
            --selector=app.kubernetes.io/component=controller \
            --timeout=180s \
            || error "ingress-nginx controller did not become ready."
        log "ingress-nginx ready."
    else
        log "ingress-nginx already installed."
    fi

    # ── KServe ────────────────────────────────────────────────────────────────
    if ! kubectl get deployment kserve-controller-manager -n kserve &>/dev/null; then
        info "Installing KServe ${KSERVE_VERSION} …"
        kubectl apply --server-side -f \
            "https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve.yaml"
        kubectl apply --server-side -f \
            "https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve-cluster-resources.yaml" \
            || warn "kserve-cluster-resources returned non-zero — may be partial."    
        wait_for_deployment kserve kserve-controller-manager 300
        log "KServe installed."
    else
        log "KServe already installed."
    fi

    info "Patching KServe configmap (RawDeployment + nginx ingress) …"
    kubectl patch configmap/inferenceservice-config -n kserve --type=strategic \
        -p '{"data":{"deploy":"{\"defaultDeploymentMode\":\"RawDeployment\"}"}}' \
        2>/dev/null || warn "Could not patch deploy config."
    kubectl patch configmap/inferenceservice-config -n kserve --type=strategic \
        -p '{"data":{"ingress":"{\"ingressClassName\":\"nginx\",\"disableIngressCreation\":false}"}}' \
        2>/dev/null || warn "Could not patch ingress config."

    log "KServe configured for RawDeployment."
}

# =============================================================================
# PHASE 6 — Kyverno
# =============================================================================
load_image_to_kind() {
    local image="$1"
    local tar_file="/tmp/$(echo "$image" | tr '/:' '_').tar"

    info "Pulling image: $image"
    docker pull "$image"

    info "Saving image to tar: $tar_file"
    docker save "$image" -o "$tar_file"

    info "Importing image into KIND containerd ..."
    docker exec -i mlops-cluster-control-plane \
        ctr --namespace=k8s.io images import \
        --snapshotter=overlayfs - < "$tar_file"

    rm -f "$tar_file"

    log "Loaded into KIND: $image"
}
phase_kyverno() {
    step "Phase 6 — Kyverno (${KYVERNO_VERSION}, Audit mode)"

    if ! kubectl get deployment kyverno-admission-controller -n kyverno &>/dev/null; then
        info "Installing Kyverno ${KYVERNO_VERSION} …"
        kubectl apply --server-side=true -f \
            "https://github.com/kyverno/kyverno/releases/download/${KYVERNO_VERSION}/install.yaml"
        
        # ADD THIS LINE: Wait for CRDs before proceeding
        kubectl wait --for=condition=established --timeout=120s crd/clusterpolicies.kyverno.io
        
        for dep in kyverno-admission-controller kyverno-background-controller \
                   kyverno-reports-controller kyverno-cleanup-controller; do
            kubectl get deployment "$dep" -n kyverno &>/dev/null \
                && wait_for_deployment kyverno "$dep" 300 || true
        done
        log "Kyverno installed."
    else
        log "Kyverno already installed."
        # ALSO ADD THIS HERE: Ensure CRDs are ready even if already installed
        kubectl wait --for=condition=established --timeout=60s crd/clusterpolicies.kyverno.io
    fi

    info "Applying starter Audit policies …"
    kubectl apply -f "$REPO_DIR/security/kyverno/policies/disallow-privileged-kserve.yaml"
    kubectl apply -f "$REPO_DIR/security/kyverno/policies/require-pod-resources-kserve.yaml"
    log "Kyverno policies applied."
}

# =============================================================================
# PHASE 7 — Docker Compose stack
# =============================================================================
phase_compose() {
    step "Phase 7 — Docker Compose MLflow Stack"

    info "Starting non-mlflow compose services …"
    docker compose -f "$REPO_DIR/infrastructure/docker-compose.yml" up -d --build \
        postgres minio minio-setup grafana prometheus node-exporter

    info "Waiting 15 s for postgres and minio to stabilise …"
    sleep 15

    for svc in mlflow-postgres mlflow-minio; do
        status=$(docker inspect "$svc" --format '{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
        [ "$status" = "healthy" ] \
            && log "  ✔ $svc healthy" \
            || warn "  $svc status: $status"
    done

    start_mlflow_server

    for svc in mlflow-server grafana prometheus; do
        docker ps --format '{{.Names}}' | grep -q "$svc" \
            && log "  ✔ $svc running" \
            || warn "  $svc not running"
    done
}

# =============================================================================
# PHASE 8 — Network bridging + ConfigMaps
# =============================================================================
phase_network() {
    step "Phase 8 — Network Bridging and ConfigMaps"

    info "Connecting compose containers to KIND network …"
    for container in mlflow-server mlflow-minio; do
        docker network connect kind "$container" 2>/dev/null \
            && log "  Connected $container" \
            || log "  $container already on KIND network"
    done
    sleep 5

    info "Detecting container IPs on KIND network …"
    MLFLOW_IP=$(docker inspect mlflow-server \
        --format '{{range $k,$v := .NetworkSettings.Networks}}{{if eq $k "kind"}}{{$v.IPAddress}}{{end}}{{end}}')
    MINIO_IP=$(docker inspect mlflow-minio \
        --format '{{range $k,$v := .NetworkSettings.Networks}}{{if eq $k "kind"}}{{$v.IPAddress}}{{end}}{{end}}')

    [ -z "$MLFLOW_IP" ] && error "Could not detect MLflow IP on KIND network."
    [ -z "$MINIO_IP"  ] && error "Could not detect MinIO IP on KIND network."

    log "MLflow IP : $MLFLOW_IP"
    log "MinIO  IP : $MINIO_IP"

    for ns in kubeflow kserve; do
        kubectl get namespace "$ns" &>/dev/null || continue
        kubectl create configmap mlops-endpoints \
            --from-literal=MLFLOW_TRACKING_URI="http://${MLFLOW_IP}:5000" \
            --from-literal=MLFLOW_S3_ENDPOINT_URL="http://${MINIO_IP}:9000" \
            --from-literal=MINIO_ENDPOINT_URL="http://${MINIO_IP}:9000" \
            --from-literal=AWS_ACCESS_KEY_ID="minio" \
            --from-literal=AWS_SECRET_ACCESS_KEY="minio123" \
            -n "$ns" --dry-run=client -o yaml | kubectl apply -f -
        log "  ConfigMap applied → namespace: $ns"
    done

    export MLFLOW_IP MINIO_IP
    export MLFLOW_TRACKING_URI="http://localhost:5000"
    export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
    export MINIO_ENDPOINT_URL="http://localhost:9000"
    export AWS_ACCESS_KEY_ID="minio"
    export AWS_SECRET_ACCESS_KEY="minio123"
    log "Network bridging and ConfigMaps complete."
}

# =============================================================================
# PHASE 9 — Container images
# =============================================================================
phase_images() {
    step "Phase 9 — Container Images"

    info "Building kfp-base:latest …"
    docker build -f "$REPO_DIR/Dockerfile.kfp-base" -t kfp-base:latest "$REPO_DIR"
    kind load docker-image kfp-base:latest --name mlops-cluster
    log "kfp-base:latest built and loaded."
    load_image_to_kind "kserve/xgbserver:v0.14.0"
    load_image_to_kind "amazon/aws-cli:latest"
    load_image_to_kind "busybox:1.36"
    log "All images loaded into KIND."
}

# =============================================================================
# PHASE 10 — MinIO buckets
# =============================================================================
phase_minio_buckets() {
    step "Phase 10 — MinIO Buckets"

    ensure_venv_active

    info "Waiting for MinIO to be healthy …"
    for i in $(seq 1 30); do
        curl -sf "http://localhost:9000/minio/health/live" &>/dev/null && break
        sleep 3
        [ "$i" -eq 30 ] && error "MinIO did not become healthy in 90 s."
    done
    log "MinIO is healthy."

    "$PYTHON_BIN" - <<'PYEOF'
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
        print(f"  already exists: {bucket}")
    except ClientError:
        s3.create_bucket(Bucket=bucket)
        print(f"  created: {bucket}")
PYEOF
    log "MinIO buckets ready."
}

# =============================================================================
# PHASE 11 — Data pipeline
# =============================================================================
phase_data() {
    step "Phase 11 — Data Pipeline (generate → ingest → train → register)"

    ensure_venv_active

    export MLFLOW_TRACKING_URI="http://localhost:5000"
    export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
    export MINIO_ENDPOINT_URL="http://localhost:9000"
    export AWS_ACCESS_KEY_ID="minio"
    export AWS_SECRET_ACCESS_KEY="minio123"

    if [ ! -f "$REPO_DIR/data/raw/telemetry.csv" ]; then
        info "Generating synthetic GPON telemetry …"
        mkdir -p "$REPO_DIR/data/raw"
        "$PYTHON_BIN" "$REPO_DIR/scripts/generate_data.py"
        log "Raw data generated."
    else
        log "Raw telemetry already exists — skipping generation."
    fi

    info "Waiting for MLflow server …"
    for i in $(seq 1 40); do
        curl -sf "http://localhost:5000/health" &>/dev/null && break
        sleep 3
        [ "$i" -eq 40 ] && error "MLflow did not become ready in 120 s."
    done
    log "MLflow is ready."


    info "Uploading raw telemetry to MinIO …"
    "$PYTHON_BIN" - <<'PYEOF'
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
print("  uploaded → s3://gpon-telemetry/raw/telemetry.csv")
PYEOF
    log "Telemetry uploaded."

    info "Compiling Kubeflow pipeline …"
    "$PYTHON_BIN" -m pipelines.kubeflow_pipeline
    log "Pipeline compiled → pipelines/pipeline.yaml"
}
phase_port_forward() {
    step "Phase 12 — Local Port Forwards"

    if lsof -i :3002 >/dev/null 2>&1; then
        log "Port 3002 already in use — skipping KFP port-forward."
        return
    fi

    info "Starting Kubeflow Pipelines UI port-forward on :3002 …"

    nohup kubectl -n kubeflow \
    port-forward --address 0.0.0.0 \
    svc/ml-pipeline-ui 3002:80 \
        >/tmp/kfp-portforward.log 2>&1 &

    sleep 3

    if lsof -i :3002 >/dev/null 2>&1; then
        log "KFP UI available at http://localhost:3002"
    else
        warn "KFP port-forward may have failed. Check /tmp/kfp-portforward.log"
    fi
}


# =============================================================================
# PHASE 12 — Summary
# =============================================================================
phase_summary() {
    step "Setup Complete"

    _get_ip() {
        docker inspect "$1" \
            --format '{{range $k,$v := .NetworkSettings.Networks}}{{if eq $k "kind"}}{{$v.IPAddress}}{{end}}{{end}}' \
            2>/dev/null || echo "unknown"
    }
    local mf_ip mn_ip
    mf_ip="$(_get_ip mlflow-server)"
    mn_ip="$(_get_ip mlflow-minio)"

    echo
    echo "══════════════════════════════════════"
    echo " Environment Ready"
    echo "══════════════════════════════════════"
    echo
    echo "Activate Python environment:"
    echo "  source venv/bin/activate"
    echo
    echo "Compile pipeline:"
    echo "  python pipelines/compile_pipeline.py"
    echo

    echo ""
    echo -e "${BOLD}  ── On this VM ──────────────────────────────────${NC}"
    echo "  MLflow UI     →  http://localhost:5000"
    echo "  MinIO Console →  http://localhost:9001  (minio / minio123)"
    echo "  Grafana       →  http://localhost:3000  (admin / admin123)"
    echo "  Prometheus    →  http://localhost:9090"
    echo "  KFP UI        →  http://localhost:3002"
    echo ""
    echo -e "${BOLD}  ── From your laptop (SSH tunnel) ──────────────${NC}"
    echo "  ssh -N \\"
    echo "    -L 5000:localhost:5000 \\"
    echo "    -L 9001:localhost:9001 \\"
    echo "    -L 3000:localhost:3000 \\"
    echo "    -L 9090:localhost:9090 \\"
    echo "    <YOUR_GCP_USER>@<VM_EXTERNAL_IP>"
    echo ""
    echo -e "${BOLD}  ── KIND network IPs (pod-to-pod traffic) ──────${NC}"
    echo "  MLflow → http://${mf_ip}:5000"
    echo "  MinIO  → http://${mn_ip}:9000"
    echo ""
    echo -e "${BOLD}  ── Verification ────────────────────────────────${NC}"
    echo "  kubectl get pods -n kubeflow"
    echo "  kubectl get inferenceservice -n kserve"
    echo "  kubectl get cpol"
    echo ""
    echo -e "${BOLD}  ── Every session ───────────────────────────────${NC}"
    echo "  ./start_mlops.sh"
    echo ""
    echo -e "${BOLD}  ── Monitoring workflow (after gateway is up) ───${NC}"
    echo "  kubectl port-forward -n kserve \\"
    echo "    svc/gpon-failure-predictor-predictor 8085:80 &"
    echo "  make gateway-run &"
    echo "  python simulate_trafic.py --target fastapi --api-key gpon-dev-key"
    echo "  python -m monitoring.metrics_exporter --port 8000 --interval 30"
    echo "  python -m monitoring.drift_detection \\"
    echo "    --baseline data/processed/processed.csv \\"
    echo "    --current  data/predictions/latest.csv"
    echo ""
    log "GPON MLOps Platform is ready."
}

# =============================================================================
# MAIN
# =============================================================================
main() {
    echo ""
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║   MLOps — VM Setup Script                ║${NC}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════╝${NC}"
    echo ""
    info "Repo   : $REPO_DIR"
    info "User   : $USER"
    info "Skips  : prereqs=$SKIP_PREREQS  k8s=$SKIP_K8S  compose=$SKIP_COMPOSE  images=$SKIP_IMAGES  data=$SKIP_DATA  deploy=$SKIP_DEPLOY"
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
    phase_port_forward
    phase_summary
}

main "$@"
