#!/usr/bin/env bash
# =============================================================================
# start_mlops.sh — Session startup for the GPON MLOps Platform
# =============================================================================
# Run at the start of every working session to:
#   1. Start the docker compose stack
#   2. Bridge mlflow-server and mlflow-minio to the KIND network
#   3. Detect current KIND IPs (change after every VM reboot)
#   4. Re-apply Kubernetes ConfigMaps with fresh IPs
#   5. Rebuild kfp-base from current src/ code and reload into KIND
#   6. Restart KServe predictor so it picks up fresh endpoint env vars
#
# Usage:
#   ./start_mlops.sh
#   ./start_mlops.sh --no-rebuild   # skip kfp-base rebuild (faster)
#   ./start_mlops.sh --no-compose   # skip all docker restarts
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +'%H:%M:%S')] ✔  $*${NC}"; }
info() { echo -e "${CYAN}[$(date +'%H:%M:%S')] ℹ  $*${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠  $*${NC}"; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

REBUILD=true
RESTART_COMPOSE=true

for arg in "$@"; do
    case "$arg" in
        --no-rebuild) REBUILD=false ;;
        --no-compose) RESTART_COMPOSE=false ;;
    esac
done

echo ""
echo -e "${BOLD}${CYAN}══════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}  GPON MLOps — Session Startup${NC}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════${NC}"
echo ""

# ── 1. Verify KIND cluster ────────────────────────────────────────────────────
info "Checking KIND cluster…"
if ! kind get clusters 2>/dev/null | grep -q "mlops-cluster"; then
    warn "KIND cluster 'mlops-cluster' not found. Run ./setup_gcp.sh to recreate it."
    exit 1
fi
kubectl cluster-info --context kind-mlops-cluster > /dev/null \
    && log "KIND cluster is reachable." \
    || { warn "Cannot reach cluster. Try: kubectl config use-context kind-mlops-cluster"; exit 1; }

# ── 2. Start compose services ──────────────────────────────────────────────────
if [ "$RESTART_COMPOSE" = true ]; then
    info "Starting docker compose services…"
    docker compose -f "$REPO_DIR/infrastructure/docker-compose.yml" up -d --build

    info "Waiting for mlflow-server to start…"
    for i in $(seq 1 20); do
        if curl -sf --max-time 5 http://localhost:5000/health &>/dev/null; then
            log "mlflow-server is healthy."
            break
        fi
        info "  Waiting… ($i/20)"
        sleep 3
    done

    log "Docker compose stack started."
else
    info "Skipping Docker Compose restart (--no-compose)."
fi

# ── 3. Bridge both containers to KIND network ─────────────────────────────────
info "Bridging mlflow-server and mlflow-minio to KIND network…"
for CONTAINER in mlflow-server mlflow-minio; do
    docker inspect "$CONTAINER" &>/dev/null \
        || { warn "$CONTAINER is not running — check docker ps"; exit 1; }
    docker network connect kind "$CONTAINER" 2>/dev/null \
        && log "  Connected $CONTAINER → KIND network" \
        || log "  $CONTAINER already on KIND network"
done
sleep 3

# ── 4. Detect KIND-network IPs ────────────────────────────────────────────────
info "Detecting IPs on KIND network…"
MLFLOW_IP=$(docker inspect mlflow-server \
    --format '{{range $k,$v := .NetworkSettings.Networks}}{{if eq $k "kind"}}{{$v.IPAddress}}{{end}}{{end}}')
MINIO_IP=$(docker inspect mlflow-minio \
    --format '{{range $k,$v := .NetworkSettings.Networks}}{{if eq $k "kind"}}{{$v.IPAddress}}{{end}}{{end}}')

if [ -z "$MLFLOW_IP" ] || [ -z "$MINIO_IP" ]; then
    warn "Could not detect IPs after bridging."
    docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "mlflow|minio|NAME" || true
    exit 1
fi

log "MLflow IP  : $MLFLOW_IP"
log "MinIO  IP  : $MINIO_IP"

# ── 5. Update Kubernetes ConfigMaps ───────────────────────────────────────────
info "Updating Kubernetes ConfigMaps…"
apply_configmap() {
    local ns="$1"
    kubectl get namespace "$ns" &>/dev/null || return
    kubectl create configmap mlops-endpoints \
        --from-literal=MLFLOW_TRACKING_URI="http://${MLFLOW_IP}:5000" \
        --from-literal=MLFLOW_S3_ENDPOINT_URL="http://${MINIO_IP}:9000" \
        --from-literal=MINIO_ENDPOINT_URL="http://${MINIO_IP}:9000" \
        --from-literal=AWS_ACCESS_KEY_ID="minio" \
        --from-literal=AWS_SECRET_ACCESS_KEY="minio123" \
        -n "$ns" --dry-run=client -o yaml | kubectl apply -f -
    log "  ConfigMap updated → namespace: $ns"
}
apply_configmap kubeflow
apply_configmap kserve

# ── 6. Rebuild and reload kfp-base ────────────────────────────────────────────
if [ "$REBUILD" = true ]; then
    info "Rebuilding kfp-base:latest from current src/ code…"
    docker build -f "$REPO_DIR/Dockerfile.kfp-base" -t kfp-base:latest "$REPO_DIR"
    info "Loading kfp-base:latest into KIND…"
    kind load docker-image kfp-base:latest --name mlops-cluster
    log "kfp-base:latest built and loaded."
else
    info "Skipping kfp-base rebuild (--no-rebuild)."
fi

# ── 7. Restart KServe predictor ───────────────────────────────────────────────
if kubectl get deployment gpon-failure-predictor-predictor -n kserve &>/dev/null; then
    info "Restarting KServe predictor so it refreshes endpoint env vars…"
    kubectl rollout restart deployment/gpon-failure-predictor-predictor -n kserve
    kubectl rollout status deployment/gpon-failure-predictor-predictor -n kserve --timeout=180s \
        && log "KServe predictor restarted." \
        || warn "KServe predictor restart timed out — check: kubectl get pods -n kserve"
else
    info "KServe predictor deployment not found yet; skipping rollout restart."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}  ─── Ready ────────────────────────────────${NC}"
echo "  MLflow UI     →  http://localhost:5000"
echo "  MinIO Console →  http://localhost:9001  (minio / minio123)"
echo "  Grafana       →  http://localhost:3000  (admin / admin123)"
echo "  KFP UI        →  http://localhost:3002"
echo ""
echo -e "${BOLD}  KFP pods:${NC}"
echo "  kubectl get pods -n kubeflow"
echo ""
echo -e "${BOLD}  MLflow tracking URI for pods (KIND):${NC}"
echo "  http://${MLFLOW_IP}:5000"
echo ""
echo -e "${BOLD}  MinIO endpoint for pods (KIND):${NC}"
echo "  http://${MINIO_IP}:9000"
echo ""
