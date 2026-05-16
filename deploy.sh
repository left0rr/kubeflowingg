#!/usr/bin/env bash
# =============================================================================
# deploy_model.sh — Promote MLflow champion and deploy to KServe
# =============================================================================
# Run this after every successful Kubeflow pipeline run.
#
# What it does:
#   1. Verifies the latest registered MLflow model version is READY
#   2. Runs promote_champion.py:
#        - compares candidate vs current champion metric (test_auc_roc)
#        - if better (or no champion yet): writes model to
#          s3://deployment-models/gpon-failure-predictor/champion/model.bst
#   3. Applies deployment/kserve/inference_service.yaml
#   4. Waits for the KServe predictor pod to become Ready
#   5. Port-forwards KServe on :8085 and fires a smoke-test curl request
#
# Usage:
#   ./deploy_model.sh                         # normal run
#   ./deploy_model.sh --force                 # promote even if metric didn't improve
#   ./deploy_model.sh --skip-kserve-wait      # apply YAML but don't wait for Ready
#   ./deploy_model.sh --smoke-test-only       # just run curl against existing port-fwd
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
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

# ── Resolve paths ─────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

VENV_PYTHON="$REPO_DIR/venv/bin/python"
[ -x "$VENV_PYTHON" ] || error "Python venv not found at $REPO_DIR/venv. Run ./setup_gcp.sh first."
export PATH="$REPO_DIR/venv/bin:$PATH"

# ── Parse flags ───────────────────────────────────────────────────────────────
FORCE_PROMOTE=false
SKIP_KSERVE_WAIT=false
SMOKE_TEST_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --force)              FORCE_PROMOTE=true ;;
        --skip-kserve-wait)   SKIP_KSERVE_WAIT=true ;;
        --smoke-test-only)    SMOKE_TEST_ONLY=true ;;
    esac
done

# ── Environment — sourced from running containers ─────────────────────────────
# We detect the current IPs at runtime so this script works after VM reboots
# without any manual editing.
detect_ips() {
    MLFLOW_IP=$(docker inspect mlflow-server \
        --format '{{range $k,$v := .NetworkSettings.Networks}}{{if eq $k "kind"}}{{$v.IPAddress}}{{end}}{{end}}' \
        2>/dev/null || true)
    MINIO_IP=$(docker inspect mlflow-minio \
        --format '{{range $k,$v := .NetworkSettings.Networks}}{{if eq $k "kind"}}{{$v.IPAddress}}{{end}}{{end}}' \
        2>/dev/null || true)

    [ -z "$MLFLOW_IP" ] && error "Could not detect MLflow IP. Is mlflow-server running? Try: ./start_mlops.sh"
    [ -z "$MINIO_IP"  ] && error "Could not detect MinIO IP. Is mlflow-minio running?  Try: ./start_mlops.sh"
}

export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
export MINIO_ENDPOINT_URL="http://localhost:9000"
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"

# ── Smoke test only shortcut ───────────────────────────────────────────────────
if [ "$SMOKE_TEST_ONLY" = true ]; then
    step "Smoke Test — KServe inference"
    run_smoke_test
    exit 0
fi

# =============================================================================
# Step 1 — Verify MLflow has a READY model version to promote
# =============================================================================
step "Step 1 — Verify registered model in MLflow"

info "Checking MLflow for registered model versions …"
"$VENV_PYTHON" - <<'PYEOF'
import sys
import mlflow
from mlflow import MlflowClient

client = MlflowClient(tracking_uri="http://localhost:5000")
model_name = "gpon-xgboost-classifier"

try:
    versions = client.search_model_versions(f"name = '{model_name}'")
except Exception as e:
    print(f"  ERROR: cannot connect to MLflow: {e}", file=sys.stderr)
    sys.exit(1)

if not versions:
    print(f"  ERROR: no registered versions found for '{model_name}'.", file=sys.stderr)
    print("  Make sure the KFP pipeline completed the registration step.", file=sys.stderr)
    sys.exit(1)

latest = max(versions, key=lambda v: int(v.version))
print(f"  Latest version : {latest.version}")
print(f"  Status         : {latest.status}")
print(f"  Run ID         : {latest.run_id}")

if latest.status != "READY":
    print(f"  ERROR: version {latest.version} is not READY (status={latest.status}).", file=sys.stderr)
    print("  Wait for the KFP pipeline to finish, then re-run this script.", file=sys.stderr)
    sys.exit(1)

# Print the AUC so the user can see what they're promoting
run = client.get_run(latest.run_id)
auc = run.data.metrics.get("test_auc_roc", "n/a")
print(f"  test_auc_roc   : {auc}")
print("  Model is READY — proceeding to promotion.")
PYEOF

log "MLflow model version verified."

# =============================================================================
# Step 2 — Promote champion to stable MinIO deployment path
# =============================================================================
step "Step 2 — Promote champion model"

detect_ips

FORCE_FLAG=""
[ "$FORCE_PROMOTE" = true ] && FORCE_FLAG="--force"

info "Running promote_champion.py …"
info "  MLflow         : http://localhost:5000"
info "  MinIO          : http://localhost:9000  (KIND IP: $MINIO_IP)"
info "  Deployment URI : s3://deployment-models/gpon-failure-predictor/champion/model.bst"
info "  Force          : $FORCE_PROMOTE"

"$VENV_PYTHON" "$REPO_DIR/monitoring/promote_champion.py" \
    --tracking-uri    http://localhost:5000 \
    --model-name      gpon-xgboost-classifier \
    --alias           champion \
    --metric-name     test_auc_roc \
    --deployment-model-uri "s3://deployment-models/gpon-failure-predictor/champion/model.bst" \
    --minio-endpoint  http://localhost:9000 \
    --skip-rollout-restart \
    $FORCE_FLAG \
    || error "promote_champion.py failed. Check the output above."

log "Champion model promoted."

# =============================================================================
# Step 3 — Verify the artifact actually landed in MinIO
# =============================================================================
step "Step 3 — Verify artifact in MinIO"

"$VENV_PYTHON" - <<'PYEOF'
import boto3, sys
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
key = "gpon-failure-predictor/champion/model.bst"
try:
    resp = s3.head_object(Bucket="deployment-models", Key=key)
    size_kb = resp["ContentLength"] // 1024
    print(f"  ✔  deployment-models/{key}  ({size_kb} KB)")
except ClientError as e:
    print(f"  ✘  artifact not found: {e}", file=sys.stderr)
    print("  promote_champion.py may have skipped promotion (candidate did not beat champion).", file=sys.stderr)
    print("  Re-run with --force to promote regardless of metric comparison.", file=sys.stderr)
    sys.exit(1)
PYEOF

log "Artifact confirmed in MinIO."

# =============================================================================
# Step 4 — Apply (or re-apply) the KServe InferenceService
# =============================================================================
step "Step 4 — Apply KServe InferenceService"

# Check the ConfigMap is current (IPs may have changed since setup)
info "Refreshing mlops-endpoints ConfigMap in kserve namespace …"
kubectl create configmap mlops-endpoints \
    --from-literal=MLFLOW_TRACKING_URI="http://${MLFLOW_IP}:5000" \
    --from-literal=MLFLOW_S3_ENDPOINT_URL="http://${MINIO_IP}:9000" \
    --from-literal=MINIO_ENDPOINT_URL="http://${MINIO_IP}:9000" \
    --from-literal=AWS_ACCESS_KEY_ID="minio" \
    --from-literal=AWS_SECRET_ACCESS_KEY="minio123" \
    -n kserve --dry-run=client -o yaml | kubectl apply -f -
log "ConfigMap refreshed."

info "Applying deployment/kserve/inference_service.yaml …"
kubectl apply -f "$REPO_DIR/deployment/kserve/inference_service.yaml"
log "InferenceService applied."

# If the predictor deployment already exists, restart it so the new model.bst
# is downloaded by the aws-cli sidecar (old pod still has the old model cached)
if kubectl get deployment gpon-failure-predictor-predictor -n kserve &>/dev/null; then
    info "Restarting existing predictor deployment to pick up the new model …"
    kubectl rollout restart deployment/gpon-failure-predictor-predictor -n kserve
    log "Rollout restart triggered."
fi

# =============================================================================
# Step 5 — Wait for predictor to become Ready
# =============================================================================
if [ "$SKIP_KSERVE_WAIT" = false ]; then
    step "Step 5 — Wait for KServe InferenceService to become Ready"

    info "Waiting for InferenceService Ready condition (up to 7 min) …"
    info "  This includes: pod scheduling → model-downloader sidecar → xgbserver readiness probe"

    # The InferenceService Ready condition is the most reliable signal
    kubectl wait \
        --for=condition=Ready \
        inferenceservice/gpon-failure-predictor \
        -n kserve \
        --timeout=420s \
        && log "InferenceService is Ready." \
        || {
            warn "InferenceService did not report Ready within 7 min."
            warn "Useful diagnostics:"
            warn "  kubectl describe inferenceservice -n kserve gpon-failure-predictor"
            warn "  kubectl get pods -n kserve"
            warn "  kubectl logs -n kserve -l serving.kserve.io/inferenceservice=gpon-failure-predictor -c model-downloader"
            warn "  kubectl logs -n kserve -l serving.kserve.io/inferenceservice=gpon-failure-predictor -c kserve-container"
        }
else
    info "Skipping KServe wait (--skip-kserve-wait)."
    info "Check status with: kubectl get inferenceservice -n kserve"
fi

# =============================================================================
# Step 6 — Port-forward KServe and smoke test
# =============================================================================
step "Step 6 — Port-forward KServe and smoke test"

# Kill any stale port-forward on 8085
if lsof -ti :8085 &>/dev/null; then
    info "Killing stale port-forward on :8085 …"
    lsof -ti :8085 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

info "Starting KServe port-forward :8085 → svc/gpon-failure-predictor-predictor:80 …"
kubectl port-forward \
    -n kserve \
    svc/gpon-failure-predictor-predictor \
    8085:80 \
    --address 0.0.0.0 \
    >/tmp/kserve-portforward.log 2>&1 &

KSERVE_PF_PID=$!
sleep 4

if ! kill -0 $KSERVE_PF_PID 2>/dev/null; then
    warn "Port-forward process exited immediately. Check /tmp/kserve-portforward.log"
    cat /tmp/kserve-portforward.log || true
else
    log "Port-forward running (PID $KSERVE_PF_PID) → KServe at http://localhost:8085"
fi

# Smoke test — send one real inference request
info "Sending smoke-test inference request …"
RESPONSE=$(curl -sf \
    --max-time 10 \
    -X POST http://localhost:8085/v1/models/gpon-failure-predictor:predict \
    -H "Content-Type: application/json" \
    -d '{
        "instances": [[
            -18.5,
            2.3,
            42.1,
            35.2,
            12,
            1,
            4,
            730,
            0,
            3.3
        ]]
    }' 2>&1 || true)

if echo "$RESPONSE" | grep -q "predictions"; then
    log "Smoke test PASSED  →  $RESPONSE"
else
    warn "Smoke test did not return expected response."
    warn "Response was: $RESPONSE"
    warn "KServe may still be starting up. Retry in 30 s:"
    warn "  curl -X POST http://localhost:8085/v1/models/gpon-failure-predictor:predict \\"
    warn "    -H 'Content-Type: application/json' \\"
    warn "    -d '{\"instances\": [[-18.5,2.3,42.1,35.2,12,1,4,730,0,3.3]]}'"
fi

# =============================================================================
# Summary
# =============================================================================
step "Deployment Complete"

echo ""
echo -e "${BOLD}  KServe endpoint (VM-local):${NC}"
echo "  http://localhost:8085/v1/models/gpon-failure-predictor:predict"
echo ""
echo -e "${BOLD}  Example inference request:${NC}"
cat <<'EXAMPLE'
  curl -X POST http://localhost:8085/v1/models/gpon-failure-predictor:predict \
    -H "Content-Type: application/json" \
    -d '{"instances": [[-18.5, 2.3, 42.1, 35.2, 12, 1, 4, 730, 0, 3.3]]}'
EXAMPLE
echo ""
echo -e "${BOLD}  FastAPI gateway (start it separately):${NC}"
echo "  export FASTAPI_GATEWAY_API_KEY=gpon-dev-key"
echo "  export KSERVE_PREDICT_URL=http://localhost:8085/v1/models/gpon-failure-predictor:predict"
echo "  make gateway-run"
echo ""
echo -e "${BOLD}  Promote a new version in the future:${NC}"
echo "  ./deploy_model.sh                 # only promotes if AUC improved"
echo "  ./deploy_model.sh --force         # always promotes latest version"
echo ""
log "Done."