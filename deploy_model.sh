#!/usr/bin/env bash
# =============================================================================
# deploy_model.sh — Promote MLflow champion and deploy to KServe
# =============================================================================
# Run this after every successful Kubeflow pipeline run.
#
# Usage:
#   ./deploy_model.sh                    # promote only if AUC improved
#   ./deploy_model.sh --force            # always promote latest version
#   ./deploy_model.sh --skip-kserve-wait # apply YAML but don't wait for Ready
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

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

VENV_PYTHON="$REPO_DIR/venv/bin/python"
[ -x "$VENV_PYTHON" ] || error "venv not found at $REPO_DIR/venv — run ./setup_gcp.sh first."
export PATH="$REPO_DIR/venv/bin:$PATH"

# ── Flags ─────────────────────────────────────────────────────────────────────
FORCE_PROMOTE=false
SKIP_KSERVE_WAIT=false

for arg in "$@"; do
    case "$arg" in
        --force)             FORCE_PROMOTE=true ;;
        --skip-kserve-wait)  SKIP_KSERVE_WAIT=true ;;
    esac
done

# ── Env ───────────────────────────────────────────────────────────────────────
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
export MINIO_ENDPOINT_URL="http://localhost:9000"
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"

detect_ips() {
    for container in mlflow-server mlflow-minio; do
        docker inspect "$container" &>/dev/null \
            || error "Container $container is not running. Start the stack first."
        docker network connect kind "$container" 2>/dev/null || true
    done

    MLFLOW_IP=$(docker inspect mlflow-server \
        --format '{{range $k,$v := .NetworkSettings.Networks}}{{if eq $k "kind"}}{{$v.IPAddress}}{{end}}{{end}}' \
        2>/dev/null || true)
    MINIO_IP=$(docker inspect mlflow-minio \
        --format '{{range $k,$v := .NetworkSettings.Networks}}{{if eq $k "kind"}}{{$v.IPAddress}}{{end}}{{end}}' \
        2>/dev/null || true)
        
    if [ -z "$MLFLOW_IP" ]; then error "Cannot detect MLflow IP — run ./start_mlops.sh first."; fi
    if [ -z "$MINIO_IP" ]; then error "Cannot detect MinIO IP  — run ./start_mlops.sh first."; fi
}

# =============================================================================
# Step 1 — Wait for MLflow to be fully ready (workers stable, not just HTTP up)
# =============================================================================
step "Step 1 — Wait for MLflow to be ready"

# MLflow's /health returns OK even when gunicorn workers are still warming up
# or recovering from a crash. We probe the actual model-versions API with curl
# (fast, no persistent connection) until we get a valid JSON response.
# This is the same endpoint confirm-working in diagnostics.
info "Probing MLflow API until workers are stable (up to 3 min) …"

MODEL_JSON=""
CONSECUTIVE_OK=0
for i in $(seq 1 36); do
    RESPONSE=$(curl -sf --max-time 10 \
        "http://localhost:5000/api/2.0/mlflow/model-versions/search?filter=name+%3D+%27gpon-xgboost-classifier%27&max_results=100" \
        2>/dev/null || true)

    if echo "$RESPONSE" | grep -q '"model_versions"'; then
        MODEL_JSON="$RESPONSE"
        CONSECUTIVE_OK=$((CONSECUTIVE_OK + 1))
        info "  MLflow API success streak: $CONSECUTIVE_OK/3"
        if [ "$CONSECUTIVE_OK" -ge 3 ]; then
            log "MLflow API is responding consistently (attempt $i)."
            break
        fi
        sleep 3
        continue
    fi

    CONSECUTIVE_OK=0
    warn "MLflow workers not ready yet — waiting 5 s … ($i/36)"
    sleep 5
done

[ -z "$MODEL_JSON" ] && error "MLflow did not become ready after 3 min.
  Check: docker logs mlflow-server --tail 30
  Try restarting: docker compose -f infrastructure/docker-compose.yml restart mlflow"

# Parse and validate the response — pure bash + python stdin pipe, no HTTP call
MODEL_JSON_ENV="$MODEL_JSON" "$VENV_PYTHON" - <<'PYEOF'
import json
import os
import sys

data = json.loads(os.environ["MODEL_JSON_ENV"])
versions = data.get("model_versions", [])

if not versions:
    print("  ERROR: no registered model versions found for 'gpon-xgboost-classifier'.", file=sys.stderr)
    print("  The KFP pipeline registration step may not have completed.", file=sys.stderr)
    sys.exit(1)

latest = max(versions, key=lambda v: int(v["version"]))

print(f"  Found    : {len(versions)} version(s)")
print(f"  Latest   : version {latest['version']}")
print(f"  Status   : {latest['status']}")
print(f"  Run ID   : {latest['run_id']}")

if latest["status"] != "READY":
    print(
        f"\n  ERROR: version {latest['version']} status is "
        f"'{latest['status']}' — not READY.",
        file=sys.stderr,
    )
    sys.exit(1)
PYEOF

log "Registered model confirmed READY."

# Pull run metrics via curl too (informational)
RUN_ID=$(MODEL_JSON_ENV="$MODEL_JSON" "$VENV_PYTHON" -c "
import json, os
v=json.loads(os.environ['MODEL_JSON_ENV']).get('model_versions', [])
print(max(v, key=lambda x:int(x['version']))['run_id'])
")
CANDIDATE_VERSION=$(MODEL_JSON_ENV="$MODEL_JSON" "$VENV_PYTHON" -c "
import json, os
v=json.loads(os.environ['MODEL_JSON_ENV']).get('model_versions', [])
print(max(v, key=lambda x:int(x['version']))['version'])
")
CANDIDATE_SOURCE_URI=$(MODEL_JSON_ENV="$MODEL_JSON" "$VENV_PYTHON" -c "
import json, os
v=json.loads(os.environ['MODEL_JSON_ENV']).get('model_versions', [])
print(max(v, key=lambda x:int(x['version'])).get('source', ''))
")
info "Run ID: $RUN_ID"
info "Candidate version: $CANDIDATE_VERSION"
info "Candidate source: $CANDIDATE_SOURCE_URI"

RUN_DATA=$(curl -sf --max-time 10 \
    "http://localhost:5000/api/2.0/mlflow/runs/get?run_id=${RUN_ID}" \
    2>/dev/null || echo '{}')
CANDIDATE_METRIC=$(RUN_DATA_ENV="$RUN_DATA" "$VENV_PYTHON" - <<'PYEOF'
import json
import os

metrics = (
    json.loads(os.environ["RUN_DATA_ENV"])
    .get("run", {})
    .get("data", {})
    .get("metrics", [])
)

value = ""
for metric in metrics:
    if metric.get("key") == "test_auc_roc":
        value = str(metric.get("value", ""))
        break
print(value)
PYEOF
)

RUN_DATA_ENV="$RUN_DATA" "$VENV_PYTHON" - <<'PYEOF' || true
import json
import os
import sys

data = json.loads(os.environ["RUN_DATA_ENV"])

metrics = (
    data
    .get("run", {})
    .get("data", {})
    .get("metrics", [])
)

if not metrics:
    print("  No metrics found.")
    sys.exit(0)

for metric in metrics:
    k = metric.get("key", "")
    v = metric.get("value", 0)

    if any(x in k for x in ["auc", "f1", "precision", "recall", "positive"]):
        try:
            print(f"  {k:<35} : {float(v):.4f}")
        except Exception:
            print(f"  {k:<35} : {v}")
PYEOF
# =============================================================================
# Step 2 — Promote champion (promote_champion.py makes ONE fast MLflow call
#           after the workers are confirmed stable above)
# =============================================================================
step "Step 2 — Promote champion model"

detect_ips

FORCE_FLAG=""
if [ "$FORCE_PROMOTE" = true ]; then 
    FORCE_FLAG="--force"
fi

info "Running promote_champion.py …"
info "  Target : s3://deployment-models/gpon-failure-predictor/champion/model.bst"
info "  Force  : $FORCE_PROMOTE"
info "  Version: $CANDIDATE_VERSION"
info "  Metric : $CANDIDATE_METRIC"

# promote_champion.py also calls search_model_versions internally.
# We pass the version discovered via curl above so the script avoids re-running
# the flakiest registry search call. If a worker still crashes mid-call, retry.
PROMOTE_OK=false
for attempt in 1 2 3; do
    if "$VENV_PYTHON" "$REPO_DIR/monitoring/promote_champion.py" \
        --tracking-uri         http://localhost:5000 \
        --model-name           gpon-xgboost-classifier \
        --alias                champion \
        --metric-name          test_auc_roc \
        --candidate-version    "$CANDIDATE_VERSION" \
        --candidate-run-id     "$RUN_ID" \
        --candidate-source-uri "$CANDIDATE_SOURCE_URI" \
        --candidate-metric-value "$CANDIDATE_METRIC" \
        --deployment-model-uri "s3://deployment-models/gpon-failure-predictor/champion/model.bst" \
        --minio-endpoint       http://localhost:9000 \
        --skip-rollout-restart \
        --allow-alias-failure \
        $FORCE_FLAG; then
        PROMOTE_OK=true
        break
    fi
    warn "promote_champion.py failed (attempt $attempt/3) — waiting 15 s before retry …"
    sleep 15
done

[ "$PROMOTE_OK" = false ] && error "promote_champion.py failed after 3 attempts."
log "Champion model promoted."

# =============================================================================
# Step 3 — Verify artifact in MinIO
# =============================================================================
step "Step 3 — Verify artifact in MinIO"

"$VENV_PYTHON" - <<'PYEOF'
import boto3, sys
from botocore.config import Config
from botocore.exceptions import ClientError

s3 = boto3.client("s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123",
    region_name="us-east-1",
    config=Config(signature_version="s3v4"))

key = "gpon-failure-predictor/champion/model.bst"
try:
    r = s3.head_object(Bucket="deployment-models", Key=key)
    print(f"  ✔  deployment-models/{key}  ({r['ContentLength']//1024} KB)")
except ClientError as e:
    print(f"\n  ✘  Artifact not found: {e}", file=sys.stderr)
    print("  promote_champion.py may have skipped (candidate didn't beat champion).", file=sys.stderr)
    print("  Re-run with --force to promote regardless.", file=sys.stderr)
    sys.exit(1)
PYEOF

log "Artifact confirmed in MinIO."

# =============================================================================
# Step 4 — Refresh ConfigMap and apply KServe InferenceService
# =============================================================================
step "Step 4 — Apply KServe InferenceService"

info "Refreshing mlops-endpoints ConfigMap in kserve namespace …"
kubectl create configmap mlops-endpoints \
    --from-literal=MLFLOW_TRACKING_URI="http://${MLFLOW_IP}:5000" \
    --from-literal=MLFLOW_S3_ENDPOINT_URL="http://${MINIO_IP}:9000" \
    --from-literal=MINIO_ENDPOINT_URL="http://${MINIO_IP}:9000" \
    --from-literal=AWS_ACCESS_KEY_ID="minio" \
    --from-literal=AWS_SECRET_ACCESS_KEY="minio123" \
    -n kserve --dry-run=client -o yaml | kubectl apply -f -
log "ConfigMap refreshed."

if [ -f "$REPO_DIR/deployment/kserve/xgb-runtime.yaml" ]; then
    info "Applying deployment/kserve/xgb-runtime.yaml …"
    kubectl apply -f "$REPO_DIR/deployment/kserve/xgb-runtime.yaml"
    log "ServingRuntime applied."
else
    warn "deployment/kserve/xgb-runtime.yaml not found — skipping ServingRuntime apply."
fi

info "Applying deployment/kserve/inference_service.yaml …"
kubectl apply -f "$REPO_DIR/deployment/kserve/inference_service.yaml"
log "InferenceService applied."

if kubectl get deployment gpon-failure-predictor-predictor -n kserve &>/dev/null; then
    info "Restarting predictor deployment to pick up new model …"
    kubectl rollout restart deployment/gpon-failure-predictor-predictor -n kserve
    log "Rollout restart triggered."
fi

# =============================================================================
# Step 5 — Wait for Ready
# =============================================================================
if [ "$SKIP_KSERVE_WAIT" = false ]; then
    step "Step 5 — Wait for KServe InferenceService Ready"
    info "Covers: pod scheduling → model-downloader sidecar → xgbserver readiness probe"
    info "Timeout: 7 minutes"

    kubectl wait \
        --for=condition=Ready \
        inferenceservice/gpon-failure-predictor \
        -n kserve \
        --timeout=420s \
        && log "InferenceService is Ready." \
        || {
            warn "InferenceService did not report Ready within 7 min."
            warn "Diagnose with:"
            warn "  kubectl get pods -n kserve"
            warn "  kubectl describe inferenceservice -n kserve gpon-failure-predictor"
            warn "  kubectl logs -n kserve -l serving.kserve.io/inferenceservice=gpon-failure-predictor -c model-downloader"
            warn "  kubectl logs -n kserve -l serving.kserve.io/inferenceservice=gpon-failure-predictor -c kserve-container"
        }
fi

# =============================================================================
# Step 6 — Port-forward and smoke test
# =============================================================================
step "Step 6 — Port-forward KServe and smoke test"

if lsof -ti :8085 &>/dev/null; then
    info "Killing stale port-forward on :8085 …"
    lsof -ti :8085 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

info "Starting port-forward :8085 → kserve/gpon-failure-predictor-predictor:80 …"
kubectl port-forward \
    -n kserve \
    svc/gpon-failure-predictor-predictor \
    8085:80 \
    --address 0.0.0.0 \
    >/tmp/kserve-portforward.log 2>&1 &
PF_PID=$!
sleep 4

if ! kill -0 $PF_PID 2>/dev/null; then
    warn "Port-forward exited immediately — check /tmp/kserve-portforward.log"
else
    log "Port-forward running (PID $PF_PID)"
fi

info "Smoke test …"
RESPONSE=$(curl -sf --max-time 10 \
    -X POST http://localhost:8085/v1/models/gpon-failure-predictor:predict \
    -H "Content-Type: application/json" \
    -d '{"instances": [[-18.5, 2.3, 42.1, 35.2, 12, 1, 4, 730, 0, 3.3]]}' \
    2>&1 || true)

if echo "$RESPONSE" | grep -q "predictions"; then
    log "Smoke test PASSED → $RESPONSE"
else
    warn "Smoke test response: $RESPONSE"
    warn "KServe may still be initialising. Retry manually in 30 s:"
    warn "  curl -X POST http://localhost:8085/v1/models/gpon-failure-predictor:predict \\"
    warn "    -H 'Content-Type: application/json' \\"
    warn "    -d '{\"instances\": [[-18.5,2.3,42.1,35.2,12,1,4,730,0,3.3]]}'"
fi

# =============================================================================
# Summary
# =============================================================================
step "Deployment Complete"
echo ""
echo -e "${BOLD}  KServe endpoint:${NC}"
echo "  POST http://localhost:8085/v1/models/gpon-failure-predictor:predict"
echo ""
echo -e "${BOLD}  Start FastAPI gateway (separate terminal):${NC}"
echo "  export FASTAPI_GATEWAY_API_KEY=gpon-dev-key"
echo "  export KSERVE_PREDICT_URL=http://localhost:8085/v1/models/gpon-failure-predictor:predict"
echo "  make gateway-run"
echo ""
echo -e "${BOLD}  Future deployments:${NC}"
echo "  ./deploy_model.sh           # promote only if AUC improved"
echo "  ./deploy_model.sh --force   # always promote latest version"
echo ""
log "Done."
