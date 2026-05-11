# FastAPI Inference Gateway

This folder documents the FastAPI gateway that now sits in front of the KServe
model endpoint.

## Why This Gateway Exists

KServe should stay focused on one job:

- receive model features
- return a prediction score

The FastAPI gateway handles the operational concerns around that model call:

- API-key authentication
- lightweight per-client rate limiting
- Pydantic request validation
- preserving router metadata that should not be part of the model features
- central prediction logging for monitoring, alerting, and future retraining

## High-Level Flow

```text
simulate_trafic.py
        |
        v
FastAPI /predict
        |
        +--> validate metadata + feature ranges
        +--> enforce X-API-Key
        +--> apply rate limiting
        +--> forward only the 10 model features to KServe
        +--> write monitoring log
        +--> write enriched event log
        +--> optionally write labeled feedback log
        |
        v
KServe XGBoost model
```

## Request Shape

The gateway accepts rich requests such as:

```json
{
  "device_id": "tn_router_00001",
  "router_serial_number": "TN-ONT-00001-4821",
  "telecom_number": "+21620123456",
  "timestamp": "2026-04-18T12:00:00+00:00",
  "features": {
    "Optical_RX_Power_dBm": -18.5,
    "Optical_TX_Power_dBm": 2.3,
    "Temperature_C": 42.1,
    "Bias_Current_mA": 35.2,
    "Interface_Error_Count": 12,
    "Reboot_Count_Last_7D": 1,
    "Connected_Devices": 4,
    "Device_Age_Days": 730,
    "Maintenance_Count_Last_30D": 0,
    "Voltage_V": 3.3
  },
  "source_mode": "baseline-replay",
  "drift_profile": "none",
  "drift_applied": false,
  "true_status": "Nominal",
  "true_failure_in_7_days": 0
}
```

The important design choice is:

- router identity lives in the gateway request and event logs
- only `features` are forwarded to KServe

## Log Files

The gateway writes three different CSV outputs:

### 1. Monitoring log

Default path:

```text
data/predictions/latest.csv
```

Used by:

- `monitoring.metrics_exporter`
- `monitoring.drift_detection`

This file stays compatible with the existing monitoring workflow.

### 2. Enriched event log

Default path:

```text
data/predictions/prediction_events.csv
```

Used for:

- future email alerting
- router/operator investigation
- joining predictions back to a router identity

This file includes:

- `device_id`
- `router_serial_number`
- `telecom_number`
- full features
- prediction score
- alert candidate flag

### 3. Labeled feedback log

Default path:

```text
data/feedback/labeled_feedback.csv
```

Used for:

- future retraining data assembly

This log is only written when the request includes `true_failure_in_7_days`.

## Environment Variables

You can run the gateway with defaults, but these variables are supported:

- `FASTAPI_GATEWAY_API_KEY`
- `FASTAPI_RATE_LIMIT_PER_MINUTE`
- `KSERVE_PREDICT_URL`
- `KSERVE_TIMEOUT_SECONDS`
- `ALERT_SCORE_THRESHOLD`
- `MONITORING_LOG_PATH`
- `PREDICTION_EVENT_LOG_PATH`
- `LABELED_FEEDBACK_LOG_PATH`

## Run Locally

### 1. Port-forward KServe

```bash
kubectl port-forward -n kserve svc/gpon-failure-predictor-predictor 8085:80
```

### 2. Start the gateway

```bash
export FASTAPI_GATEWAY_API_KEY=gpon-dev-key
uvicorn src.api.inference_gateway:app --host 0.0.0.0 --port 8010
```

Or with Makefile:

```bash
make gateway-run
```

### 3. Start the simulator against FastAPI

```bash
python simulate_trafic.py \
  --target fastapi \
  --gateway-url http://localhost:8010/predict \
  --api-key gpon-dev-key
```

## Tunisia Phone Numbers

The simulator now generates telecom numbers in Tunisian format:

```text
+216XXXXXXXX
```

This is intentional so the future alerting flow looks close to a real ISP
operations workflow.
