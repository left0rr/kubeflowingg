"""FastAPI gateway for authenticated, rate-limited inference requests.

This service sits in front of the KServe model endpoint. It is responsible for:

1. Authenticating callers with an API key.
2. Applying lightweight request rate limiting.
3. Validating business metadata and model feature ranges.
4. Forwarding only model features to KServe.
5. Writing both monitoring-friendly and operations-friendly logs.

The gateway keeps KServe focused on pure inference while the gateway handles
operational concerns such as metadata preservation, audit trails, and later
alerting hooks.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import threading
import time
from typing import Deque, Dict, List, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel, Field, field_validator
import requests

logger = logging.getLogger(__name__)
csv_write_lock = threading.Lock()

FEATURE_NAMES: List[str] = [
    "Optical_RX_Power_dBm",
    "Optical_TX_Power_dBm",
    "Temperature_C",
    "Bias_Current_mA",
    "Interface_Error_Count",
    "Reboot_Count_Last_7D",
    "Connected_Devices",
    "Device_Age_Days",
    "Maintenance_Count_Last_30D",
    "Voltage_V",
]

MONITORING_LOG_COLUMNS = FEATURE_NAMES + [
    "timestamp",
    "prediction_score",
    "predicted_failure_label",
    "Failure_In_7_Days",
    "true_status",
    "source_mode",
    "drift_profile",
    "drift_applied",
]

EVENT_LOG_COLUMNS = [
    "event_id",
    "device_id",
    "router_serial_number",
    "telecom_number",
    "timestamp",
] + FEATURE_NAMES + [
    "prediction_score",
    "predicted_failure_label",
    "true_failure_in_7_days",
    "true_status",
    "source_mode",
    "drift_profile",
    "drift_applied",
    "alert_candidate",
]

FEEDBACK_LOG_COLUMNS = [
    "event_id",
    "device_id",
    "router_serial_number",
    "telecom_number",
    "timestamp",
] + FEATURE_NAMES + [
    "true_failure_in_7_days",
    "prediction_score",
    "predicted_failure_label",
    "true_status",
    "source_mode",
    "drift_profile",
    "drift_applied",
]


def configure_logging() -> None:
    """Set up structured logging for the inference gateway."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@dataclass(frozen=True)
class GatewaySettings:
    """Runtime configuration for the gateway."""

    api_key: str
    kserve_url: str
    rate_limit_per_minute: int
    kserve_timeout_seconds: float
    alert_threshold: float
    monitoring_log_path: Path
    event_log_path: Path
    feedback_log_path: Path


def load_settings() -> GatewaySettings:
    """Load gateway settings from environment variables."""
    return GatewaySettings(
        api_key=os.getenv("FASTAPI_GATEWAY_API_KEY", "gpon-dev-key"),
        kserve_url=os.getenv(
            "KSERVE_PREDICT_URL",
            "http://localhost:8085/v1/models/gpon-failure-predictor:predict",
        ),
        rate_limit_per_minute=int(os.getenv("FASTAPI_RATE_LIMIT_PER_MINUTE", "60")),
        kserve_timeout_seconds=float(os.getenv("KSERVE_TIMEOUT_SECONDS", "5")),
        alert_threshold=float(os.getenv("ALERT_SCORE_THRESHOLD", "0.80")),
        monitoring_log_path=Path(
            os.getenv("MONITORING_LOG_PATH", "data/predictions/latest.csv")
        ),
        event_log_path=Path(
            os.getenv("PREDICTION_EVENT_LOG_PATH", "data/predictions/prediction_events.csv")
        ),
        feedback_log_path=Path(
            os.getenv("LABELED_FEEDBACK_LOG_PATH", "data/feedback/labeled_feedback.csv")
        ),
    )


class FixedWindowRateLimiter:
    """Small in-memory per-key rate limiter for local development."""

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Return True if the caller may proceed, else False."""
        now = time.time()
        with self._lock:
            bucket = self._buckets[key]
            while bucket and now - bucket[0] >= self.window_seconds:
                bucket.popleft()

            if len(bucket) >= self.limit:
                return False

            bucket.append(now)
            return True


class TelemetryFeatures(BaseModel):
    """Feature payload sent onward to KServe."""

    Optical_RX_Power_dBm: float = Field(..., ge=-40.0, le=0.0)
    Optical_TX_Power_dBm: float = Field(..., ge=-10.0, le=10.0)
    Temperature_C: float = Field(..., ge=-40.0, le=125.0)
    Bias_Current_mA: float = Field(..., ge=0.0, le=200.0)
    Interface_Error_Count: int = Field(..., ge=0)
    Reboot_Count_Last_7D: int = Field(..., ge=0)
    Connected_Devices: int = Field(..., ge=0)
    Device_Age_Days: int = Field(..., ge=0)
    Maintenance_Count_Last_30D: int = Field(..., ge=0)
    Voltage_V: float = Field(..., ge=0.0, le=5.0)


class PredictionRequest(BaseModel):
    """Authenticated inference request accepted by the gateway."""

    device_id: str = Field(..., min_length=3, max_length=128)
    router_serial_number: str = Field(..., min_length=3, max_length=128)
    telecom_number: str = Field(..., description="Subscriber number in Tunisian +216 format.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    features: TelemetryFeatures
    source_mode: str = Field(default="fastapi-gateway")
    drift_profile: str = Field(default="none")
    drift_applied: bool = Field(default=False)
    true_status: Optional[str] = Field(default=None)
    true_failure_in_7_days: Optional[int] = Field(default=None, ge=0, le=1)

    @field_validator("telecom_number")
    @classmethod
    def validate_tunisian_number(cls, value: str) -> str:
        """Require Tunisian phone numbers for the simulated operator workflow."""
        normalized = value.strip()
        if not normalized.startswith("+216") or len(normalized) != 12 or not normalized[1:].isdigit():
            raise ValueError(
                "telecom_number must use Tunisian format +216XXXXXXXX"
            )
        return normalized


class PredictionResponse(BaseModel):
    """Gateway response returned to clients."""

    event_id: str
    prediction_score: float
    predicted_failure_label: int
    alert_candidate: bool
    monitoring_log_path: str
    event_log_path: str
    feedback_logged: bool


def _ordered_feature_list(features: TelemetryFeatures) -> List[float]:
    """Convert validated features to the exact order expected by KServe."""
    return [float(getattr(features, feature_name)) for feature_name in FEATURE_NAMES]


def _append_csv_row(path: Path, columns: List[str], row: Dict[str, object]) -> None:
    """Append one row to a CSV file, creating headers if needed."""
    with csv_write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = not path.exists()
        pd.DataFrame([row], columns=columns).to_csv(
            path,
            mode="a",
            header=header,
            index=False,
        )


def _build_monitoring_row(
    request_data: PredictionRequest,
    prediction_score: float,
    predicted_failure_label: int,
) -> Dict[str, object]:
    """Build the monitoring-friendly row used by drift detection and metrics exporter."""
    row = request_data.features.model_dump()
    row["timestamp"] = request_data.timestamp.isoformat()
    row["prediction_score"] = prediction_score
    row["predicted_failure_label"] = predicted_failure_label
    row["Failure_In_7_Days"] = ""
    row["true_status"] = request_data.true_status or ""
    row["source_mode"] = request_data.source_mode
    row["drift_profile"] = request_data.drift_profile
    row["drift_applied"] = request_data.drift_applied
    return row


def _build_event_row(
    event_id: str,
    request_data: PredictionRequest,
    prediction_score: float,
    predicted_failure_label: int,
    alert_candidate: bool,
) -> Dict[str, object]:
    """Build the enriched event row used for future alerting and auditing."""
    row: Dict[str, object] = {
        "event_id": event_id,
        "device_id": request_data.device_id,
        "router_serial_number": request_data.router_serial_number,
        "telecom_number": request_data.telecom_number,
        "timestamp": request_data.timestamp.isoformat(),
    }
    row.update(request_data.features.model_dump())
    row["prediction_score"] = prediction_score
    row["predicted_failure_label"] = predicted_failure_label
    row["true_failure_in_7_days"] = (
        request_data.true_failure_in_7_days
        if request_data.true_failure_in_7_days is not None
        else ""
    )
    row["true_status"] = request_data.true_status or ""
    row["source_mode"] = request_data.source_mode
    row["drift_profile"] = request_data.drift_profile
    row["drift_applied"] = request_data.drift_applied
    row["alert_candidate"] = alert_candidate
    return row


def _build_feedback_row(
    event_id: str,
    request_data: PredictionRequest,
    prediction_score: float,
    predicted_failure_label: int,
) -> Optional[Dict[str, object]]:
    """Build an optional labeled-feedback row for simulated retraining workflows."""
    if request_data.true_failure_in_7_days is None:
        return None

    row: Dict[str, object] = {
        "event_id": event_id,
        "device_id": request_data.device_id,
        "router_serial_number": request_data.router_serial_number,
        "telecom_number": request_data.telecom_number,
        "timestamp": request_data.timestamp.isoformat(),
    }
    row.update(request_data.features.model_dump())
    row["true_failure_in_7_days"] = request_data.true_failure_in_7_days
    row["prediction_score"] = prediction_score
    row["predicted_failure_label"] = predicted_failure_label
    row["true_status"] = request_data.true_status or ""
    row["source_mode"] = request_data.source_mode
    row["drift_profile"] = request_data.drift_profile
    row["drift_applied"] = request_data.drift_applied
    return row


configure_logging()
settings = load_settings()
rate_limiter = FixedWindowRateLimiter(limit=settings.rate_limit_per_minute)
app = FastAPI(
    title="GPON Inference Gateway",
    version="1.0.0",
    description=(
        "Authenticated FastAPI gateway that forwards validated GPON telemetry "
        "requests to the KServe XGBoost model."
    ),
)


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> str:
    """Require callers to provide the configured gateway API key."""
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )
    return x_api_key


@app.get("/health")
def health() -> Dict[str, object]:
    """Health endpoint for local smoke testing."""
    return {
        "status": "ok",
        "kserve_url": settings.kserve_url,
        "rate_limit_per_minute": settings.rate_limit_per_minute,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    request: Request,
    api_key: str = Depends(require_api_key),
) -> PredictionResponse:
    """Validate a rich inference request, call KServe, and log the outcome."""
    client_host = request.client.host if request.client and request.client.host else "unknown"
    client_key = f"{client_host}:{api_key}"
    if not rate_limiter.allow(client_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please retry later.",
        )

    feature_values = _ordered_feature_list(payload.features)
    kserve_payload = {"instances": [feature_values]}

    try:
        response = requests.post(
            settings.kserve_url,
            json=kserve_payload,
            timeout=settings.kserve_timeout_seconds,
        )
        response.raise_for_status()
        prediction_score = float(response.json()["predictions"][0])
    except requests.RequestException as exc:
        logger.exception("KServe request failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"KServe request failed: {exc}",
        ) from exc
    except (KeyError, TypeError, ValueError) as exc:
        logger.exception("Unexpected KServe response payload")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unexpected KServe response payload: {exc}",
        ) from exc

    predicted_failure_label = 1 if prediction_score >= 0.5 else 0
    alert_candidate = prediction_score >= settings.alert_threshold
    event_id = str(uuid4())

    monitoring_row = _build_monitoring_row(payload, prediction_score, predicted_failure_label)
    event_row = _build_event_row(
        event_id=event_id,
        request_data=payload,
        prediction_score=prediction_score,
        predicted_failure_label=predicted_failure_label,
        alert_candidate=alert_candidate,
    )
    feedback_row = _build_feedback_row(
        event_id=event_id,
        request_data=payload,
        prediction_score=prediction_score,
        predicted_failure_label=predicted_failure_label,
    )

    _append_csv_row(settings.monitoring_log_path, MONITORING_LOG_COLUMNS, monitoring_row)
    _append_csv_row(settings.event_log_path, EVENT_LOG_COLUMNS, event_row)
    if feedback_row is not None:
        _append_csv_row(settings.feedback_log_path, FEEDBACK_LOG_COLUMNS, feedback_row)

    logger.info(
        "Prediction logged for device_id=%s event_id=%s score=%.4f alert_candidate=%s",
        payload.device_id,
        event_id,
        prediction_score,
        alert_candidate,
    )

    return PredictionResponse(
        event_id=event_id,
        prediction_score=prediction_score,
        predicted_failure_label=predicted_failure_label,
        alert_candidate=alert_candidate,
        monitoring_log_path=str(settings.monitoring_log_path),
        event_log_path=str(settings.event_log_path),
        feedback_logged=feedback_row is not None,
    )


@app.exception_handler(Exception)
def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Log unexpected exceptions in a user-friendly way."""
    logger.exception("Unhandled gateway exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Unhandled gateway exception"},
    )
