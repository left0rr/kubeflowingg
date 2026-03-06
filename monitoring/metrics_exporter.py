"""Prometheus metrics exporter for GPON router failure-prediction pipeline.

This module exposes a ``prediction_failure_ratio`` Gauge metric via an
HTTP endpoint that Prometheus can scrape.  The ratio represents the
fraction of model predictions that indicate an imminent device failure
(``Failure_In_7_Days == 1``) over a configurable evaluation window.

Architecture:
    1. A background thread periodically loads the latest batch of
       predictions from a CSV file produced by the serving layer.
    2. The failure ratio is computed as ``count(pred == 1) / total``
       and pushed into the Prometheus Gauge.
    3. ``prometheus_client.start_http_server`` exposes ``/metrics`` on
       a configurable port (default ``8000``) for Prometheus scraping.

Integration points:
    * Prometheus service discovery in ``infrastructure/docker-compose.yml``
      should target ``<host>:8000/metrics``.
    * Alertmanager rules can fire on
      ``prediction_failure_ratio > <threshold>`` to page on-call.
    * Grafana dashboards can visualise the gauge alongside drift
      metrics from ``monitoring.drift_detection``.

Usage::

    python -m monitoring.metrics_exporter \
        --predictions data/predictions/latest.csv \
        --port 8000 \
        --interval 30

Requirements:
    prometheus_client==0.20.0
    pandas==2.2.2
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from prometheus_client import Gauge, start_http_server

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

PREDICTION_FAILURE_RATIO = Gauge(
    "prediction_failure_ratio",
    "Fraction of model predictions indicating device failure within 7 days "
    "(range 0.0–1.0).  Computed over the latest prediction batch.",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COLUMN = "Failure_In_7_Days"
DEFAULT_PREDICTIONS_PATH = Path("data/predictions/latest.csv")
DEFAULT_PORT = 8000
DEFAULT_INTERVAL_SECONDS = 30


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def configure_logging(level: int = logging.INFO) -> None:
    """Set up structured logging for the metrics exporter.

    Args:
        level: Python logging level.  Defaults to ``logging.INFO``.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------
def load_predictions(path: Path) -> Optional[pd.DataFrame]:
    """Load the latest prediction batch from a CSV file.

    Returns ``None`` instead of raising when the file is missing so
    the update loop can degrade gracefully during cold-start or when
    the serving layer has not yet produced its first output.

    Args:
        path: Filesystem path to the predictions CSV.

    Returns:
        DataFrame with at least a ``Failure_In_7_Days`` column,
        or ``None`` if the file does not exist or is unreadable.
    """
    if not path.exists():
        logger.warning("Predictions file not found: %s", path)
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        logger.exception("Failed to read predictions from %s", path)
        return None

    if TARGET_COLUMN not in df.columns:
        logger.error(
            "Column '%s' missing from predictions file. "
            "Available columns: %s",
            TARGET_COLUMN,
            list(df.columns),
        )
        return None

    logger.debug("Loaded %d predictions from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_failure_ratio(df: pd.DataFrame) -> float:
    """Compute the ratio of positive failure predictions.

    Args:
        df: DataFrame containing a ``Failure_In_7_Days`` column with
            binary values (0 or 1).

    Returns:
        Float in ``[0.0, 1.0]`` representing the fraction of rows
        where the model predicted a failure.  Returns ``0.0`` when the
        DataFrame is empty.
    """
    if df.empty:
        logger.warning("Empty predictions DataFrame; returning 0.0")
        return 0.0

    total = len(df)
    failures = int((df[TARGET_COLUMN] == 1).sum())
    ratio = failures / total

    logger.info(
        "Failure ratio: %d / %d = %.4f",
        failures,
        total,
        ratio,
    )
    return ratio


def update_metric(predictions_path: Path) -> float:
    """Load predictions and push the failure ratio into the Gauge.

    This is the single atomic "tick" of the update loop.  It is
    intentionally separated from the loop itself so that callers
    (tests, Kubeflow components) can trigger a one-shot update.

    Args:
        predictions_path: Path to the predictions CSV.

    Returns:
        The computed failure ratio, or ``-1.0`` if the file could not
        be loaded (the Gauge is **not** modified in that case).
    """
    df = load_predictions(predictions_path)
    if df is None:
        return -1.0

    ratio = compute_failure_ratio(df)
    PREDICTION_FAILURE_RATIO.set(ratio)
    return ratio


# ---------------------------------------------------------------------------
# Background update loop
# ---------------------------------------------------------------------------
def _update_loop(
    predictions_path: Path,
    interval_seconds: int,
    stop_event: threading.Event,
) -> None:
    """Periodically refresh the Prometheus Gauge.

    Runs in a daemon thread started by :func:`start_update_loop`.
    Each iteration calls :func:`update_metric` and then sleeps for
    *interval_seconds*.  The loop exits cleanly when *stop_event*
    is set.

    Args:
        predictions_path: Path to the predictions CSV.
        interval_seconds: Seconds between successive refreshes.
        stop_event: Threading event used to signal graceful shutdown.
    """
    logger.info(
        "Update loop started — refreshing every %ds from %s",
        interval_seconds,
        predictions_path,
    )

    while not stop_event.is_set():
        update_metric(predictions_path)
        stop_event.wait(timeout=interval_seconds)

    logger.info("Update loop stopped")


def start_update_loop(
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> tuple:
    """Launch the background metric-refresh thread.

    Args:
        predictions_path: Path to the predictions CSV.
        interval_seconds: Seconds between refreshes.

    Returns:
        A ``(thread, stop_event)`` tuple.  Set ``stop_event`` to
        terminate the loop and then ``thread.join()`` to wait for
        clean exit.
    """
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_update_loop,
        args=(predictions_path, interval_seconds, stop_event),
        daemon=True,
        name="metrics-update-loop",
    )
    thread.start()
    return thread, stop_event


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
def start_metrics_server(port: int = DEFAULT_PORT) -> None:
    """Start the Prometheus HTTP metrics server.

    The server runs in a background daemon thread and exposes the
    default ``/metrics`` endpoint on *port*.

    Args:
        port: TCP port to bind.  Defaults to ``8000``.
    """
    start_http_server(port)
    logger.info("Prometheus metrics server listening on :%d/metrics", port)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------
def run_exporter(
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    port: int = DEFAULT_PORT,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> None:
    """Start the metrics server and enter the update loop.

    This is the main entry-point used by both the CLI and container
    ``CMD``.  It blocks indefinitely until interrupted.

    Args:
        predictions_path: Path to the predictions CSV.
        port: Prometheus HTTP port.
        interval_seconds: Gauge refresh interval.
    """
    logger.info("=== Prometheus Metrics Exporter START ===")

    # 1 — Expose /metrics
    start_metrics_server(port)

    # 2 — Seed the gauge with an initial value
    update_metric(predictions_path)

    # 3 — Enter blocking update loop on the main thread
    logger.info(
        "Entering update loop (Ctrl+C to stop) — "
        "predictions=%s, interval=%ds",
        predictions_path,
        interval_seconds,
    )

    stop_event = threading.Event()

    try:
        _update_loop(predictions_path, interval_seconds, stop_event)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — shutting down")
        stop_event.set()

    logger.info("=== Prometheus Metrics Exporter STOPPED ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry-point for the Prometheus metrics exporter."""
    parser = argparse.ArgumentParser(
        description=(
            "Expose GPON failure-prediction metrics to Prometheus.  "
            "Reads the latest prediction batch and publishes "
            "prediction_failure_ratio as a Gauge on /metrics."
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Path to the predictions CSV file.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="TCP port for the Prometheus HTTP server.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Seconds between metric refreshes.",
    )
    args = parser.parse_args()

    configure_logging()
    run_exporter(
        predictions_path=args.predictions,
        port=args.port,
        interval_seconds=args.interval,
    )


if __name__ == "__main__":
    main()
