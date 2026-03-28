"""Prometheus metrics exporter for GPON router failure-prediction pipeline.

This module exposes a ``prediction_failure_ratio`` Gauge via an HTTP
endpoint that Prometheus can scrape. The ratio is computed over a recent
window of predictions so the signal reacts to fresh regressions rather
than the full lifetime of the CSV log.

Usage::

    python -m monitoring.metrics_exporter \
        --predictions data/predictions/latest.csv \
        --port 8000 \
        --interval 30 \
        --window-rows 300
"""

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import Optional

import pandas as pd
from prometheus_client import Gauge, start_http_server

logger = logging.getLogger(__name__)

PREDICTION_FAILURE_RATIO = Gauge(
    "prediction_failure_ratio",
    "Fraction of recent model predictions indicating device failure.",
)
PREDICTION_WINDOW_SAMPLE_SIZE = Gauge(
    "prediction_window_sample_size",
    "Number of predictions included in the most recent monitoring window.",
)

PREDICTION_LABEL_COLUMNS = (
    "predicted_failure_label",
    "Failure_In_7_Days",
)
TIMESTAMP_COLUMN = "timestamp"

DEFAULT_PREDICTIONS_PATH = Path("data/predictions/latest.csv")
DEFAULT_PORT = 8000
DEFAULT_INTERVAL_SECONDS = 30
DEFAULT_WINDOW_ROWS = 300


def configure_logging(level: int = logging.INFO) -> None:
    """Set up structured logging for the metrics exporter."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_predictions(path: Path) -> Optional[pd.DataFrame]:
    """Load the latest prediction log from disk."""
    if not path.exists():
        logger.warning("Predictions file not found: %s", path)
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        logger.exception("Failed to read predictions from %s", path)
        return None

    logger.debug("Loaded %d predictions from %s", len(df), path)
    return df


def resolve_prediction_label_column(df: pd.DataFrame) -> str:
    """Return the prediction-label column used by the exporter."""
    for candidate in PREDICTION_LABEL_COLUMNS:
        if candidate in df.columns:
            return candidate

    raise KeyError(
        "No prediction label column found. Expected one of "
        f"{list(PREDICTION_LABEL_COLUMNS)}; available columns: {list(df.columns)}"
    )


def select_recent_predictions(
    df: pd.DataFrame,
    window_rows: Optional[int] = DEFAULT_WINDOW_ROWS,
    window_minutes: Optional[int] = None,
) -> pd.DataFrame:
    """Slice the prediction log down to a recent monitoring window."""
    if df.empty:
        return df

    if window_minutes is not None and TIMESTAMP_COLUMN in df.columns:
        timestamped_df = df.copy()
        timestamped_df[TIMESTAMP_COLUMN] = pd.to_datetime(
            timestamped_df[TIMESTAMP_COLUMN],
            errors="coerce",
        )
        timestamped_df = timestamped_df.dropna(subset=[TIMESTAMP_COLUMN])

        if not timestamped_df.empty:
            latest_ts = timestamped_df[TIMESTAMP_COLUMN].max()
            cutoff_ts = latest_ts - pd.Timedelta(minutes=window_minutes)
            recent_df = timestamped_df[
                timestamped_df[TIMESTAMP_COLUMN] >= cutoff_ts
            ].reset_index(drop=True)
            logger.info(
                "Using %d prediction rows from the last %d minute(s)",
                len(recent_df),
                window_minutes,
            )
            return recent_df

        logger.warning(
            "Could not parse '%s' for time-window slicing; falling back to row window",
            TIMESTAMP_COLUMN,
        )

    if window_rows is not None and window_rows > 0 and len(df) > window_rows:
        recent_df = df.tail(window_rows).reset_index(drop=True)
        logger.info("Using the last %d prediction row(s)", len(recent_df))
        return recent_df

    logger.info("Using all %d available prediction row(s)", len(df))
    return df.reset_index(drop=True)


def compute_failure_ratio(
    df: pd.DataFrame,
    target_column: str,
) -> float:
    """Compute the ratio of positive failure predictions."""
    if df.empty:
        logger.warning("Empty predictions DataFrame; returning 0.0")
        return 0.0

    total = len(df)
    failures = int((df[target_column] == 1).sum())
    ratio = failures / total

    logger.info(
        "Failure ratio over current window: %d / %d = %.4f",
        failures,
        total,
        ratio,
    )
    return ratio


def update_metric(
    predictions_path: Path,
    window_rows: Optional[int] = DEFAULT_WINDOW_ROWS,
    window_minutes: Optional[int] = None,
) -> float:
    """Load predictions and push the current failure ratio into the Gauge."""
    df = load_predictions(predictions_path)
    if df is None:
        return -1.0

    recent_df = select_recent_predictions(
        df,
        window_rows=window_rows,
        window_minutes=window_minutes,
    )

    try:
        target_column = resolve_prediction_label_column(recent_df)
    except KeyError:
        logger.exception("Prediction label column missing")
        return -1.0

    ratio = compute_failure_ratio(recent_df, target_column=target_column)
    PREDICTION_FAILURE_RATIO.set(ratio)
    PREDICTION_WINDOW_SAMPLE_SIZE.set(float(len(recent_df)))
    return ratio


def _update_loop(
    predictions_path: Path,
    interval_seconds: int,
    stop_event: threading.Event,
    window_rows: Optional[int],
    window_minutes: Optional[int],
) -> None:
    """Periodically refresh the Prometheus Gauges."""
    logger.info(
        "Update loop started: interval=%ds, predictions=%s, window_rows=%s, "
        "window_minutes=%s",
        interval_seconds,
        predictions_path,
        window_rows,
        window_minutes,
    )

    while not stop_event.is_set():
        update_metric(
            predictions_path,
            window_rows=window_rows,
            window_minutes=window_minutes,
        )
        stop_event.wait(timeout=interval_seconds)

    logger.info("Update loop stopped")


def start_metrics_server(port: int = DEFAULT_PORT) -> None:
    """Start the Prometheus HTTP metrics server."""
    start_http_server(port)
    logger.info("Prometheus metrics server listening on :%d/metrics", port)


def run_exporter(
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    port: int = DEFAULT_PORT,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    window_rows: Optional[int] = DEFAULT_WINDOW_ROWS,
    window_minutes: Optional[int] = None,
) -> None:
    """Start the metrics server and enter the update loop."""
    logger.info("=== Prometheus Metrics Exporter START ===")
    start_metrics_server(port)

    update_metric(
        predictions_path,
        window_rows=window_rows,
        window_minutes=window_minutes,
    )

    stop_event = threading.Event()

    try:
        _update_loop(
            predictions_path,
            interval_seconds,
            stop_event,
            window_rows,
            window_minutes,
        )
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received; shutting down")
        stop_event.set()

    logger.info("=== Prometheus Metrics Exporter STOPPED ===")


def main() -> None:
    """CLI entry-point for the Prometheus metrics exporter."""
    parser = argparse.ArgumentParser(
        description=(
            "Expose GPON failure-prediction metrics to Prometheus. Reads the "
            "latest prediction log and publishes prediction_failure_ratio as a "
            "Gauge on /metrics."
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
    parser.add_argument(
        "--window-rows",
        type=int,
        default=DEFAULT_WINDOW_ROWS,
        help="Use only the most recent N rows when computing the ratio.",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=None,
        help="Use only rows from the last N minutes when timestamps are present.",
    )
    args = parser.parse_args()

    configure_logging()
    run_exporter(
        predictions_path=args.predictions,
        port=args.port,
        interval_seconds=args.interval,
        window_rows=args.window_rows,
        window_minutes=args.window_minutes,
    )


if __name__ == "__main__":
    main()
