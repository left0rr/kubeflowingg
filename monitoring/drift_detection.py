"""Data drift detection for GPON router failure-prediction pipeline.

This module compares a baseline training dataset against a recent window
of production-style predictions. The current dataset can be sliced by
row count or by timestamp so the drift signal reflects fresh behavior
instead of the full lifetime of the CSV log.

Usage::

    python -m monitoring.drift_detection \
        --baseline data/processed/processed.csv \
        --current data/predictions/latest.csv \
        --output monitoring/reports/drift_report.html \
        --current-window-rows 500
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from prometheus_client import CollectorRegistry, Gauge, write_to_textfile

logger = logging.getLogger(__name__)

TIMESTAMP_COLUMN = "timestamp"
NON_FEATURE_COLUMNS: List[str] = [
    "Failure_In_7_Days",
    "predicted_failure_label",
    TIMESTAMP_COLUMN,
    "prediction_score",
    "true_status",
    "source_mode",
    "drift_profile",
    "drift_applied",
]

DEFAULT_BASELINE_PATH = Path("data/processed/processed.csv")
DEFAULT_OUTPUT_PATH = Path("monitoring/reports/drift_report.html")
DEFAULT_PROMETHEUS_TEXTFILE_PATH = Path("monitoring/prometheus_textfile/drift_detection.prom")
DEFAULT_CURRENT_WINDOW_ROWS = 500
DEFAULT_MIN_CURRENT_ROWS = 100


@dataclass
class DriftResult:
    """Container for drift detection artefacts."""

    report: Report
    report_path: Path
    drift_detected: bool
    drifted_columns: List[str]
    drift_summary: Dict
    shared_columns: List[str]
    drift_by_column: Dict[str, Dict[str, Optional[float]]]
    baseline_row_count: int
    current_row_count: int


def configure_logging(level: int = logging.INFO) -> None:
    """Set up structured logging for the drift-detection pipeline."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_dataset(path: Path, label: str = "dataset") -> pd.DataFrame:
    """Load a CSV dataset from disk."""
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")

    logger.info("Loading %s dataset from %s", label, path)
    df = pd.read_csv(path)
    logger.info(
        "%s dataset: %d rows x %d columns",
        label.capitalize(),
        len(df),
        len(df.columns),
    )
    return df


def select_recent_rows(
    df: pd.DataFrame,
    label: str,
    window_rows: Optional[int] = DEFAULT_CURRENT_WINDOW_ROWS,
    window_minutes: Optional[int] = None,
) -> pd.DataFrame:
    """Restrict a dataset to a recent monitoring window."""
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
                "Using %d %s row(s) from the last %d minute(s)",
                len(recent_df),
                label,
                window_minutes,
            )
            return recent_df

        logger.warning(
            "Could not parse '%s' in %s dataset; falling back to row window",
            TIMESTAMP_COLUMN,
            label,
        )

    if window_rows is not None and window_rows > 0 and len(df) > window_rows:
        recent_df = df.tail(window_rows).reset_index(drop=True)
        logger.info("Using the last %d %s row(s)", len(recent_df), label)
        return recent_df

    logger.info("Using all %d available %s row(s)", len(df), label)
    return df.reset_index(drop=True)


def select_feature_columns(
    df: pd.DataFrame,
    exclude: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Drop metadata columns so drift analysis covers model inputs only."""
    if exclude is None:
        exclude = NON_FEATURE_COLUMNS

    cols_to_drop = [column for column in exclude if column in df.columns]
    if cols_to_drop:
        logger.info("Excluding non-feature columns: %s", cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    logger.info("Feature columns for drift analysis: %s", list(df.columns))
    return df


def align_shared_columns(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Align reference and current datasets to their shared feature columns."""
    common_cols = sorted(set(reference.columns) & set(current.columns))
    if not common_cols:
        raise ValueError(
            "No overlapping columns between reference and current datasets. "
            f"Reference: {list(reference.columns)}, "
            f"Current: {list(current.columns)}"
        )

    reference = reference[common_cols].copy()
    current = current[common_cols].copy()
    return reference, current, common_cols


def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> Tuple[Report, List[str]]:
    """Create and execute an Evidently DataDriftPreset report."""
    reference, current, common_cols = align_shared_columns(reference, current)
    logger.info("Building drift report with %d shared feature columns", len(common_cols))

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    logger.info("Drift report executed successfully")
    return report, common_cols


def save_report(report: Report, output_path: Path) -> Path:
    """Persist the Evidently report as a self-contained HTML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    logger.info("Drift report saved to %s", output_path)
    return output_path


def extract_drift_summary(report: Report) -> Dict:
    """Convert the Evidently report to a Python dictionary."""
    return report.as_dict()


def parse_drifted_columns(summary: Dict) -> List[str]:
    """Extract the list of columns that exhibited statistically significant drift."""
    drifted: List[str] = []

    for metric_result in summary.get("metrics", []):
        metric_data = metric_result.get("result", {})
        drift_by_columns = metric_data.get("drift_by_columns", {})

        for col_name, col_info in drift_by_columns.items():
            if col_info.get("drift_detected", False):
                drifted.append(col_name)

    drifted.sort()
    return drifted


def is_dataset_drift_detected(summary: Dict) -> bool:
    """Check whether dataset-level drift was flagged."""
    for metric_result in summary.get("metrics", []):
        metric_data = metric_result.get("result", {})
        if "dataset_drift" in metric_data:
            return bool(metric_data["dataset_drift"])
    return False


def extract_column_drift_metrics(summary: Dict) -> Dict[str, Dict[str, Optional[float]]]:
    """Extract per-column drift flags and scores from the Evidently summary."""
    result: Dict[str, Dict[str, Optional[float]]] = {}

    for metric_result in summary.get("metrics", []):
        metric_data = metric_result.get("result", {})
        drift_by_columns = metric_data.get("drift_by_columns", {})

        for col_name, col_info in drift_by_columns.items():
            drift_score = col_info.get("drift_score")
            result[col_name] = {
                "drift_detected": float(bool(col_info.get("drift_detected", False))),
                "drift_score": float(drift_score) if isinstance(drift_score, (int, float)) else None,
            }

    return result


def write_prometheus_textfile(
    drift_result: DriftResult,
    output_path: Path,
) -> Path:
    """Publish drift metrics via the Prometheus textfile collector format."""
    registry = CollectorRegistry()

    dataset_drift_metric = Gauge(
        "gpon_data_drift_detected",
        "Whether the last GPON drift detection run found dataset-level drift.",
        registry=registry,
    )
    drifted_columns_metric = Gauge(
        "gpon_drifted_columns_count",
        "Number of feature columns flagged as drifted in the last run.",
        registry=registry,
    )
    drift_fraction_metric = Gauge(
        "gpon_drifted_columns_fraction",
        "Fraction of shared feature columns flagged as drifted in the last run.",
        registry=registry,
    )
    drift_check_success_metric = Gauge(
        "gpon_drift_check_success",
        "Whether the last GPON drift detection run completed successfully.",
        registry=registry,
    )
    drift_check_timestamp_metric = Gauge(
        "gpon_drift_check_timestamp_seconds",
        "Unix timestamp of the last GPON drift detection run.",
        registry=registry,
    )
    baseline_rows_metric = Gauge(
        "gpon_drift_baseline_rows",
        "Number of baseline rows used during the last GPON drift detection run.",
        registry=registry,
    )
    current_rows_metric = Gauge(
        "gpon_drift_current_rows",
        "Number of current rows used during the last GPON drift detection run.",
        registry=registry,
    )
    shared_columns_metric = Gauge(
        "gpon_drift_shared_feature_columns",
        "Number of shared feature columns compared during the last GPON drift detection run.",
        registry=registry,
    )
    feature_drift_metric = Gauge(
        "gpon_feature_drift_detected",
        "Whether a specific GPON feature was flagged as drifted in the last run.",
        labelnames=["feature"],
        registry=registry,
    )
    feature_drift_score_metric = Gauge(
        "gpon_feature_drift_score",
        "Drift score reported by Evidently for a specific GPON feature in the last run.",
        labelnames=["feature"],
        registry=registry,
    )

    dataset_drift_metric.set(1.0 if drift_result.drift_detected else 0.0)
    drifted_columns_metric.set(float(len(drift_result.drifted_columns)))
    drift_fraction_metric.set(
        float(len(drift_result.drifted_columns)) / float(len(drift_result.shared_columns))
        if drift_result.shared_columns
        else 0.0
    )
    drift_check_success_metric.set(1.0)
    drift_check_timestamp_metric.set(time.time())
    baseline_rows_metric.set(float(drift_result.baseline_row_count))
    current_rows_metric.set(float(drift_result.current_row_count))
    shared_columns_metric.set(float(len(drift_result.shared_columns)))

    for column_name in drift_result.shared_columns:
        column_metrics = drift_result.drift_by_column.get(column_name, {})
        feature_drift_metric.labels(feature=column_name).set(
            float(column_metrics.get("drift_detected", 0.0))
        )
        drift_score = column_metrics.get("drift_score")
        if drift_score is not None:
            feature_drift_score_metric.labels(feature=column_name).set(float(drift_score))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_to_textfile(str(output_path), registry)
    logger.info("Prometheus drift metrics written to %s", output_path)
    return output_path


def write_failed_prometheus_textfile(output_path: Path) -> Path:
    """Write a failed drift-check marker so Grafana reflects the error state."""
    registry = CollectorRegistry()
    drift_check_success_metric = Gauge(
        "gpon_drift_check_success",
        "Whether the last GPON drift detection run completed successfully.",
        registry=registry,
    )
    drift_check_timestamp_metric = Gauge(
        "gpon_drift_check_timestamp_seconds",
        "Unix timestamp of the last GPON drift detection run.",
        registry=registry,
    )

    drift_check_success_metric.set(0.0)
    drift_check_timestamp_metric.set(time.time())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_to_textfile(str(output_path), registry)
    logger.warning("Prometheus drift metrics marked as failed at %s", output_path)
    return output_path


def run_drift_detection(
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    current_path: Optional[Path] = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    prometheus_textfile_path: Optional[Path] = DEFAULT_PROMETHEUS_TEXTFILE_PATH,
    feature_only: bool = True,
    current_window_rows: Optional[int] = DEFAULT_CURRENT_WINDOW_ROWS,
    current_window_minutes: Optional[int] = None,
    min_current_rows: int = DEFAULT_MIN_CURRENT_ROWS,
) -> DriftResult:
    """Execute the full drift-detection pipeline."""
    if current_path is None:
        raise ValueError("current_path is required. Pass the production CSV path.")

    logger.info("=== Data Drift Detection Pipeline START ===")

    baseline_df = load_dataset(baseline_path, label="baseline")
    current_df = load_dataset(current_path, label="current")
    current_df = select_recent_rows(
        current_df,
        label="current",
        window_rows=current_window_rows,
        window_minutes=current_window_minutes,
    )

    if min_current_rows > 0 and len(current_df) < min_current_rows:
        raise ValueError(
            "Not enough current rows for a reliable drift decision: "
            f"{len(current_df)} < {min_current_rows}"
        )

    if feature_only:
        baseline_df = select_feature_columns(baseline_df)
        current_df = select_feature_columns(current_df)
    else:
        logger.info("Skipping feature selection; all columns included")

    baseline_row_count = len(baseline_df)
    current_row_count = len(current_df)

    report, shared_columns = build_drift_report(reference=baseline_df, current=current_df)
    saved_path = save_report(report, output_path)

    summary = extract_drift_summary(report)
    drifted_cols = parse_drifted_columns(summary)
    dataset_drift = is_dataset_drift_detected(summary)
    drift_by_column = extract_column_drift_metrics(summary)

    if dataset_drift:
        logger.warning(
            "Dataset-level drift detected in %d column(s): %s",
            len(drifted_cols),
            drifted_cols,
        )
    else:
        logger.info(
            "No dataset-level drift detected (%d drifted column(s): %s)",
            len(drifted_cols),
            drifted_cols,
        )

    drift_result = DriftResult(
        report=report,
        report_path=saved_path,
        drift_detected=dataset_drift,
        drifted_columns=drifted_cols,
        drift_summary=summary,
        shared_columns=shared_columns,
        drift_by_column=drift_by_column,
        baseline_row_count=baseline_row_count,
        current_row_count=current_row_count,
    )
    if prometheus_textfile_path is not None:
        write_prometheus_textfile(drift_result, prometheus_textfile_path)

    logger.info("=== Data Drift Detection Pipeline COMPLETE ===")
    return drift_result


def main() -> None:
    """CLI entry-point for drift detection."""
    parser = argparse.ArgumentParser(
        description="Detect data drift between baseline and recent production GPON telemetry.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="Path to the baseline (training) CSV.",
    )
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Path to the current (production) CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination path for the HTML drift report.",
    )
    parser.add_argument(
        "--include-target",
        action="store_true",
        default=False,
        help="Include target and metadata columns in drift analysis.",
    )
    parser.add_argument(
        "--current-window-rows",
        type=int,
        default=DEFAULT_CURRENT_WINDOW_ROWS,
        help="Use only the most recent N current rows for drift detection.",
    )
    parser.add_argument(
        "--current-window-minutes",
        type=int,
        default=None,
        help="Use only current rows from the last N minutes when timestamps are present.",
    )
    parser.add_argument(
        "--min-current-rows",
        type=int,
        default=DEFAULT_MIN_CURRENT_ROWS,
        help="Minimum number of current rows required before making a drift decision.",
    )
    parser.add_argument(
        "--prometheus-textfile",
        type=Path,
        default=DEFAULT_PROMETHEUS_TEXTFILE_PATH,
        help="Destination path for Prometheus textfile collector drift metrics.",
    )
    args = parser.parse_args()

    configure_logging()
    try:
        result = run_drift_detection(
            baseline_path=args.baseline,
            current_path=args.current,
            output_path=args.output,
            prometheus_textfile_path=args.prometheus_textfile,
            feature_only=not args.include_target,
            current_window_rows=args.current_window_rows,
            current_window_minutes=args.current_window_minutes,
            min_current_rows=args.min_current_rows,
        )
    except Exception:
        if args.prometheus_textfile is not None:
            write_failed_prometheus_textfile(args.prometheus_textfile)
        raise

    if result.drift_detected:
        logger.warning(
            "Exiting with code 1 because drift was detected in %d column(s)",
            len(result.drifted_columns),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
