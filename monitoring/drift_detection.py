"""Data drift detection for GPON router failure-prediction pipeline.

This module uses Evidently AI to compare a baseline (training) dataset
against a current (production) dataset and detect statistically
significant distribution shifts across all telemetry features.

The pipeline:
    1. Load the baseline training CSV produced by ``src.data.ingest``.
    2. Load the current production CSV collected from live ONT devices.
    3. Optionally restrict analysis to model feature columns only
       (excluding the target ``Failure_In_7_Days`` and ``timestamp``).
    4. Run the ``DataDriftPreset`` report from Evidently.
    5. Persist the HTML report to disk for review.

Usage::

    python -m monitoring.drift_detection \
        --baseline data/processed/processed.csv \
        --current  data/production/latest.csv \
        --output   monitoring/reports/drift_report.html

Requirements:
    evidently==0.4.16
    pandas==2.2.2
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

logger = logging.getLogger(__name__)

# Columns that are NOT model features and should be excluded from drift
# analysis.  Mirrors ``NON_FEATURE_COLUMNS`` in
# ``src.training.train_xgboost``.
NON_FEATURE_COLUMNS: List[str] = [
    "Failure_In_7_Days",
    "timestamp",
]

# Default paths consistent with the rest of the kubeflowingg project.
DEFAULT_BASELINE_PATH = Path("data/processed/processed.csv")
DEFAULT_OUTPUT_PATH = Path("monitoring/reports/drift_report.html")


# ------------------------------------------------------------------
# Data-class for structured results
# ------------------------------------------------------------------
@dataclass
class DriftResult:
    """Container for drift detection artefacts.

    Attributes:
        report: The Evidently ``Report`` object (already executed).
        report_path: Filesystem path where the HTML report was saved.
        drift_detected: ``True`` when the dataset-level drift test fires.
        drifted_columns: List of column names that individually drifted.
        drift_summary: Full dictionary representation of the report.
    """

    report: Report
    report_path: Path
    drift_detected: bool
    drifted_columns: List[str]
    drift_summary: Dict


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
def configure_logging(level: int = logging.INFO) -> None:
    """Set up structured logging for the drift-detection pipeline.

    Args:
        level: Python logging level.  Defaults to ``logging.INFO``.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def load_dataset(path: Path, label: str = "dataset") -> pd.DataFrame:
    """Load a CSV dataset from disk.

    Args:
        path: Filesystem path to the CSV file.
        label: Human-readable label used in log messages
            (e.g. ``"baseline"`` or ``"current"``).

    Returns:
        DataFrame loaded from the CSV.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")

    logger.info("Loading %s dataset from %s", label, path)
    df = pd.read_csv(path)
    logger.info(
        "%s dataset: %d rows × %d columns", label.capitalize(), len(df), len(df.columns)
    )
    return df


def select_feature_columns(
    df: pd.DataFrame,
    exclude: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Drop non-feature columns so drift analysis covers model inputs only.

    Args:
        df: Full DataFrame that may include target / metadata columns.
        exclude: Column names to drop.  Defaults to
            :data:`NON_FEATURE_COLUMNS`.

    Returns:
        DataFrame restricted to feature columns only.
    """
    if exclude is None:
        exclude = NON_FEATURE_COLUMNS

    cols_to_drop = [c for c in exclude if c in df.columns]
    if cols_to_drop:
        logger.info("Excluding non-feature columns: %s", cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    logger.info("Feature columns for drift analysis: %s", list(df.columns))
    return df


# ------------------------------------------------------------------
# Drift report
# ------------------------------------------------------------------
def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> Report:
    """Create and execute an Evidently DataDriftPreset report.

    The report compares the statistical distributions of every column
    in *reference* against *current* using the default per-column
    drift detection methods (Kolmogorov–Smirnov for numerical features,
    chi-squared / Jensen–Shannon for categorical ones).

    Args:
        reference: Baseline (training) DataFrame.
        current: Production DataFrame to test for drift.

    Returns:

    Raises:
        ValueError: If the two DataFrames have no overlapping columns.
    """
    common_cols = set(reference.columns) & set(current.columns)
    if not common_cols:
        raise ValueError(
            "No overlapping columns between reference and current datasets. "
            f"Reference: {list(reference.columns)}, "
            f"Current: {list(current.columns)}"
        )

    logger.info(
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    logger.info("Drift report executed successfully")
    return report


def save_report(report: Report, output_path: Path) -> Path:
    """Persist the Evidently report as a self-contained HTML file.

    Creates parent directories if they do not already exist.
        output_path: Destination file path for the HTML output.

    Returns:
        The resolved output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    logger.info("Drift report saved to %s", output_path)
    return output_path


def extract_drift_summary(report: Report) -> Dict:
    """Convert the Evidently report to a Python dictionary.

    Args:
        report: An executed Evidently ``Report``.
    """
    return report.as_dict()


def parse_drifted_columns(summary: Dict) -> List[str]:
    """Extract the list of columns that exhibited statistically significant drift.

    Walks the Evidently report dictionary to find per-column drift
    results from the ``DataDriftTable`` metric.

    Args:
        summary: Dictionary representation of the drift report
            (output of :func:`extract_drift_summary`).

    Returns:
        Sorted list of column names where drift was detected.
    """
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
    """Check whether dataset-level drift was flagged.

    Args:
        summary: Dictionary representation of the drift report.

    Returns:
        ``True`` if the overall dataset drift test fired.
    """
    for metric_result in summary.get("metrics", []):
        metric_data = metric_result.get("result", {})
        if "dataset_drift" in metric_data:
            return bool(metric_data["dataset_drift"])

# ------------------------------------------------------------------
# Pipeline orchestrator
# ------------------------------------------------------------------
def run_drift_detection(
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    current_path: Optional[Path] = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    feature_only: bool = True,
) -> DriftResult:
    """Execute the full drift-detection pipeline.

    Steps:
        1. Load baseline and current CSVs.
        2. Restrict to feature columns (optional).
        3. Build and run the Evidently ``DataDriftPreset`` report.
        4. Save the HTML report.
        5. Parse per-column and dataset-level drift results.

    Args:
        baseline_path: Path to the baseline (training) CSV.
        current_path: Path to the current (production) CSV.
            **Required** — ``None`` raises :class:`ValueError`.
        output_path: Destination for the HTML report.
        feature_only: If ``True``, exclude non-feature columns
            (target, timestamp) from the analysis.

    Returns:
        A :class:`DriftResult` dataclass with all artefacts.

    Raises:
        ValueError: If *current_path* is not supplied.
    """
    if current_path is None:
        raise ValueError(
            "current_path is required. Pass the path to the production CSV."
        )

    logger.info("=== Data Drift Detection Pipeline START ===")

    # 1 — Load data
    baseline_df = load_dataset(baseline_path, label="baseline")
    current_df = load_dataset(current_path, label="current")

    # 2 — Feature selection
    if feature_only:
        baseline_df = select_feature_columns(baseline_df)
        )
    else:
        logger.info(
            "✅ No dataset-level drift detected (%d column(s) drifted: %s)",
            len(drifted_cols),
            drifted_cols,
        )

    logger.info("=== Data Drift Detection Pipeline COMPLETE ===")

    return DriftResult(
        report=report,
        report_path=saved_path,
        drift_detected=dataset_drift,
        drifted_columns=drifted_cols,
        drift_summary=summary,
    )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main() -> None:
    """CLI entry-point for drift detection."""
    parser = argparse.ArgumentParser(
        description="Detect data drift between baseline and production GPON telemetry.",
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
    args = parser.parse_args()
    result = run_drift_detection(
        baseline_path=args.baseline,
        current_path=args.current,
        output_path=args.output,
    # Exit with non-zero code when drift is detected so CI/CD
    # pipelines can gate on distribution stability.
    if result.drift_detected:
        logger.warning(
            "Exiting with code 1 — drift detected in %d column(s)",
            len(result.drifted_columns),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()        feature_only=not args.include_target,
    )


    configure_logging()

    parser.add_argument(
        help="Include the target column in drift analysis.",
    )
        "--output",
        type=Path,
        default=False,
    parser.add_argument(
        "--include-target",
        action="store_true",
        default=DEFAULT_OUTPUT_PATH,
        help="Destination path for the HTML drift report.",
    )
        current_df = select_feature_columns(current_df)

    # 3 — Build & run
    report = build_drift_report(reference=baseline_df, current=current_df)
    drifted_cols = parse_drifted_columns(summary)
            drifted_cols,
    dataset_drift = is_dataset_drift_detected(summary)
            "⚠️  DATASET-LEVEL DRIFT DETECTED — %d column(s) drifted: %s",
            len(drifted_cols),

    if dataset_drift:
        logger.warning(

    # 5 — Parse summary
    summary = extract_drift_summary(report)

    # 4 — Persist
    saved_path = save_report(report, output_path)

    return False


    Returns:
        Nested dictionary mirroring the JSON structure of the report.

    Args:
        report: An executed Evidently ``Report``.
        "Building drift report — %d shared feature columns", len(common_cols)
    )


