"""Data ingestion pipeline for GPON router telemetry.

This module orchestrates the end-to-end data ingestion workflow:

1. Load raw CSV telemetry data from disk.
2. Validate every row against the ``TelemetryRecord`` Pydantic schema,
   discarding invalid records with structured logging.
3. Apply feature engineering transformations (voltage normalisation,
   rolling RX-power statistics).
4. Persist the processed DataFrame to ``data/processed/processed.csv``.

Usage::

    python -m src.data.ingest --input data/raw/telemetry.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from pydantic import ValidationError

from src.data.feature_engineering import (
    compute_rx_power_rolling_features,
    normalize_voltage,
)
from src.data.validation import TelemetryRecord

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = Path("data/processed/processed.csv")

def configure_logging(level: int = logging.INFO) -> None:
    """Set up structured logging for the ingestion pipeline.

    Args:
        level: Python logging level. Defaults to ``logging.INFO``.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def load_csv(path: Path) -> pd.DataFrame:
    """Load a raw CSV file into a pandas DataFrame.

    Args:
        path: Filesystem path to the CSV file.

    Returns:
        Raw DataFrame with all columns as-read from the CSV.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    logger.info("Loading raw CSV from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows and %d columns", len(df), len(df.columns))
    return df

def validate_records(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[int]]:
    """Validate each row against the TelemetryRecord Pydantic schema.

    Rows that fail validation are dropped and their indices logged.

    Args:
        df: Raw DataFrame whose columns must match
            :class:`~src.data.validation.TelemetryRecord` fields.

    Returns:
        A tuple of:
        - DataFrame containing only valid rows.
        - List of original row indices that failed validation.
    """
    valid_rows: List[dict] = []
    invalid_indices: List[int] = []

    for idx, row in df.iterrows():
        try:
            record = TelemetryRecord(**row.to_dict())
            valid_rows.append(record.model_dump())
        except ValidationError as exc:
            invalid_indices.append(int(idx))
            logger.warning(
                "Row %d failed validation: %s",
                idx,
                exc.error_count(),
            )

    valid_df = pd.DataFrame(valid_rows)
    logger.info(
        "Validation complete: %d valid, %d invalid out of %d total rows",
        len(valid_rows),
        len(invalid_indices),
        len(df),
    )
    return valid_df, invalid_indices

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations.

    Current transformations:
        1. ``normalize_voltage`` -- convert ``Voltage_mV`` to ``Voltage_V``.
        2. ``compute_rx_power_rolling_features`` -- 24 h rolling mean and
           standard deviation for ``Optical_RX_Power_dBm`` (requires a
           ``timestamp`` column; skipped gracefully if absent).

    Args:
        df: Validated DataFrame.

    Returns:
        Transformed DataFrame ready for model training or serving.
    """
    logger.info("Applying feature engineering transformations")

    df = normalize_voltage(df)
    logger.info("Voltage normalised (mV -> V)")

    if "timestamp" in df.columns:
        df = compute_rx_power_rolling_features(df)
        logger.info("Rolling RX power features computed (24 h window)")
    else:
        logger.info(
            "Column 'timestamp' not found; skipping rolling features"
        )

    return df

def save_processed(
    df: pd.DataFrame,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Save the processed DataFrame to CSV.

    Creates parent directories if they do not exist.

    Args:
        df: Processed DataFrame to persist.
        output_path: Destination file path.
            Defaults to ``data/processed/processed.csv``.

    Returns:
        The resolved output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(
        "Saved %d processed rows to %s", len(df), output_path
    )
    return output_path

def run_pipeline(
    input_path: Path,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Execute the full ingestion pipeline.

    Steps:
        1. Load raw CSV.
        2. Validate rows via Pydantic.
        3. Apply feature engineering.
        4. Save processed output.

    Args:
        input_path: Path to the raw telemetry CSV.
        output_path: Destination for the processed CSV.

    Returns:
        The final processed DataFrame.
    """
    logger.info("=== GPON Telemetry Ingestion Pipeline START ===")

    raw_df = load_csv(input_path)
    valid_df, _ = validate_records(raw_df)

    if valid_df.empty:
        logger.error("No valid records after validation. Aborting.")
        sys.exit(1)

    processed_df = apply_feature_engineering(valid_df)
    save_processed(processed_df, output_path)

    logger.info("=== GPON Telemetry Ingestion Pipeline COMPLETE ===")
    return processed_df

def main() -> None:
    """CLI entry-point for the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest and process GPON telemetry data.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw telemetry CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the processed output CSV.",
    )
    args = parser.parse_args()

    configure_logging()
    run_pipeline(input_path=args.input, output_path=args.output)

if __name__ == "__main__":
    main()
