"""XGBoost training pipeline for GPON router failure prediction.

This module handles the end-to-end model training workflow:

1. Load a processed telemetry CSV produced by ``src.data.ingest``.
2. Perform a chronological (time-respecting) train/test split to
   prevent data leakage from future observations.
3. Train an XGBoost binary classifier targeting ``Failure_In_7_Days``.
4. Return the fitted model alongside train/test predictions for
   downstream evaluation and MLflow logging.

Usage::

    python -m src.training.train_xgboost \
        --input data/processed/processed.csv \
        --test-size 0.2
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Failure_In_7_Days"

NON_FEATURE_COLUMNS: List[str] = [
    TARGET_COLUMN,
    "timestamp",
]

DEFAULT_XGB_PARAMS: Dict[str, object] = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1.0,
    "random_state": 42,
    "use_label_encoder": False,
}

@dataclass
class TrainingResult:
    """Container for training artefacts.

    Attributes:
        model: Fitted XGBClassifier instance.
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training target labels.
        y_test: Test target labels.
        y_pred: Binary predictions on the test set.
        y_pred_proba: Probability predictions on the test set.
        feature_names: Ordered list of feature column names.
    """

    model: xgb.XGBClassifier
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    y_pred_proba: np.ndarray
    feature_names: List[str] = field(default_factory=list)


def configure_logging(level: int = logging.INFO) -> None:
    """Set up structured logging for the training pipeline.

    Args:
        level: Python logging level. Defaults to ``logging.INFO``.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_processed_dataset(path: Path) -> pd.DataFrame:
    """Load a processed CSV dataset from disk.

    Args:
        path: Filesystem path to the processed CSV file.

    Returns:
        DataFrame ready for feature/target splitting.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If the target column is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")

    logger.info("Loading processed dataset from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows and %d columns", len(df), len(df.columns))

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def chronological_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data chronologically to prevent future-data leakage.

    If a *timestamp_col* is present the DataFrame is sorted by that
    column before splitting; otherwise the existing row order is
    assumed to be chronological.

    Args:
        df: Full processed DataFrame.
        test_size: Fraction of rows reserved for the test set.
            Defaults to ``0.2``.
        timestamp_col: Name of the datetime column used for ordering.
            Defaults to ``"timestamp"``.

    Returns:
        A ``(train_df, test_df)`` tuple.

    Raises:
        ValueError: If *test_size* is not in the open interval (0, 1).
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(
            f"test_size must be between 0 and 1 exclusive, got {test_size}"
        )

    df = df.copy()

    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        logger.info("Sorted dataset by '%s' for chronological split", timestamp_col)
    else:
        logger.info(
            "Column '%s' not found; using existing row order as chronological proxy",
            timestamp_col,
        )

    split_idx = int(len(df) * (1.0 - test_size))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    logger.info(
        "Train/test split: %d train rows, %d test rows (test_size=%.2f)",
        len(train_df),
        len(test_df),
        test_size,
    )
    return train_df, test_df


def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Separate feature matrix and target vector.

    Drops non-feature columns (target, timestamp, etc.) from the
    DataFrame.

    Args:
        df: DataFrame containing both features and the target.

    Returns:
        A tuple of ``(X, y, feature_names)``.
    """
    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COLUMN]
    feature_names = list(X.columns)

    logger.info(
        "Prepared %d features: %s",
        len(feature_names),
        feature_names,
    )
    return X, y, feature_names


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict[str, object]] = None,
) -> xgb.XGBClassifier:
    """Train an XGBoost binary classifier.

    Args:
        X_train: Training feature matrix.
        y_train: Training target labels.
        params: XGBoost hyperparameters.  Uses ``DEFAULT_XGB_PARAMS``
            when ``None``.

    Returns:
        Fitted ``XGBClassifier`` instance.
    """
    if params is None:
        params = DEFAULT_XGB_PARAMS.copy()

    logger.info("Training XGBoost with params: %s", params)

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    logger.info(
        "Training complete - best iteration: %s",
        model.best_iteration if hasattr(model, "best_iteration") else "N/A",
    )
    return model


def generate_predictions(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate binary and probability predictions.

    Args:
        model: Fitted XGBClassifier.
        X_test: Test feature matrix.

    Returns:
        A tuple of ``(y_pred, y_pred_proba)`` where *y_pred* contains
        binary labels and *y_pred_proba* contains positive-class
        probabilities.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    logger.info(
        "Predictions generated - positive rate: %.4f",
        float(np.mean(y_pred)),
    )
    return y_pred, y_pred_proba


def run_training_pipeline(
    input_path: Path,
    test_size: float = 0.2,
    params: Optional[Dict[str, object]] = None,
) -> TrainingResult:
    """Execute the full training pipeline.

    Steps:
        1. Load processed dataset.
        2. Chronological train/test split.
        3. Prepare features and target.
        4. Train XGBoost classifier.
        5. Generate predictions.

    Args:
        input_path: Path to the processed CSV.
        test_size: Fraction reserved for testing.
        params: Optional XGBoost hyperparameters.

    Returns:
        A :class:`TrainingResult` dataclass containing all artefacts.
    """
    logger.info("=== XGBoost Training Pipeline START ===")

    df = load_processed_dataset(input_path)
    train_df, test_df = chronological_train_test_split(df, test_size=test_size)

    X_train, y_train, feature_names = prepare_features_and_target(train_df)
    X_test, y_test, _ = prepare_features_and_target(test_df)

    model = train_xgboost(X_train, y_train, params=params)
    y_pred, y_pred_proba = generate_predictions(model, X_test)

    logger.info("=== XGBoost Training Pipeline COMPLETE ===")

    return TrainingResult(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        feature_names=feature_names,
    )


def main() -> None:
    """CLI entry-point for the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost classifier for GPON failure prediction.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/processed.csv"),
        help="Path to the processed telemetry CSV.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing (0-1).",
    )
    args = parser.parse_args()

    configure_logging()
    result = run_training_pipeline(
        input_path=args.input,
        test_size=args.test_size,
    )

    from src.training.evaluate import (
        calculate_auc,
        calculate_f1,
        calculate_precision,
        calculate_recall,
    )

    logger.info("--- Test Set Metrics ---")
    logger.info("AUC-ROC   : %.4f", calculate_auc(result.y_test, result.y_pred_proba))
    logger.info("Precision : %.4f", calculate_precision(result.y_test, result.y_pred))
    logger.info("Recall    : %.4f", calculate_recall(result.y_test, result.y_pred))
    logger.info("F1 Score  : %.4f", calculate_f1(result.y_test, result.y_pred))


if __name__ == "__main__":
    main()
