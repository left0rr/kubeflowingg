"""MLflow model registration for GPON router failure prediction.

This module orchestrates the full MLflow experiment lifecycle:

1. Start an MLflow run with experiment tracking.
2. Enable ``mlflow.xgboost.autolog()`` for automatic parameter/metric
   capture.
3. Train the XGBoost model via ``src.training.train_xgboost``.
4. Evaluate using ``src.training.evaluate`` metric functions.
5. Log all metrics explicitly to MLflow.
6. Generate and log a feature-importance bar chart.
7. Register the trained model in the MLflow Model Registry.

Usage::

    python -m src.training.register_model \
        --input data/processed/processed.csv \
        --experiment-name gpon-failure-prediction \
        --model-name gpon-xgboost-classifier
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb

from src.training.evaluate import (
    calculate_auc,
    calculate_f1,
    calculate_precision,
    calculate_recall,
)
from src.training.train_xgboost import TrainingResult, run_training_pipeline

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "gpon-failure-prediction"
DEFAULT_MODEL_NAME = "gpon-xgboost-classifier"

def configure_logging(level: int = logging.INFO) -> None:
    """Set up structured logging for the registration pipeline.

    Args:
        level: Python logging level. Defaults to ``logging.INFO``.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def compute_metrics(result: TrainingResult) -> Dict[str, float]:
    """Compute all evaluation metrics from a training result.

    Args:
        result: A :class:`TrainingResult` containing ground-truth
            labels and model predictions.

    Returns:
        Dictionary mapping metric names to their float values.
    """
    metrics: Dict[str, float] = {
        "test_auc_roc": calculate_auc(result.y_test, result.y_pred_proba),
        "test_precision": calculate_precision(result.y_test, result.y_pred),
        "test_recall": calculate_recall(result.y_test, result.y_pred),
        "test_f1": calculate_f1(result.y_test, result.y_pred),
        "test_positive_rate": float(np.mean(result.y_pred)),
        "train_size": float(len(result.y_train)),
        "test_size": float(len(result.y_test)),
    }

    for name, value in metrics.items():
        logger.info("Metric %-20s : %.4f", name, value)

    return metrics

def log_metrics_to_mlflow(metrics: Dict[str, float]) -> None:
    """Log a dictionary of metrics to the active MLflow run.

    Args:
        metrics: Metric name-value pairs to log.
    """
    mlflow.log_metrics(metrics)
    logger.info("Logged %d metrics to MLflow", len(metrics))

def generate_feature_importance_plot(
    model: xgb.XGBClassifier,
    feature_names: List[str],
    top_n: int = 15,
) -> str:
    """Create a horizontal bar chart of XGBoost feature importances.

    The plot is saved to a temporary PNG file and its path returned
    for MLflow artifact logging.

    Args:
        model: Fitted XGBClassifier with ``feature_importances_``.
        feature_names: Ordered list of feature column names matching
            the training matrix.
        top_n: Number of top features to display. Defaults to ``15``.

    Returns:
        Filesystem path to the generated PNG image.
    """
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    top_idx = sorted_idx[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [feature_names[i] for i in top_idx],
        importances[top_idx],
        color="#2196F3",
        edgecolor="#1565C0",
    )
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("XGBoost Feature Importance — GPON Failure Prediction")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    tmp_path = tempfile.mktemp(suffix="_feature_importance.png")
    fig.savefig(tmp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Feature importance plot saved to %s", tmp_path)
    return tmp_path

def log_feature_importance_artifact(
    model: xgb.XGBClassifier,
    feature_names: List[str],
) -> None:
    """Generate and log the feature importance plot as an MLflow artifact.

    Args:
        model: Fitted XGBClassifier.
        feature_names: Ordered feature column names.
    """
    plot_path = generate_feature_importance_plot(model, feature_names)
    mlflow.log_artifact(plot_path, artifact_path="plots")
    logger.info("Feature importance plot logged to MLflow artifacts")

def register_model_in_registry(
    model: xgb.XGBClassifier,
    model_name: str,
) -> None:
    """Log the XGBoost model and register it in the MLflow Model Registry.

    Uses ``mlflow.xgboost.log_model`` with the ``registered_model_name``
    parameter to atomically log and register in a single call.

    Args:
        model: Fitted XGBClassifier to register.
        model_name: Name under which the model is registered in the
            MLflow Model Registry.
    """
    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="model",
        registered_model_name=model_name,
    )
    logger.info(
        "Model registered in MLflow Model Registry as '%s'", model_name
    )

def run_registration_pipeline(
    input_path: Path,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
    test_size: float = 0.2,
    tracking_uri: Optional[str] = None,
) -> None:
    """Execute the full MLflow registration pipeline.

    Steps:
        1. Configure MLflow tracking URI and experiment.
        2. Enable XGBoost autologging.
        3. Start an MLflow run.
        4. Train model via ``run_training_pipeline``.
        5. Compute and log evaluation metrics.
        6. Generate and log feature importance plot.
        7. Register model in the MLflow Model Registry.

    Args:
        input_path: Path to the processed telemetry CSV.
        experiment_name: MLflow experiment name.
        model_name: MLflow Model Registry name.
        test_size: Fraction reserved for testing.
        tracking_uri: Optional MLflow tracking server URI.
            Defaults to ``http://localhost:5000``.
    """
    logger.info("=== MLflow Registration Pipeline START ===")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI: %s", tracking_uri)
    else:
        mlflow.set_tracking_uri("http://localhost:5000")
        logger.info("MLflow tracking URI: http://localhost:5000 (default)")

    mlflow.set_experiment(experiment_name)
    logger.info("MLflow experiment: %s", experiment_name)

    mlflow.xgboost.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=False,
    )
    logger.info("XGBoost autologging enabled")

    with mlflow.start_run(run_name="xgboost-gpon-failure") as run:
        logger.info("MLflow run ID: %s", run.info.run_id)

        result = run_training_pipeline(
            input_path=input_path,
            test_size=test_size,
        )

        metrics = compute_metrics(result)
        log_metrics_to_mlflow(metrics)

        mlflow.log_param("target_column", "Failure_In_7_Days")
        mlflow.log_param("feature_count", len(result.feature_names))
        mlflow.log_param("feature_names", str(result.feature_names))

        log_feature_importance_artifact(
            model=result.model,
            feature_names=result.feature_names,
        )

        register_model_in_registry(
            model=result.model,
            model_name=model_name,
        )

        logger.info("=== MLflow Registration Pipeline COMPLETE ===")
        logger.info("Run ID   : %s", run.info.run_id)
        logger.info("Model    : %s", model_name)
        logger.info("AUC-ROC  : %.4f", metrics["test_auc_roc"])
        logger.info("F1 Score : %.4f", metrics["test_f1"]))

def main() -> None:
    """CLI entry-point for the MLflow registration pipeline."""
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and register XGBoost model in MLflow.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/processed.csv"),
        help="Path to the processed telemetry CSV.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="MLflow Model Registry name.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        help="MLflow tracking server URI.",
    configure_logging()
    run_registration_pipeline(
        input_path=args.input,
        tracking_uri=args.tracking_uri,
    )


if __name__ == "__main__":
    main()        experiment_name=args.experiment_name,
        model_name=args.model_name,
        test_size=args.test_size,
    )
    args = parser.parse_args()

        "--tracking-uri",
        type=str,
        default=None,
        default=0.2,
    )
    parser.add_argument(

