"""Kubeflow Pipelines component wrappers for GPON failure prediction.

Best-practice goal:

- keep the core ML logic in ``src/``
- keep KFP components thin and orchestration-focused
- let local CLI workflows and Kubeflow reuse the same source-of-truth code

These components now handle only KFP-specific concerns such as artifact paths,
S3 download/materialisation, and metadata logging.
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output


# ---------------------------------------------------------------------------
# 1. Ingestion Component
# ---------------------------------------------------------------------------

@dsl.component(base_image="kfp-base:latest", packages_to_install=[])
def ingestion_component(
    input_csv_path: str,
    processed_dataset: Output[Dataset],
) -> None:
    """Ingest raw telemetry by reusing the ``src.data.ingest`` workflow."""
    import logging
    import os
    from pathlib import Path
    import tempfile
    from typing import Optional
    from urllib.parse import urlparse

    import boto3
    from botocore.config import Config

    from src.data.ingest import (
        apply_feature_engineering,
        configure_logging,
        load_csv,
        save_processed,
        validate_records,
    )

    configure_logging()
    logger = logging.getLogger("ingestion_component")

    parsed = urlparse(input_csv_path)
    temp_input_path: Optional[Path] = None

    if parsed.scheme == "s3":
        s3_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
        s3_client = boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123"),
            region_name="us-east-1",
            config=Config(signature_version="s3v4"),
        )
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            temp_input_path = Path(tmp.name)
        s3_client.download_file(bucket, key, str(temp_input_path))
        logger.info("Downloaded s3://%s/%s to %s", bucket, key, temp_input_path)
        local_input_path = temp_input_path
    else:
        local_input_path = Path(input_csv_path)

    try:
        raw_df = load_csv(local_input_path)
        valid_df, invalid_indices = validate_records(raw_df)

        if valid_df.empty:
            raise RuntimeError("No valid records after validation. Aborting.")

        processed_df = apply_feature_engineering(valid_df)
        save_processed(processed_df, output_path=Path(processed_dataset.path))

        processed_dataset.metadata["row_count"] = len(processed_df)
        processed_dataset.metadata["column_count"] = len(processed_df.columns)
        processed_dataset.metadata["invalid_row_count"] = len(invalid_indices)
        logger.info("Processed dataset written to %s", processed_dataset.path)
    finally:
        if temp_input_path is not None and temp_input_path.exists():
            temp_input_path.unlink()


# ---------------------------------------------------------------------------
# 2. Training Component
# ---------------------------------------------------------------------------

@dsl.component(base_image="kfp-base:latest", packages_to_install=[])
def training_component(
    processed_dataset: Input[Dataset],
    test_size: float,
    model_artifact: Output[Model],
    test_data_artifact: Output[Dataset],
) -> None:
    """Train the model by reusing the ``src.training.train_xgboost`` workflow."""
    import json
    import logging
    from pathlib import Path

    from src.training.train_xgboost import (
        chronological_train_test_split,
        configure_logging,
        generate_predictions,
        load_processed_dataset,
        prepare_features_and_target,
        train_xgboost,
    )

    configure_logging()
    logger = logging.getLogger("training_component")

    df = load_processed_dataset(Path(processed_dataset.path))
    train_df, test_df = chronological_train_test_split(df, test_size=test_size)

    X_train, y_train, feature_names = prepare_features_and_target(train_df)
    X_test, _, _ = prepare_features_and_target(test_df)

    model = train_xgboost(X_train, y_train)
    y_pred, y_pred_proba = generate_predictions(model, X_test)

    model_path = Path(model_artifact.path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    model_artifact.metadata["framework"] = "xgboost"
    model_artifact.metadata["feature_names"] = json.dumps(feature_names)
    logger.info("Model saved to %s", model_path)

    test_output = test_df.copy()
    test_output["y_pred"] = y_pred
    test_output["y_pred_proba"] = y_pred_proba
    test_output.to_csv(test_data_artifact.path, index=False)
    test_data_artifact.metadata["test_row_count"] = len(test_output)
    logger.info("Test data with predictions written to %s", test_data_artifact.path)


# ---------------------------------------------------------------------------
# 3. Evaluation Component
# ---------------------------------------------------------------------------

@dsl.component(base_image="kfp-base:latest", packages_to_install=[])
def evaluation_component(
    test_data_artifact: Input[Dataset],
    auc_threshold: float,
    metrics_artifact: Output[Metrics],
) -> bool:
    """Evaluate predictions by reusing the shared metric functions in ``src``."""
    import logging

    import numpy as np
    import pandas as pd

    from src.training.evaluate import (
        calculate_auc,
        calculate_f1,
        calculate_precision,
        calculate_recall,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("evaluation_component")

    target_column = "Failure_In_7_Days"
    test_df = pd.read_csv(test_data_artifact.path)
    logger.info("Loaded %d test rows from %s", len(test_df), test_data_artifact.path)

    y_true = test_df[target_column].values
    y_pred = test_df["y_pred"].values
    y_pred_proba = test_df["y_pred_proba"].values

    auc_roc = calculate_auc(y_true, y_pred_proba)
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    f1 = calculate_f1(y_true, y_pred)
    positive_rate = float(np.mean(y_pred))

    logger.info("AUC-ROC   : %.4f", auc_roc)
    logger.info("Precision : %.4f", precision)
    logger.info("Recall    : %.4f", recall)
    logger.info("F1 Score  : %.4f", f1)
    logger.info("Pos. Rate : %.4f", positive_rate)

    metrics_artifact.log_metric("auc_roc", auc_roc)
    metrics_artifact.log_metric("precision", precision)
    metrics_artifact.log_metric("recall", recall)
    metrics_artifact.log_metric("f1_score", f1)
    metrics_artifact.log_metric("positive_rate", positive_rate)
    metrics_artifact.log_metric("test_row_count", float(len(test_df)))
    metrics_artifact.log_metric("auc_threshold", auc_threshold)

    passed = auc_roc >= auc_threshold
    metrics_artifact.log_metric("passed_gate", float(passed))

    if not passed:
        raise RuntimeError(
            "Quality gate failed: "
            f"auc_roc={auc_roc:.4f} < threshold={auc_threshold:.4f}"
        )

    logger.info("PASSED: AUC-ROC %.4f >= threshold %.4f", auc_roc, auc_threshold)
    return True


# ---------------------------------------------------------------------------
# 4. MLflow Registration Component
# ---------------------------------------------------------------------------

@dsl.component(base_image="kfp-base:latest", packages_to_install=[])
def registration_component(
    model_artifact: Input[Model],
    test_data_artifact: Input[Dataset],
    experiment_name: str,
    model_name: str,
) -> None:
    """Register the trained model by reusing the shared MLflow helper functions."""
    import json
    import logging
    import os

    import mlflow
    import pandas as pd
    import xgboost as xgb

    from src.training.evaluate import (
        calculate_auc,
        calculate_f1,
        calculate_precision,
        calculate_recall,
    )
    from src.training.register_model import (
        configure_logging,
        generate_feature_importance_plot,
        log_metrics_to_mlflow,
        register_model_in_registry,
    )

    configure_logging()
    logger = logging.getLogger("registration_component")

    mlflow_tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking URI : %s", mlflow_tracking_uri)
    logger.info("MLflow experiment   : %s", experiment_name)

    model = xgb.XGBClassifier()
    model.load_model(model_artifact.path)
    feature_names = json.loads(model_artifact.metadata.get("feature_names", "[]"))
    logger.info("Model loaded from %s", model_artifact.path)

    test_df = pd.read_csv(test_data_artifact.path)
    y_true = test_df["Failure_In_7_Days"].values
    y_pred = test_df["y_pred"].values
    y_pred_proba = test_df["y_pred_proba"].values

    metrics = {
        "test_auc_roc": calculate_auc(y_true, y_pred_proba),
        "test_precision": calculate_precision(y_true, y_pred),
        "test_recall": calculate_recall(y_true, y_pred),
        "test_f1": calculate_f1(y_true, y_pred),
        "test_positive_rate": float(y_pred.mean()),
        "test_size": float(len(test_df)),
    }
    for name, value in metrics.items():
        logger.info("Metric %-20s : %.4f", name, value)

    plot_path = generate_feature_importance_plot(model, feature_names)

    with mlflow.start_run(run_name="kfp-xgboost-gpon-failure") as run:
        logger.info("MLflow run ID: %s", run.info.run_id)
        log_metrics_to_mlflow(metrics)
        mlflow.log_param("feature_count", len(feature_names))
        mlflow.log_param("feature_names", str(feature_names))
        mlflow.log_param("source", "kubeflow-pipeline")
        mlflow.log_artifact(plot_path, artifact_path="plots")
        register_model_in_registry(model=model, model_name=model_name)

        logger.info("Model registered in MLflow as '%s'", model_name)
        logger.info("AUC-ROC : %.4f", metrics["test_auc_roc"])
        logger.info("F1 Score: %.4f", metrics["test_f1"])
