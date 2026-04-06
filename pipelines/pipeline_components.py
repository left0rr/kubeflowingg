"""Kubeflow Pipelines v2 component definitions for GPON failure prediction."""

from kfp import dsl
from kfp.dsl import Input, Metrics, Model, Output, Dataset


# ---------------------------------------------------------------------------
# 1. Ingestion Component
# ---------------------------------------------------------------------------

@dsl.component(base_image="kfp-base:latest", packages_to_install=[])
def ingestion_component(
    input_csv_path: str,
    processed_dataset: Output[Dataset],
) -> None:
    """Ingest and validate raw GPON telemetry data."""
    import logging
    import os
    import sys
    import tempfile
    from typing import List
    from urllib.parse import urlparse

    import boto3
    from botocore.config import Config
    import pandas as pd
    from pydantic import BaseModel, Field, ValidationError, field_validator

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("ingestion_component")

    class TelemetryRecord(BaseModel):
        Optical_RX_Power_dBm: float = Field(..., ge=-40.0, le=0.0)
        Optical_TX_Power_dBm: float = Field(..., ge=-10.0, le=10.0)
        Temperature_C: float = Field(..., ge=-40.0, le=125.0)
        Voltage_mV: float = Field(..., ge=0.0, le=5000.0)
        Bias_Current_mA: float = Field(..., ge=0.0, le=200.0)
        Interface_Error_Count: int = Field(..., ge=0)
        Reboot_Count_Last_7D: int = Field(..., ge=0)
        Connected_Devices: int = Field(..., ge=0)
        Device_Age_Days: int = Field(..., ge=0)
        Maintenance_Count_Last_30D: int = Field(..., ge=0)
        Failure_In_7_Days: int = Field(..., ge=0, le=1)

        @field_validator("Failure_In_7_Days")
        @classmethod
        def validate_binary_target(cls, v: int) -> int:
            if v not in (0, 1):
                raise ValueError("Failure_In_7_Days must be 0 or 1")
            return v

    def normalize_voltage(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Voltage_V"] = df["Voltage_mV"] / 1000.0
        return df.drop(columns=["Voltage_mV"])

    def compute_rx_power_rolling_features(
        df: pd.DataFrame, window: str = "24h", timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col).set_index(timestamp_col)
        rolling = df["Optical_RX_Power_dBm"].rolling(window=window, min_periods=1)
        df["RX_Power_Rolling_Mean_dBm"] = rolling.mean()
        df["RX_Power_Rolling_Std_dBm"] = rolling.std().fillna(0.0)
        return df.reset_index()

    logger.info("Loading raw CSV from %s", input_csv_path)
    parsed = urlparse(input_csv_path)

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
            tmp_path = tmp.name
        s3_client.download_file(bucket, key, tmp_path)
        logger.info("Downloaded s3://%s/%s to %s", bucket, key, tmp_path)
        raw_df = pd.read_csv(tmp_path)
        os.unlink(tmp_path)
    else:
        raw_df = pd.read_csv(input_csv_path)

    logger.info("Loaded %d rows, %d columns", len(raw_df), len(raw_df.columns))

    valid_rows: List[dict] = []
    invalid_indices: List[int] = []
    for idx, row in raw_df.iterrows():
        try:
            record = TelemetryRecord(**row.to_dict())
            valid_rows.append(record.model_dump())
        except ValidationError as exc:
            invalid_indices.append(int(idx))
            logger.warning("Row %d failed validation: %s errors", idx, exc.error_count())

    valid_df = pd.DataFrame(valid_rows)
    logger.info("Validation: %d valid, %d invalid out of %d total",
                len(valid_rows), len(invalid_indices), len(raw_df))

    if valid_df.empty:
        raise RuntimeError("No valid records after validation. Aborting.")

    valid_df = normalize_voltage(valid_df)
    logger.info("Voltage normalised (mV -> V)")

    if "timestamp" in valid_df.columns:
        valid_df = compute_rx_power_rolling_features(valid_df)
        logger.info("Rolling RX power features computed (24 h window)")

    valid_df.to_csv(processed_dataset.path, index=False)
    processed_dataset.metadata["row_count"] = len(valid_df)
    processed_dataset.metadata["column_count"] = len(valid_df.columns)
    processed_dataset.metadata["invalid_row_count"] = len(invalid_indices)
    logger.info("Processed dataset written to %s", processed_dataset.path)


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
    """Train an XGBoost classifier on processed GPON telemetry data."""
    import json
    import logging
    import sys
    from pathlib import Path
    from typing import Dict, List

    import numpy as np
    import pandas as pd
    import xgboost as xgb

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("training_component")

    TARGET_COLUMN = "Failure_In_7_Days"
    NON_FEATURE_COLUMNS = [TARGET_COLUMN, "timestamp"]
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

    logger.info("Loading processed dataset from %s", processed_dataset.path)
    df = pd.read_csv(processed_dataset.path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    split_idx = int(len(df) * (1.0 - test_size))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    logger.info("Split: %d train, %d test", len(train_df), len(test_df))

    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in train_df.columns]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=drop_cols)
    feature_names: List[str] = list(X_train.columns)
    logger.info("Features (%d): %s", len(feature_names), feature_names)

    model = xgb.XGBClassifier(**DEFAULT_XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    logger.info("Training complete")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    logger.info("Positive prediction rate: %.4f", float(np.mean(y_pred)))

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
    """Evaluate XGBoost predictions and gate on AUC-ROC threshold."""
    import logging
    import sys

    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("evaluation_component")
    TARGET_COLUMN = "Failure_In_7_Days"

    logger.info("Loading test data from %s", test_data_artifact.path)
    test_df = pd.read_csv(test_data_artifact.path)
    logger.info("Loaded %d test rows", len(test_df))

    y_true = test_df[TARGET_COLUMN].values
    y_pred = test_df["y_pred"].values
    y_pred_proba = test_df["y_pred_proba"].values

    auc_roc = float(roc_auc_score(y_true, y_pred_proba))
    precision = float(precision_score(y_true, y_pred, zero_division=0.0))
    recall = float(recall_score(y_true, y_pred, zero_division=0.0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0.0))
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

    if passed:
        logger.info("PASSED: AUC-ROC %.4f >= threshold %.4f", auc_roc, auc_threshold)
    else:
        logger.warning("FAILED: AUC-ROC %.4f < threshold %.4f", auc_roc, auc_threshold)
        raise RuntimeError(
            "Quality gate failed: "
            f"auc_roc={auc_roc:.4f} < threshold={auc_threshold:.4f}"
        )

    return True


# ---------------------------------------------------------------------------
# 4. MLflow Registration Component
# ---------------------------------------------------------------------------

@dsl.component(
    base_image="kfp-base:latest",
    packages_to_install=[],
)
def registration_component(
    model_artifact: Input[Model],
    test_data_artifact: Input[Dataset],
    experiment_name: str,
    model_name: str,
) -> None:
    """Log metrics, artifacts, and register the model in MLflow."""
    import json
    import logging
    import os
    import sys
    import tempfile

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow
    import mlflow.xgboost
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("registration_component")

    # Point MLflow artifact store at MinIO
    # All env vars injected from mlops-endpoints ConfigMap by kubeflow_pipeline.py

    mlflow_tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking URI : %s", mlflow_tracking_uri)
    logger.info("MLflow experiment   : %s", experiment_name)

    # Load model
    logger.info("Loading model from %s", model_artifact.path)
    model = xgb.XGBClassifier()
    model.load_model(model_artifact.path)
    feature_names = json.loads(model_artifact.metadata.get("feature_names", "[]"))
    logger.info("Model loaded. Features: %s", feature_names)

    # Load test data
    logger.info("Loading test data from %s", test_data_artifact.path)
    test_df = pd.read_csv(test_data_artifact.path)
    y_true = test_df["Failure_In_7_Days"].values
    y_pred = test_df["y_pred"].values
    y_pred_proba = test_df["y_pred_proba"].values

    # Compute metrics
    metrics = {
        "test_auc_roc":       float(roc_auc_score(y_true, y_pred_proba)),
        "test_precision":     float(precision_score(y_true, y_pred, zero_division=0.0)),
        "test_recall":        float(recall_score(y_true, y_pred, zero_division=0.0)),
        "test_f1":            float(f1_score(y_true, y_pred, zero_division=0.0)),
        "test_positive_rate": float(np.mean(y_pred)),
        "test_size":          float(len(test_df)),
    }
    for k, v in metrics.items():
        logger.info("%-25s: %.4f", k, v)

    # Feature importance plot
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color="#2196F3",
        edgecolor="#1565C0",
    )
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("XGBoost Feature Importance — GPON Failure Prediction")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plot_path = tempfile.mktemp(suffix="_feature_importance.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance plot saved to %s", plot_path)

    # Log to MLflow and register model
    with mlflow.start_run(run_name="kfp-xgboost-gpon-failure") as run:
        logger.info("MLflow run ID: %s", run.info.run_id)
        mlflow.log_metrics(metrics)
        mlflow.log_param("feature_count", len(feature_names))
        mlflow.log_param("feature_names", str(feature_names))
        mlflow.log_param("source", "kubeflow-pipeline")
        mlflow.log_artifact(plot_path, artifact_path="plots")
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        logger.info("Model registered in MLflow as '%s'", model_name)
        logger.info("AUC-ROC : %.4f", metrics["test_auc_roc"])
        logger.info("F1 Score: %.4f", metrics["test_f1"])
