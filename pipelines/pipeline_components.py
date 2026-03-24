"""Kubeflow Pipelines v2 component definitions for GPON failure prediction.

This module defines self-contained KFP v2 components that wrap the
project's ``src/`` modules.  Each component installs its own
dependencies at runtime via ``pip`` inside the base container, making
them portable across any Kubernetes cluster with KFP installed.

Components:
    * ``ingestion_component``   -- data loading, validation, feature engineering.
    * ``training_component``    -- chronological split and XGBoost training.
    * ``evaluation_component``  -- metric computation and threshold gating.

Usage::

    from pipelines.pipeline_components import (
        ingestion_component,
        evaluation_component,
        training_component,
    )
"""

from kfp import dsl
from kfp.dsl import Input, Metrics, Model, Output, Dataset
from uvicorn import Config


# ---------------------------------------------------------------------------
# 1. Ingestion Component
# ---------------------------------------------------------------------------

@dsl.component(
    base_image="kfp-base:latest",
    packages_to_install=[],  # already in image
)
def ingestion_component(
    input_csv_path: str,
    processed_dataset: Output[Dataset],
) -> None:
    """Ingest and validate raw GPON telemetry data.

    Loads raw CSV telemetry, validates each row against the Pydantic
    ``TelemetryRecord`` schema, applies feature engineering (voltage
    normalisation, rolling RX-power stats), and writes the processed
    DataFrame as a CSV artifact.

    Args:
        input_csv_path: GCS / S3 / local path to the raw telemetry CSV.
        processed_dataset: KFP output artifact for the processed CSV.
    """
    import logging
    import sys
    from pathlib import Path
    from typing import List, Tuple

    import pandas as pd
    from pydantic import BaseModel, Field, ValidationError, field_validator

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("ingestion_component")

    # ---- Inline validation schema ----------------------------------------

    class TelemetryRecord(BaseModel):
        """Pydantic schema mirroring src.data.validation.TelemetryRecord."""

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

    # ---- Inline feature engineering --------------------------------------

    def normalize_voltage(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Voltage_V"] = df["Voltage_mV"] / 1000.0
        df = df.drop(columns=["Voltage_mV"])
        return df

    def compute_rx_power_rolling_features(
        df: pd.DataFrame,
        window: str = "24h",
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col).set_index(timestamp_col)
        rolling = df["Optical_RX_Power_dBm"].rolling(window=window, min_periods=1)
        df["RX_Power_Rolling_Mean_dBm"] = rolling.mean()
        df["RX_Power_Rolling_Std_dBm"] = rolling.std().fillna(0.0)
        df = df.reset_index()
        return df

    # ---- Load CSV --------------------------------------------------------

    import os
    import tempfile
    import boto3
    from urllib.parse import urlparse
    from botocore.config import Config

    logger.info("Loading raw CSV from %s", input_csv_path)

    parsed = urlparse(input_csv_path)
    if parsed.scheme == "s3":
        s3_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://172.19.0.1:9000")
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
    else:
        raw_df = pd.read_csv(input_csv_path)

    logger.info("Loaded %d rows, %d columns", len(raw_df), len(raw_df.columns))

    # ---- Validate --------------------------------------------------------

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
    logger.info(
        "Validation: %d valid, %d invalid out of %d total",
        len(valid_rows), len(invalid_indices), len(raw_df),
    )

    if valid_df.empty:
        raise RuntimeError("No valid records after validation. Aborting.")

    # ---- Feature engineering ---------------------------------------------

    valid_df = normalize_voltage(valid_df)
    logger.info("Voltage normalised (mV -> V)")

    if "timestamp" in valid_df.columns:
        valid_df = compute_rx_power_rolling_features(valid_df)
        logger.info("Rolling RX power features computed (24 h window)")

    # ---- Persist artifact ------------------------------------------------

    valid_df.to_csv(processed_dataset.path, index=False)
    processed_dataset.metadata["row_count"] = len(valid_df)
    processed_dataset.metadata["column_count"] = len(valid_df.columns)
    processed_dataset.metadata["invalid_row_count"] = len(invalid_indices)
    logger.info("Processed dataset written to %s", processed_dataset.path)


# ---------------------------------------------------------------------------
# 2. Training Component
# ---------------------------------------------------------------------------

@dsl.component(
    base_image="kfp-base:latest",
    packages_to_install=[],  # already in image
)
def training_component(
    processed_dataset: Input[Dataset],
    test_size: float,
    model_artifact: Output[Model],
    test_data_artifact: Output[Dataset],
) -> None:
    """Train an XGBoost classifier on processed GPON telemetry data.

    Performs a chronological train/test split, trains an XGBoost binary
    classifier for ``Failure_In_7_Days``, serialises the model as a
    JSON artifact, and writes the test set with predictions for
downstream evaluation.

    Args:
        processed_dataset: KFP input artifact -- processed CSV from
            ``ingestion_component``.
        test_size: Fraction of data reserved for the test set (0-1).
        model_artifact: KFP output artifact for the serialised XGBoost model.
        test_data_artifact: KFP output artifact for the test CSV with
            ``y_pred`` and ``y_pred_proba`` columns appended.
    """
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

    # ---- Load processed dataset ------------------------------------------

    logger.info("Loading processed dataset from %s", processed_dataset.path)
    df = pd.read_csv(processed_dataset.path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # ---- Chronological split ---------------------------------------------

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info("Sorted by timestamp for chronological split")

    split_idx = int(len(df) * (1.0 - test_size))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    logger.info("Split: %d train, %d test", len(train_df), len(test_df))

    # ---- Prepare features ------------------------------------------------

    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in train_df.columns]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[TARGET_COLUMN]
    feature_names: List[str] = list(X_train.columns)
    logger.info("Features (%d): %s", len(feature_names), feature_names)

    # ---- Train -----------------------------------------------------------

    logger.info("Training XGBoost with params: %s", DEFAULT_XGB_PARAMS)
    model = xgb.XGBClassifier(**DEFAULT_XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    logger.info("Training complete")

    # ---- Predict ---------------------------------------------------------

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    logger.info("Positive prediction rate: %.4f", float(np.mean(y_pred)))

    # ---- Persist model artifact ------------------------------------------

    model_path = Path(model_artifact.path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    model_artifact.metadata["framework"] = "xgboost"
    model_artifact.metadata["n_estimators"] = DEFAULT_XGB_PARAMS["n_estimators"]
    model_artifact.metadata["max_depth"] = DEFAULT_XGB_PARAMS["max_depth"]
    model_artifact.metadata["feature_names"] = json.dumps(feature_names)
    logger.info("Model saved to %s", model_path)

    # ---- Persist test data with predictions ------------------------------

    test_output = test_df.copy()
    test_output["y_pred"] = y_pred
    test_output["y_pred_proba"] = y_pred_proba
    test_output.to_csv(test_data_artifact.path, index=False)
    test_data_artifact.metadata["test_row_count"] = len(test_output)
    logger.info("Test data with predictions written to %s", test_data_artifact.path)


# ---------------------------------------------------------------------------
# 3. Evaluation Component
# ---------------------------------------------------------------------------

@dsl.component(
    base_image="kfp-base:latest",
    packages_to_install=[],  # already in image
)
def evaluation_component(
    test_data_artifact: Input[Dataset],
    auc_threshold: float,
    metrics_artifact: Output[Metrics],
) -> bool:
    """Evaluate XGBoost predictions and gate on AUC-ROC threshold.

    Reads the test CSV (with ``y_pred`` and ``y_pred_proba`` columns
    produced by ``training_component``), computes AUC-ROC, precision,
    recall, and F1, logs them to the KFP Metrics artifact, and returns
    whether the model passes the AUC gate.

    Args:
        test_data_artifact: KFP input artifact -- test CSV with
            prediction columns from ``training_component``.
        auc_threshold: Minimum AUC-ROC required for the model to pass.
        metrics_artifact: KFP output artifact for logged metrics.

    Returns:
        ``True`` if AUC-ROC >= *auc_threshold*, ``False`` otherwise.
    """
    import logging
    import sys

    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("evaluation_component")

    TARGET_COLUMN = "Failure_In_7_Days"

    # ---- Load test data --------------------------------------------------

    logger.info("Loading test data from %s", test_data_artifact.path)
    test_df = pd.read_csv(test_data_artifact.path)
    logger.info("Loaded %d test rows", len(test_df))

    y_true = test_df[TARGET_COLUMN].values
    y_pred = test_df["y_pred"].values
    y_pred_proba = test_df["y_pred_proba"].values

    # ---- Compute metrics -------------------------------------------------

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

    # ---- Log to KFP Metrics artifact -------------------------------------

    metrics_artifact.log_metric("auc_roc", auc_roc)
    metrics_artifact.log_metric("precision", precision)
    metrics_artifact.log_metric("recall", recall)
    metrics_artifact.log_metric("f1_score", f1)
    metrics_artifact.log_metric("positive_rate", positive_rate)
    metrics_artifact.log_metric("test_row_count", float(len(test_df)))
    metrics_artifact.log_metric("auc_threshold", auc_threshold)

    # ---- Gate decision ---------------------------------------------------

    passed = auc_roc >= auc_threshold
    metrics_artifact.log_metric("passed_gate", float(passed))

    if passed:
        logger.info(
            "PASSED: AUC-ROC %.4f >= threshold %.4f", auc_roc, auc_threshold
        )
    else:
        logger.warning(
            "FAILED: AUC-ROC %.4f < threshold %.4f", auc_roc, auc_threshold
        )

    return passed
