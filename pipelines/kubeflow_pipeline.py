"""Kubeflow Pipelines v2 pipeline definition for GPON failure prediction.

This module defines the end-to-end ML pipeline that orchestrates three
sequential stages:

    1. **Ingestion** -- load, validate, and feature-engineer raw telemetry.
    2. **Training** -- chronological split and XGBoost model fitting.
    3. **Evaluation** -- metric computation and AUC-ROC quality gate.

The pipeline is compiled to ``pipeline.yaml`` using the KFP v2 compiler,
which can then be uploaded to any Kubeflow Pipelines deployment.

Usage::

    # Compile only (produces pipeline.yaml alongside this file):
    python pipelines/kubeflow_pipeline.py

    # Then upload pipeline.yaml via the KFP UI or SDK:
    from kfp.client import Client
    client = Client(host="<KFP_HOST>")
    client.upload_pipeline("pipeline.yaml", pipeline_name="gpon-failure-prediction")
"""

from pathlib import Path

from kfp import compiler, dsl

from pipelines.pipeline_components import (
    evaluation_component,
    ingestion_component,
    training_component,
)

# ---------------------------------------------------------------------------
# Pipeline defaults
# ---------------------------------------------------------------------------

_DEFAULT_INPUT_CSV = "gs://gpon-telemetry/raw/telemetry.csv"
_DEFAULT_TEST_SIZE = 0.2
_DEFAULT_AUC_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------

@dsl.pipeline(
    name="gpon-failure-prediction-pipeline",
    description=(
        "End-to-end ML pipeline for predicting GPON router failures within "
        "a 7-day horizon.  Stages: ingestion → training → evaluation."
    ),
)
def gpon_failure_prediction_pipeline(
    input_csv_path: str = _DEFAULT_INPUT_CSV,
    test_size: float = _DEFAULT_TEST_SIZE,
    auc_threshold: float = _DEFAULT_AUC_THRESHOLD,
) -> None:
    """Orchestrate ingestion, training, and evaluation.

    Args:
        input_csv_path: Cloud Storage or local path to the raw telemetry
            CSV file.  Defaults to ``gs://gpon-telemetry/raw/telemetry.csv``.
        test_size: Fraction of data reserved for the chronological test
            split.  Must be in the open interval (0, 1).  Defaults to ``0.2``.
        auc_threshold: Minimum AUC-ROC score the trained model must
            achieve for the evaluation gate to pass.  Defaults to ``0.75``.
    """

    # Step 1 – Ingestion: load CSV → validate → feature engineer
    ingest_task = ingestion_component(
        input_csv_path=input_csv_path,
    )
    ingest_task.set_display_name("Ingest & Validate Telemetry")
    ingest_task.set_caching_options(enable_caching=True)

    # Step 2 – Training: chronological split → XGBoost fit → predictions
    train_task = training_component(
        processed_dataset=ingest_task.outputs["processed_dataset"],
        test_size=test_size,
    )
    train_task.set_display_name("Train XGBoost Classifier")
    train_task.set_caching_options(enable_caching=True)
    train_task.after(ingest_task)

    # Step 3 – Evaluation: metrics → AUC gate
    eval_task = evaluation_component(
        test_data_artifact=train_task.outputs["test_data_artifact"],
        auc_threshold=auc_threshold,
    )
    eval_task.set_display_name("Evaluate Model & Quality Gate")
    eval_task.set_caching_options(enable_caching=False)
    eval_task.after(train_task)


# ---------------------------------------------------------------------------
# Compilation entry-point
# ---------------------------------------------------------------------------

def compile_pipeline() -> Path:
    """Compile the pipeline to an IR YAML file.

    The compiled artifact is written to ``pipeline.yaml`` in the same
    directory as this module.

    Returns:
        Resolved path to the compiled ``pipeline.yaml``.
    """
    output_path = Path(__file__).resolve().parent / "pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=gpon_failure_prediction_pipeline,
        package_path=str(output_path),
    )
    print(f"Pipeline compiled successfully → {output_path}")
    return output_path


if __name__ == "__main__":
    compile_pipeline()
