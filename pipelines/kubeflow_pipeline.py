"""Kubeflow Pipelines v2 pipeline definition for GPON failure prediction."""

from pathlib import Path

from kfp import compiler, dsl
from kfp import kubernetes

from pipelines.pipeline_components import (
    evaluation_component,
    ingestion_component,
    registration_component,
    training_component,
)

_DEFAULT_INPUT_CSV = "s3://gpon-telemetry/raw/telemetry.csv"
_DEFAULT_TEST_SIZE = 0.2
_DEFAULT_AUC_THRESHOLD = 0.75
_DEFAULT_EXPERIMENT_NAME = "gpon-failure-prediction"
_DEFAULT_MODEL_NAME = "gpon-xgboost-classifier"


@dsl.pipeline(
    name="gpon-failure-prediction-pipeline",
    description=(
        "End-to-end ML pipeline for predicting GPON router failures within "
        "a 7-day horizon. Stages: ingestion → training → evaluation → MLflow registration."
    ),
)
def gpon_failure_prediction_pipeline(
    input_csv_path: str = _DEFAULT_INPUT_CSV,
    test_size: float = _DEFAULT_TEST_SIZE,
    auc_threshold: float = _DEFAULT_AUC_THRESHOLD,
    experiment_name: str = _DEFAULT_EXPERIMENT_NAME,
    model_name: str = _DEFAULT_MODEL_NAME,
) -> None:
    """Orchestrate ingestion, training, evaluation, and MLflow registration.

    Args:
        input_csv_path: S3/MinIO path to the raw telemetry CSV.
        test_size: Fraction of data reserved for testing (0-1).
        auc_threshold: Minimum AUC-ROC to pass the quality gate.
        mlflow_tracking_uri: MLflow server URI reachable from KIND pods.
        experiment_name: MLflow experiment name.
        model_name: MLflow Model Registry name.
    """
    def _inject_env(task):
        for env_var in ["MLFLOW_TRACKING_URI", "MLFLOW_S3_ENDPOINT_URL",
                        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]:
            kubernetes.use_config_map_as_env(
                task,
                config_map_name="mlops-endpoints",
                config_map_key_to_env={env_var: env_var},
            )
        kubernetes.set_image_pull_policy(task, "Never")
        task.set_caching_options(enable_caching=False)
        return task

    # Step 1 — Ingestion
    ingest_task = ingestion_component(input_csv_path=input_csv_path)
    ingest_task.set_display_name("Ingest & Validate Telemetry")
    _inject_env(ingest_task)

    # Step 2 — Training
    train_task = training_component(
        processed_dataset=ingest_task.outputs["processed_dataset"],
        test_size=test_size,
    )
    train_task.set_display_name("Train XGBoost Classifier")
    train_task.after(ingest_task)
    _inject_env(train_task)

    # Step 3 — Evaluation
    eval_task = evaluation_component(
        test_data_artifact=train_task.outputs["test_data_artifact"],
        auc_threshold=auc_threshold,
    )
    eval_task.set_display_name("Evaluate Model & Quality Gate")
    eval_task.after(train_task)
    _inject_env(eval_task)

    # Step 4 — MLflow Registration
    with dsl.If(eval_task.output == True, name="register-if-quality-gate-passed"):
        register_task = registration_component(
            model_artifact=train_task.outputs["model_artifact"],
            test_data_artifact=train_task.outputs["test_data_artifact"],
            experiment_name=experiment_name,
            model_name=model_name,
        )
        register_task.set_display_name("Register Model in MLflow")
        register_task.after(eval_task)
        _inject_env(register_task)


def compile_pipeline() -> Path:
    """Compile the pipeline to pipeline.yaml."""
    output_path = Path(__file__).resolve().parent / "pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=gpon_failure_prediction_pipeline,
        package_path=str(output_path),
    )
    print(f"Pipeline compiled successfully → {output_path}")
    return output_path


if __name__ == "__main__":
    compile_pipeline()
