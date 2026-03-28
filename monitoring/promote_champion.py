"""Promote the best MLflow model version and refresh the KServe predictor.

This script implements a practical local workflow:

1. Resolve the latest registered model version.
2. Compare its evaluation metric to the current ``champion`` alias.
3. If better, move the alias to the candidate version.
4. Export the champion model to a stable MinIO path used by KServe.
5. Restart the KServe predictor deployment so the sidecar downloads the new model.
"""

import argparse
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import mlflow
import mlflow.xgboost
from mlflow import MlflowClient

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_MODEL_NAME = "gpon-xgboost-classifier"
DEFAULT_ALIAS = "champion"
DEFAULT_METRIC_NAME = "test_auc_roc"
DEFAULT_DEPLOYMENT_MODEL_URI = (
    "s3://deployment-models/gpon-failure-predictor/champion/model.bst"
)
DEFAULT_PREDICTOR_DEPLOYMENT = "gpon-failure-predictor-predictor"
DEFAULT_KSERVE_NAMESPACE = "kserve"


def configure_logging(level: int = logging.INFO) -> None:
    """Set up structured logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Promote the best MLflow model version and refresh KServe.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        help="MLflow tracking server URI.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="MLflow registered model name.",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default=DEFAULT_ALIAS,
        help="Registered model alias used for the deployed champion.",
    )
    parser.add_argument(
        "--metric-name",
        type=str,
        default=DEFAULT_METRIC_NAME,
        help="Metric used to compare candidate and champion versions.",
    )
    parser.add_argument(
        "--candidate-version",
        type=str,
        default=None,
        help="Optional explicit candidate version. Defaults to the latest registered version.",
    )
    parser.add_argument(
        "--deployment-model-uri",
        type=str,
        default=DEFAULT_DEPLOYMENT_MODEL_URI,
        help="Stable S3/MinIO URI that KServe downloads on startup.",
    )
    parser.add_argument(
        "--minio-endpoint",
        type=str,
        default=os.environ.get(
            "MINIO_ENDPOINT_URL",
            os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"),
        ),
        help="MinIO endpoint URL used for stable deployment artifacts.",
    )
    parser.add_argument(
        "--predictor-deployment",
        type=str,
        default=DEFAULT_PREDICTOR_DEPLOYMENT,
        help="Kubernetes Deployment name backing the KServe predictor.",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=DEFAULT_KSERVE_NAMESPACE,
        help="Kubernetes namespace containing the KServe predictor Deployment.",
    )
    parser.add_argument(
        "--skip-rollout-restart",
        action="store_true",
        default=False,
        help="Skip restarting the KServe predictor after syncing the champion model.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force promotion even when the candidate does not beat the current champion.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Timeout used when waiting for model readiness and rollout status.",
    )
    return parser.parse_args()


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Split an S3 URI into bucket and key."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Expected an s3:// URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def get_s3_client(endpoint_url: str):
    """Create a boto3 client configured for MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minio"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123"),
        region_name="us-east-1",
        config=Config(signature_version="s3v4"),
    )


def ensure_bucket_exists(bucket: str, endpoint_url: str) -> None:
    """Create the deployment bucket if it does not already exist."""
    s3_client = get_s3_client(endpoint_url)
    try:
        s3_client.head_bucket(Bucket=bucket)
    except ClientError:
        logger.info("Creating missing bucket '%s'", bucket)
        s3_client.create_bucket(Bucket=bucket)


def wait_for_model_version_ready(
    client: MlflowClient,
    model_name: str,
    version: str,
    timeout_seconds: int,
) -> object:
    """Poll MLflow until the requested model version is ready."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        model_version = client.get_model_version(model_name, version)
        status = getattr(model_version, "status", "")
        if status == "READY":
            return model_version
        if status == "FAILED_REGISTRATION":
            raise RuntimeError(
                f"MLflow model version {model_name}/{version} failed registration"
            )
        logger.info(
            "Waiting for MLflow model version %s/%s to become READY (current status: %s)",
            model_name,
            version,
            status,
        )
        time.sleep(3)

    raise TimeoutError(
        f"Timed out waiting for MLflow model version {model_name}/{version} to become READY"
    )


def get_latest_model_version(client: MlflowClient, model_name: str) -> object:
    """Return the numerically latest registered model version."""
    versions = client.search_model_versions(f"name = '{model_name}'")
    if not versions:
        raise ValueError(f"No registered model versions found for '{model_name}'")
    return max(versions, key=lambda model_version: int(model_version.version))


def get_metric_value(
    client: MlflowClient,
    model_version: object,
    metric_name: str,
) -> float:
    """Read a comparison metric from the run that created a model version."""
    run = client.get_run(model_version.run_id)
    metrics = run.data.metrics
    if metric_name not in metrics:
        raise KeyError(
            f"Metric '{metric_name}' not found in run {model_version.run_id}. "
            f"Available metrics: {sorted(metrics.keys())}"
        )
    return float(metrics[metric_name])


def get_current_alias_version(
    client: MlflowClient,
    model_name: str,
    alias: str,
) -> Optional[object]:
    """Return the model version currently pointed to by an alias, if any."""
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except Exception:
        return None


def should_promote(
    candidate_metric: float,
    champion_metric: Optional[float],
    force: bool,
) -> bool:
    """Decide whether the candidate should replace the current champion."""
    if force or champion_metric is None:
        return True
    return candidate_metric > champion_metric


def upload_champion_model(
    model_name: str,
    model_version: str,
    deployment_model_uri: str,
    metric_name: str,
    metric_value: float,
    endpoint_url: str,
) -> None:
    """Export the champion model to the stable MinIO path used by KServe."""
    bucket, key = parse_s3_uri(deployment_model_uri)
    ensure_bucket_exists(bucket, endpoint_url)
    s3_client = get_s3_client(endpoint_url)

    model_uri = f"models:/{model_name}/{model_version}"
    logger.info("Loading model from MLflow URI %s", model_uri)
    model = mlflow.xgboost.load_model(model_uri)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        local_model_path = tmp_dir_path / "model.bst"
        model.save_model(str(local_model_path))
        s3_client.upload_file(str(local_model_path), bucket, key)

        metadata_path = tmp_dir_path / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "model_version": model_version,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        s3_client.upload_file(
            str(metadata_path),
            bucket,
            str(Path(key).with_name("metadata.json")).replace("\\", "/"),
        )

    logger.info(
        "Champion model synced to %s (version=%s, %s=%.4f)",
        deployment_model_uri,
        model_version,
        metric_name,
        metric_value,
    )


def rollout_restart_predictor(
    deployment_name: str,
    namespace: str,
    timeout_seconds: int,
) -> None:
    """Restart the KServe predictor Deployment so it downloads the new champion."""
    logger.info(
        "Restarting deployment/%s in namespace %s",
        deployment_name,
        namespace,
    )
    subprocess.run(
        [
            "kubectl",
            "rollout",
            "restart",
            f"deployment/{deployment_name}",
            "-n",
            namespace,
        ],
        check=True,
    )
    subprocess.run(
        [
            "kubectl",
            "rollout",
            "status",
            f"deployment/{deployment_name}",
            "-n",
            namespace,
            f"--timeout={timeout_seconds}s",
        ],
        check=True,
    )


def main() -> None:
    """CLI entry-point for champion promotion and deployment."""
    args = parse_args()
    configure_logging()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    latest_version = (
        client.get_model_version(args.model_name, args.candidate_version)
        if args.candidate_version
        else get_latest_model_version(client, args.model_name)
    )
    latest_version = wait_for_model_version_ready(
        client,
        args.model_name,
        latest_version.version,
        timeout_seconds=args.timeout_seconds,
    )

    candidate_metric = get_metric_value(client, latest_version, args.metric_name)
    champion_version = get_current_alias_version(client, args.model_name, args.alias)
    champion_metric = None

    if champion_version is not None:
        champion_metric = get_metric_value(client, champion_version, args.metric_name)
        logger.info(
            "Current %s: version=%s, %s=%.4f",
            args.alias,
            champion_version.version,
            args.metric_name,
            champion_metric,
        )
    else:
        logger.info("No existing '%s' alias found; candidate will be promoted", args.alias)

    logger.info(
        "Candidate version=%s, %s=%.4f",
        latest_version.version,
        args.metric_name,
        candidate_metric,
    )

    if not should_promote(candidate_metric, champion_metric, force=args.force):
        logger.info(
            "Keeping current champion; candidate did not beat the deployed metric"
        )
        return

    client.set_registered_model_alias(
        args.model_name,
        args.alias,
        latest_version.version,
    )
    logger.info(
        "Set MLflow alias '%s' -> version %s",
        args.alias,
        latest_version.version,
    )

    upload_champion_model(
        model_name=args.model_name,
        model_version=latest_version.version,
        deployment_model_uri=args.deployment_model_uri,
        metric_name=args.metric_name,
        metric_value=candidate_metric,
        endpoint_url=args.minio_endpoint,
    )

    if not args.skip_rollout_restart:
        rollout_restart_predictor(
            deployment_name=args.predictor_deployment,
            namespace=args.namespace,
            timeout_seconds=args.timeout_seconds,
        )


if __name__ == "__main__":
    main()
