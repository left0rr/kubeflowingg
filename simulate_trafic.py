"""Traffic simulator for the GPON failure-prediction platform.

The simulator can now target either:

1. A FastAPI inference gateway that preserves router metadata for alerting, or
2. The raw KServe endpoint as a fallback.

When running through FastAPI, the simulator sends:

- router identity metadata
- Tunisian telecom numbers
- the validated model features
- optional simulated ground-truth labels for feedback logging

This keeps the model input pure while still making downstream alerting and
retraining workflows possible.
"""

import argparse
from datetime import datetime, timezone
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

FASTAPI_URL = "http://localhost:8010/predict"
KSERVE_URL = "http://localhost:8085/v1/models/gpon-failure-predictor:predict"
DEFAULT_BASELINE_PATH = Path("data/processed/processed.csv")
DEFAULT_OUTPUT_FILE = Path("data/predictions/latest.csv")
DEFAULT_SLEEP_SECONDS = 2.0
DEFAULT_GATEWAY_API_KEY = "gpon-dev-key"
DEFAULT_ROUTER_POOL_SIZE = 500

FEATURE_NAMES = [
    "Optical_RX_Power_dBm",
    "Optical_TX_Power_dBm",
    "Temperature_C",
    "Bias_Current_mA",
    "Interface_Error_Count",
    "Reboot_Count_Last_7D",
    "Connected_Devices",
    "Device_Age_Days",
    "Maintenance_Count_Last_30D",
    "Voltage_V",
]

FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Optical_RX_Power_dBm": (-40.0, 0.0),
    "Optical_TX_Power_dBm": (-10.0, 10.0),
    "Temperature_C": (-40.0, 125.0),
    "Bias_Current_mA": (0.0, 200.0),
    "Interface_Error_Count": (0.0, 5000.0),
    "Reboot_Count_Last_7D": (0.0, 50.0),
    "Connected_Devices": (0.0, 512.0),
    "Device_Age_Days": (0.0, 5000.0),
    "Maintenance_Count_Last_30D": (0.0, 50.0),
    "Voltage_V": (0.0, 5.0),
}

INTEGER_FEATURES = {
    "Interface_Error_Count",
    "Reboot_Count_Last_7D",
    "Connected_Devices",
    "Device_Age_Days",
    "Maintenance_Count_Last_30D",
}

CSV_COLUMNS = FEATURE_NAMES + [
    "timestamp",
    "prediction_score",
    "predicted_failure_label",
    "Failure_In_7_Days",
    "true_status",
    "source_mode",
    "drift_profile",
    "drift_applied",
]

TUNISIAN_PREFIXES = (
    "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
    "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
    "90", "91", "92", "93", "94", "95", "96", "97", "98", "99",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Send simulated GPON telemetry traffic to FastAPI or KServe.",
    )
    parser.add_argument(
        "--target",
        choices=("fastapi", "kserve"),
        default="fastapi",
        help="Where the simulator should send inference traffic.",
    )
    parser.add_argument(
        "--gateway-url",
        type=str,
        default=FASTAPI_URL,
        help="FastAPI gateway prediction endpoint.",
    )
    parser.add_argument(
        "--kserve-url",
        type=str,
        default=KSERVE_URL,
        help="Raw KServe prediction endpoint (used in --target kserve mode).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_GATEWAY_API_KEY,
        help="X-API-Key used when calling the FastAPI gateway.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="Processed baseline dataset used for replay-style traffic generation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Local prediction log CSV path used in --target kserve fallback mode.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Seconds between prediction requests.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "baseline-replay", "synthetic"),
        default="auto",
        help="Traffic source mode.",
    )
    parser.add_argument(
        "--drift-profile",
        choices=("none", "gradual-stress", "temperature-shift", "error-burst"),
        default="none",
        help="Controlled drift profile applied after --drift-start-after.",
    )
    parser.add_argument(
        "--drift-start-after",
        type=int,
        default=120,
        help="Number of successful requests to send before controlled drift starts.",
    )
    parser.add_argument(
        "--drift-rate",
        type=float,
        default=0.35,
        help="Probability of applying drift to a replayed row once drift starts.",
    )
    parser.add_argument(
        "--drift-strength",
        type=float,
        default=1.0,
        help="Multiplier controlling how strongly the drift profile perturbs a row.",
    )
    parser.add_argument(
        "--router-pool-size",
        type=int,
        default=DEFAULT_ROUTER_POOL_SIZE,
        help="Number of simulated Tunisian routers available to the traffic generator.",
    )
    return parser.parse_args()


def resolve_mode(mode: str, reference_df: Optional[pd.DataFrame]) -> str:
    """Resolve the runtime traffic mode."""
    if mode == "auto":
        return "baseline-replay" if reference_df is not None else "synthetic"
    return mode


def ensure_output_file(output_file: Path) -> None:
    """Create the prediction log with headers if it does not exist."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not output_file.exists():
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(output_file, index=False)
        print(f"Created {output_file} with columns: {CSV_COLUMNS}")


def load_reference_dataset(path: Path) -> Optional[pd.DataFrame]:
    """Load a processed baseline dataset that matches the model feature schema."""
    if not path.exists():
        print(f"Baseline dataset not found at {path}; falling back to synthetic mode")
        return None

    df = pd.read_csv(path)
    missing = [feature for feature in FEATURE_NAMES if feature not in df.columns]
    if missing:
        print(
            f"Baseline dataset is missing required feature columns {missing}; "
            "falling back to synthetic mode"
        )
        return None

    columns = FEATURE_NAMES.copy()
    if "Failure_In_7_Days" in df.columns:
        columns.append("Failure_In_7_Days")

    reference_df = df[columns].dropna(subset=FEATURE_NAMES).reset_index(drop=True)
    if reference_df.empty:
        print("Baseline dataset has no usable rows after dropping NaNs; using synthetic mode")
        return None

    print(f"Loaded {len(reference_df)} baseline row(s) from {path}")
    return reference_df


def build_router_profiles(pool_size: int) -> List[Dict[str, str]]:
    """Create a reusable pool of Tunisian router identities and telecom numbers."""
    if pool_size < 1:
        raise ValueError("router-pool-size must be at least 1")

    profiles: List[Dict[str, str]] = []
    for index in range(pool_size):
        prefix = TUNISIAN_PREFIXES[index % len(TUNISIAN_PREFIXES)]
        subscriber_digits = 100000 + (index % 900000)
        profiles.append(
            {
                "device_id": f"tn_router_{index + 1:05d}",
                "router_serial_number": f"TN-ONT-{index + 1:05d}-{1000 + (index % 9000)}",
                "telecom_number": f"+216{prefix}{subscriber_digits:06d}",
            }
        )
    return profiles


def choose_router_profile(router_profiles: List[Dict[str, str]]) -> Dict[str, str]:
    """Pick one router profile for the next simulated request."""
    return random.choice(router_profiles)


def clip_feature_row(feature_row: Dict[str, float]) -> Dict[str, float]:
    """Clamp features to realistic ranges and restore integer columns."""
    clipped_row: Dict[str, float] = {}
    for feature_name in FEATURE_NAMES:
        value = float(feature_row[feature_name])
        lower, upper = FEATURE_BOUNDS[feature_name]
        value = max(lower, min(upper, value))
        if feature_name in INTEGER_FEATURES:
            value = int(round(value))
        clipped_row[feature_name] = value
    return clipped_row


def generate_synthetic_router_data() -> Tuple[Dict[str, float], str, int]:
    """Generate fallback synthetic telemetry and a simple simulated label."""
    is_degraded = random.random() < 0.2

    if is_degraded:
        feature_row = {
            "Optical_RX_Power_dBm": random.uniform(-37.0, -30.0),
            "Optical_TX_Power_dBm": random.uniform(-6.0, -1.5),
            "Temperature_C": random.uniform(70.0, 92.0),
            "Bias_Current_mA": random.uniform(90.0, 155.0),
            "Interface_Error_Count": random.randint(25, 140),
            "Reboot_Count_Last_7D": random.randint(1, 4),
            "Connected_Devices": random.randint(2, 18),
            "Device_Age_Days": random.randint(900, 2200),
            "Maintenance_Count_Last_30D": random.randint(1, 4),
            "Voltage_V": random.uniform(2.95, 3.15),
        }
        return clip_feature_row(feature_row), "Degraded", 1

    feature_row = {
        "Optical_RX_Power_dBm": random.uniform(-24.0, -16.0),
        "Optical_TX_Power_dBm": random.uniform(1.2, 3.6),
        "Temperature_C": random.uniform(32.0, 48.0),
        "Bias_Current_mA": random.uniform(8.0, 28.0),
        "Interface_Error_Count": random.randint(0, 10),
        "Reboot_Count_Last_7D": random.randint(0, 1),
        "Connected_Devices": random.randint(4, 32),
        "Device_Age_Days": random.randint(120, 1600),
        "Maintenance_Count_Last_30D": random.randint(0, 2),
        "Voltage_V": random.uniform(3.18, 3.38),
    }
    return clip_feature_row(feature_row), "Nominal", 0


def sample_reference_row(reference_df: pd.DataFrame) -> Tuple[Dict[str, float], int]:
    """Sample one baseline-like row and preserve the optional simulated label."""
    sampled_row = reference_df.sample(n=1).iloc[0].to_dict()
    true_failure_in_7_days = int(sampled_row.get("Failure_In_7_Days", 0))
    feature_row = {feature_name: sampled_row[feature_name] for feature_name in FEATURE_NAMES}
    return clip_feature_row(feature_row), true_failure_in_7_days


def apply_controlled_drift(
    feature_row: Dict[str, float],
    profile: str,
    strength: float,
) -> Dict[str, float]:
    """Apply a realistic, bounded drift profile to a baseline-like row."""
    drifted_row = dict(feature_row)

    if profile == "temperature-shift":
        drifted_row["Temperature_C"] += random.uniform(2.0, 6.0) * strength
        drifted_row["Bias_Current_mA"] += random.uniform(4.0, 10.0) * strength
    elif profile == "error-burst":
        drifted_row["Interface_Error_Count"] += random.randint(12, 60)
        drifted_row["Reboot_Count_Last_7D"] += random.randint(1, 3)
    elif profile == "gradual-stress":
        drifted_row["Optical_RX_Power_dBm"] -= random.uniform(0.6, 2.0) * strength
        drifted_row["Temperature_C"] += random.uniform(2.0, 7.0) * strength
        drifted_row["Bias_Current_mA"] += random.uniform(5.0, 18.0) * strength
        drifted_row["Interface_Error_Count"] += random.randint(10, 70)
        drifted_row["Reboot_Count_Last_7D"] += random.randint(0, 2)
        drifted_row["Voltage_V"] -= random.uniform(0.05, 0.18) * strength

    return clip_feature_row(drifted_row)


def maybe_shift_true_label(
    true_failure_in_7_days: int,
    drift_profile: str,
    drift_applied: bool,
    drift_strength: float,
) -> int:
    """Adjust the simulated ground truth slightly when drift is applied."""
    if not drift_applied or true_failure_in_7_days == 1:
        return true_failure_in_7_days

    if drift_profile == "gradual-stress":
        promote_probability = min(0.70, 0.15 + (0.25 * drift_strength))
    elif drift_profile == "error-burst":
        promote_probability = min(0.60, 0.10 + (0.20 * drift_strength))
    elif drift_profile == "temperature-shift":
        promote_probability = min(0.40, 0.05 + (0.15 * drift_strength))
    else:
        promote_probability = 0.0

    return 1 if random.random() < promote_probability else 0


def build_feature_row(
    mode: str,
    reference_df: Optional[pd.DataFrame],
    drift_profile: str,
    drift_start_after: int,
    drift_rate: float,
    drift_strength: float,
    request_index: int,
) -> Tuple[Dict[str, float], str, bool, int]:
    """Build the next feature row and its simulated feedback label."""
    if mode == "baseline-replay" and reference_df is not None:
        feature_row, true_failure_in_7_days = sample_reference_row(reference_df)
        status = "Degraded" if true_failure_in_7_days == 1 else "Nominal"
    else:
        feature_row, status, true_failure_in_7_days = generate_synthetic_router_data()

    drift_applied = False
    if (
        drift_profile != "none"
        and request_index >= drift_start_after
        and random.random() < drift_rate
    ):
        feature_row = apply_controlled_drift(
            feature_row,
            profile=drift_profile,
            strength=drift_strength,
        )
        drift_applied = True
        status = "Shifted"

    true_failure_in_7_days = maybe_shift_true_label(
        true_failure_in_7_days=true_failure_in_7_days,
        drift_profile=drift_profile,
        drift_applied=drift_applied,
        drift_strength=drift_strength,
    )

    return feature_row, status, drift_applied, true_failure_in_7_days


def log_prediction_row(output_file: Path, row: Dict[str, object]) -> None:
    """Append one prediction row to the CSV log."""
    pd.DataFrame([row], columns=CSV_COLUMNS).to_csv(
        output_file,
        mode="a",
        header=False,
        index=False,
    )


def send_fastapi_request(
    gateway_url: str,
    api_key: str,
    router_profile: Dict[str, str],
    feature_row: Dict[str, float],
    request_timestamp: str,
    source_mode: str,
    drift_profile: str,
    drift_applied: bool,
    true_status: str,
    true_failure_in_7_days: int,
) -> Dict[str, object]:
    """Send an enriched request to the FastAPI inference gateway."""
    payload = {
        "device_id": router_profile["device_id"],
        "router_serial_number": router_profile["router_serial_number"],
        "telecom_number": router_profile["telecom_number"],
        "timestamp": request_timestamp,
        "features": feature_row,
        "source_mode": source_mode,
        "drift_profile": drift_profile,
        "drift_applied": drift_applied,
        "true_status": true_status,
        "true_failure_in_7_days": true_failure_in_7_days,
    }
    response = requests.post(
        gateway_url,
        json=payload,
        headers={"X-API-Key": api_key},
        timeout=8,
    )
    response.raise_for_status()
    return response.json()


def send_kserve_request(kserve_url: str, feature_row: Dict[str, float]) -> float:
    """Send a raw request directly to KServe and return the prediction score."""
    features = [feature_row[feature_name] for feature_name in FEATURE_NAMES]
    payload = {"instances": [features]}
    response = requests.post(kserve_url, json=payload, timeout=5)
    response.raise_for_status()
    return float(response.json()["predictions"][0])


def print_connection_hint(target: str) -> None:
    """Print a helpful operator hint when the target service is unavailable."""
    if target == "fastapi":
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] Connection failed. "
            "Is the FastAPI gateway running?\n"
            "  uvicorn src.api.inference_gateway:app --host 0.0.0.0 --port 8010"
        )
        return

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        "Connection failed. Is the KServe port-forward running?\n"
        "  kubectl port-forward -n kserve "
        "svc/gpon-failure-predictor-predictor 8085:80"
    )


def main() -> None:
    """Run the traffic simulator until interrupted."""
    args = parse_args()
    reference_df = load_reference_dataset(args.baseline)
    mode = resolve_mode(args.mode, reference_df)
    router_profiles = build_router_profiles(args.router_pool_size)

    print(f"Traffic target          -> {args.target}")
    if args.target == "fastapi":
        print(f"Gateway endpoint        -> {args.gateway_url}")
    else:
        print(f"KServe endpoint         -> {args.kserve_url}")
    print(f"Traffic mode            -> {mode}")
    print(f"Drift profile           -> {args.drift_profile}")
    print(f"Router pool size        -> {len(router_profiles)} (Tunisia +216 format)")
    print("Press Ctrl+C to stop.\n")

    if args.target == "kserve":
        ensure_output_file(args.output)

    request_index = 0

    try:
        while True:
            router_profile = choose_router_profile(router_profiles)
            feature_row, true_status, drift_applied, true_failure_in_7_days = build_feature_row(
                mode=mode,
                reference_df=reference_df,
                drift_profile=args.drift_profile,
                drift_start_after=args.drift_start_after,
                drift_rate=args.drift_rate,
                drift_strength=args.drift_strength,
                request_index=request_index,
            )
            request_timestamp = datetime.now(timezone.utc).isoformat()

            try:
                if args.target == "fastapi":
                    gateway_response = send_fastapi_request(
                        gateway_url=args.gateway_url,
                        api_key=args.api_key,
                        router_profile=router_profile,
                        feature_row=feature_row,
                        request_timestamp=request_timestamp,
                        source_mode=mode,
                        drift_profile=args.drift_profile,
                        drift_applied=drift_applied,
                        true_status=true_status,
                        true_failure_in_7_days=true_failure_in_7_days,
                    )
                    score = float(gateway_response["prediction_score"])
                    predicted_failure_label = int(gateway_response["predicted_failure_label"])
                    alert_candidate = bool(gateway_response.get("alert_candidate", False))
                    status_icon = "SHIFTED" if drift_applied else "STEADY"
                    alert_text = " ALERT" if alert_candidate else ""

                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"{router_profile['device_id']} | {router_profile['telecom_number']} | "
                        f"{true_status:<8} | score={score:.4f} | pred={predicted_failure_label} | "
                        f"{status_icon}{alert_text}"
                    )
                else:
                    score = send_kserve_request(args.kserve_url, feature_row)
                    predicted_failure_label = 1 if score >= 0.5 else 0
                    status_icon = "SHIFTED" if drift_applied else "STEADY"

                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"{true_status:<8} | score={score:.4f} | pred={predicted_failure_label} | "
                        f"{status_icon}"
                    )

                    row = dict(feature_row)
                    row["timestamp"] = request_timestamp
                    row["prediction_score"] = score
                    row["predicted_failure_label"] = predicted_failure_label
                    row["Failure_In_7_Days"] = ""
                    row["true_status"] = true_status
                    row["source_mode"] = mode
                    row["drift_profile"] = args.drift_profile
                    row["drift_applied"] = drift_applied
                    log_prediction_row(args.output, row)

                request_index += 1

            except requests.exceptions.ConnectionError:
                print_connection_hint(args.target)
            except Exception as exc:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {exc}")

            time.sleep(args.sleep_seconds)

    except KeyboardInterrupt:
        print("\nSimulation stopped.")


if __name__ == "__main__":
    main()
