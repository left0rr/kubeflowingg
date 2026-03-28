"""Traffic simulator for GPON router failure prediction.

The simulator can either:

1. Replay rows sampled from the processed training dataset so serving
   inputs stay close to the training distribution, or
2. Fall back to synthetic ranges when no baseline dataset is available.

Optional drift profiles can be applied gradually after a configurable
number of requests, which makes drift reports more realistic than
comparing two completely different hand-crafted distributions.
"""

import argparse
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import requests

KSERVE_URL = "http://localhost:8085/v1/models/gpon-failure-predictor:predict"
DEFAULT_BASELINE_PATH = Path("data/processed/processed.csv")
DEFAULT_OUTPUT_FILE = Path("data/predictions/latest.csv")
DEFAULT_SLEEP_SECONDS = 2.0

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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Send simulated GPON telemetry traffic to KServe and log predictions.",
    )
    parser.add_argument(
        "--kserve-url",
        type=str,
        default=KSERVE_URL,
        help="KServe prediction endpoint.",
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
        help="Prediction log CSV path.",
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

    reference_df = df[FEATURE_NAMES].dropna().reset_index(drop=True)
    if reference_df.empty:
        print("Baseline dataset has no usable rows after dropping NaNs; using synthetic mode")
        return None

    print(f"Loaded {len(reference_df)} baseline row(s) from {path}")
    return reference_df


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


def generate_synthetic_router_data() -> Tuple[Dict[str, float], str]:
    """Generate fallback synthetic telemetry close to the current workflow."""
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
        return clip_feature_row(feature_row), "Degraded"

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
    return clip_feature_row(feature_row), "Nominal"


def sample_reference_row(reference_df: pd.DataFrame) -> Dict[str, float]:
    """Sample a single row from the processed baseline dataset."""
    sampled_row = reference_df.sample(n=1).iloc[0].to_dict()
    return clip_feature_row(sampled_row)


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


def build_feature_row(
    mode: str,
    reference_df: Optional[pd.DataFrame],
    drift_profile: str,
    drift_start_after: int,
    drift_rate: float,
    drift_strength: float,
    request_index: int,
) -> Tuple[Dict[str, float], str, bool]:
    """Build the next feature row to send to KServe."""
    if mode == "baseline-replay" and reference_df is not None:
        feature_row = sample_reference_row(reference_df)
        status = "Nominal"
    else:
        feature_row, status = generate_synthetic_router_data()

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

    return feature_row, status, drift_applied


def log_prediction_row(output_file: Path, row: Dict[str, object]) -> None:
    """Append one prediction row to the CSV log."""
    pd.DataFrame([row], columns=CSV_COLUMNS).to_csv(
        output_file,
        mode="a",
        header=False,
        index=False,
    )


def main() -> None:
    """Run the traffic simulator until interrupted."""
    args = parse_args()
    reference_df = load_reference_dataset(args.baseline)
    mode = resolve_mode(args.mode, reference_df)

    print(f"Starting traffic simulation -> {args.kserve_url}")
    print(f"Traffic mode            -> {mode}")
    print(f"Drift profile           -> {args.drift_profile}")
    print("Press Ctrl+C to stop.\n")

    ensure_output_file(args.output)
    request_index = 0

    try:
        while True:
            feature_row, true_status, drift_applied = build_feature_row(
                mode=mode,
                reference_df=reference_df,
                drift_profile=args.drift_profile,
                drift_start_after=args.drift_start_after,
                drift_rate=args.drift_rate,
                drift_strength=args.drift_strength,
                request_index=request_index,
            )
            features = [feature_row[feature_name] for feature_name in FEATURE_NAMES]
            payload = {"instances": [features]}

            try:
                response = requests.post(args.kserve_url, json=payload, timeout=5)
                response.raise_for_status()

                score = float(response.json()["predictions"][0])
                predicted_failure_label = 1 if score > 0.5 else 0
                status_icon = "SHIFTED" if drift_applied else "STEADY"

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"{true_status:<8} | score={score:.4f} | {status_icon}"
                )

                row = dict(feature_row)
                row["timestamp"] = datetime.now().isoformat()
                row["prediction_score"] = score
                row["predicted_failure_label"] = predicted_failure_label
                row["Failure_In_7_Days"] = predicted_failure_label
                row["true_status"] = true_status
                row["source_mode"] = mode
                row["drift_profile"] = args.drift_profile
                row["drift_applied"] = drift_applied

                log_prediction_row(args.output, row)
                request_index += 1

            except requests.exceptions.ConnectionError:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    "Connection failed. Is the KServe port-forward running?\n"
                    "  kubectl port-forward -n kserve "
                    "svc/gpon-failure-predictor-predictor 8085:80"
                )
            except Exception as exc:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {exc}")

            time.sleep(args.sleep_seconds)

    except KeyboardInterrupt:
        print("\nSimulation stopped.")


if __name__ == "__main__":
    main()
