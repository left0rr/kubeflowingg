"""
Traffic simulator for GPON router failure prediction.

Sends synthetic telemetry requests to the KServe XGBoost model,
collects predictions, and saves both the raw feature vectors and
prediction results to CSV for drift detection and monitoring.

Output CSV columns match processed.csv feature schema so that
Evidently AI can compare baseline vs production distributions.
"""

import time
import random
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# --- Configuration ---
KSERVE_URL = "http://localhost:8085/v1/models/gpon-failure-predictor:predict"
OUTPUT_DIR = Path("data/predictions")
OUTPUT_FILE = OUTPUT_DIR / "latest.csv"
SLEEP_TIME = 2  # seconds between requests

# Feature names in exact order the model expects
# Must match training feature order from pipeline logs:
# ['Optical_RX_Power_dBm', 'Optical_TX_Power_dBm', 'Temperature_C',
#  'Bias_Current_mA', 'Interface_Error_Count', 'Reboot_Count_Last_7D',
#  'Connected_Devices', 'Device_Age_Days', 'Maintenance_Count_Last_30D',
#  'Voltage_V']
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

# CSV columns: all features + prediction metadata
CSV_COLUMNS = FEATURE_NAMES + [
    "timestamp",
    "prediction_score",
    "Failure_In_7_Days",
    "true_status",
]


def generate_router_data():
    """Generate synthetic GPON telemetry matching the training feature schema.

    Returns:
        Tuple of (features list, router_status string).
        Features are in FEATURE_NAMES order.
    """
    is_failing = random.random() < 0.2

    if is_failing:
        features = [
            random.uniform(-40.0, -35.0),   # Optical_RX_Power_dBm
            random.uniform(-10.0, -8.0),    # Optical_TX_Power_dBm
            random.uniform(85.0, 100.0),    # Temperature_C
            random.uniform(150.0, 200.0),   # Bias_Current_mA  (fixed: was 300-500, out of training range 0-200)
            random.randint(50, 200),        # Interface_Error_Count
            random.randint(3, 5),           # Reboot_Count_Last_7D
            random.randint(1, 10),          # Connected_Devices
            random.randint(1500, 2000),     # Device_Age_Days
            random.randint(3, 5),           # Maintenance_Count_Last_30D
            random.uniform(2.8, 3.0),       # Voltage_V (degraded voltage)
        ]
        router_status = "Degraded"
    else:
        features = [
            random.uniform(-25.0, -15.0),   # Optical_RX_Power_dBm
            random.uniform(1.0, 4.0),       # Optical_TX_Power_dBm
            random.uniform(30.0, 50.0),     # Temperature_C
            random.uniform(10.0, 50.0),     # Bias_Current_mA
            random.randint(0, 5),           # Interface_Error_Count
            random.randint(0, 1),           # Reboot_Count_Last_7D
            random.randint(5, 50),          # Connected_Devices
            random.randint(100, 1000),      # Device_Age_Days
            random.randint(0, 1),           # Maintenance_Count_Last_30D
            random.uniform(3.2, 3.4),       # Voltage_V (normal ~3.3V)
        ]
        router_status = "Healthy"

    return features, router_status


def main():
    print(f"Starting traffic simulation → {KSERVE_URL}")
    print("Press Ctrl+C to stop.\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialise CSV with headers if it does not exist
    if not OUTPUT_FILE.exists():
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(OUTPUT_FILE, index=False)
        print(f"Created {OUTPUT_FILE} with columns: {CSV_COLUMNS}")

    try:
        while True:
            features, true_status = generate_router_data()
            payload = {"instances": [features]}

            try:
                response = requests.post(KSERVE_URL, json=payload, timeout=5)
                response.raise_for_status()

                score = response.json()["predictions"][0]
                is_predicted_failure = 1 if score > 0.5 else 0

                status_icon = "ALERT" if is_predicted_failure else "OK"
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"{true_status:<10} | score={score:.4f} | {status_icon}"
                )

                # Build row with ALL feature values + prediction metadata
                # This is what Evidently needs for drift detection
                row = dict(zip(FEATURE_NAMES, features))
                row["timestamp"] = datetime.now().isoformat()
                row["prediction_score"] = score
                row["Failure_In_7_Days"] = is_predicted_failure
                row["true_status"] = true_status

                pd.DataFrame([row]).to_csv(
                    OUTPUT_FILE, mode="a", header=False, index=False
                )

            except requests.exceptions.ConnectionError:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    "Connection failed — is the KServe port-forward running?\n"
                    "  kubectl port-forward -n kserve "
                    "svc/gpon-failure-predictor-predictor 8085:80 &"
                )
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")

            time.sleep(SLEEP_TIME)

    except KeyboardInterrupt:
        print("\nSimulation stopped.")


if __name__ == "__main__":
    main()
