import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.special import expit
import gc

np.random.seed(42)

NUM_DEVICES = 500
DAYS = 30
INTERVAL_HOURS = 3 # Changed to 3-hour intervals

devices = []
for i in range(NUM_DEVICES):
    vendor = np.random.choice(["ALCL", "HWTC"])
    will_fail = np.random.choice([True, False], p=[0.15, 0.85])
    failure_day = np.random.randint(12, 29) if will_fail else 999

    devices.append({
        "device_id": f"{vendor}_{np.random.randint(10000, 99999)}_{i}",
        "vendor": vendor,
        "will_fail": will_fail,
        "failure_day": failure_day,
        "base_rx": np.random.uniform(-17.0, -21.0),
        "base_tx": np.random.uniform(2.0, 3.2),
        "base_temp": np.random.uniform(35.0, 42.0),
        "base_bias": np.random.uniform(7.0, 13.0),
        "age_days": np.random.randint(100, 2000),
        "maintenance_count": np.random.choice([0, 1, 2, 3], p=[0.75, 0.15, 0.08, 0.02]),
        "hidden_wear": 0.0
    })

data = []
start_date = datetime(2025, 1, 1)

print("Simulating TR-069 aligned telecom physics (Hardened - 3H Interval)...")

for dev in devices:
    current_wear = dev["hidden_wear"]
    reboots_queue = [0] * 7 # Rolling 7-day reboot history

    for day in range(DAYS):
        daily_reboots = 0

        # Step every 3 hours: 0, 3, 6, 9, 12, 15, 18, 21
        for hour in range(0, 24, INTERVAL_HOURS):
            current_time = start_date + timedelta(days=day, hours=hour)

            # 1. Base Metrics & Seasonality (Wi-Fi Load)
            is_peak = 18 <= hour <= 23
            connected_devices = max(0, int(np.random.normal(14 if is_peak else 4, 3)))

            rx_power = dev["base_rx"] + np.random.normal(0, 0.25)
            tx_power = dev["base_tx"] + np.random.normal(0, 0.05)
            # Temp scales naturally with connected devices
            temp = dev["base_temp"] + (connected_devices * 0.2) + np.random.normal(0, 0.5)
            bias = dev["base_bias"] + np.random.normal(0, 0.1)

            # Voltage normalized to mV
            voltage = 3300 + np.random.normal(0, 25 if dev["vendor"] == "HWTC" else 5)

            interfaces_errors = np.random.poisson(1.0 if is_peak else 0.2)

            # 2. Concept Drift (Day 20 Firmware Update for HWTC)
            if day >= 20 and dev["vendor"] == "HWTC":
                temp += 4.5
                bias += 2.0
                interfaces_errors += np.random.poisson(5.0)

            # 3. Transient Spurious Noise
            if np.random.random() < 0.015:
                interfaces_errors += np.random.randint(50, 300)
                rx_power -= np.random.uniform(1.0, 3.0)
                temp += np.random.uniform(2.0, 5.0)

            # 4. Stochastic Degradation
            days_to_failure = dev["failure_day"] - day
            is_failed_state = days_to_failure <= 0

            if dev["will_fail"] and not is_failed_state:
                if days_to_failure <= 12:
                    current_wear += max(-0.2, np.random.normal(0.5, 0.8))

                wear_factor = expit(current_wear * 0.4 - 2)

                rx_power -= (wear_factor * 8.5) + np.random.normal(0, wear_factor * 1.5)
                tx_power -= (wear_factor * 1.8)
                bias += (wear_factor * 18.0)
                temp += (wear_factor * 12.0) + np.random.normal(0, wear_factor * 2)
                interfaces_errors += int(wear_factor * np.random.randint(20, 150))

                if wear_factor > 0.4 and np.random.random() < (wear_factor * 0.15):
                    daily_reboots += 1
                    rx_power = -40.0
                    interfaces_errors += np.random.randint(100, 500)

                if days_to_failure > 4 and np.random.random() < 0.04:
                    current_wear *= 0.35

            elif is_failed_state:
                rx_power = -40.0
                tx_power = 0.0
                temp = 20.0
                bias = 0.0
                connected_devices = 0
                interfaces_errors = 0

            # Update reboot queue at the last interval of the day (Hour 21)
            if hour == 24 - INTERVAL_HOURS:
                reboots_queue.pop(0)
                reboots_queue.append(daily_reboots)

            # 5. Target Label
            target_label = 1 if (dev["will_fail"] and 0 < days_to_failure <= 7) else 0

            data.append({
                "Timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "Device_ID": dev["device_id"],
                "Vendor": dev["vendor"],
                "Device_Age_Days": dev["age_days"] + day,
                "Maintenance_Count_Last_30D": dev["maintenance_count"],
                "Connected_Devices": connected_devices,
                "Optical_RX_Power_dBm": round(rx_power, 2),
                "Optical_TX_Power_dBm": round(tx_power, 2),
                "Temperature_C": round(temp, 2),
                "Voltage_mV": round(voltage, 1),
                "Bias_Current_mA": round(bias, 2),
                "Interface_Error_Count": int(interfaces_errors),
                "Reboot_Count_Last_7D": sum(reboots_queue),
                "Failure_In_7_Days": target_label
            })

df = pd.DataFrame(data)
del data
gc.collect()

# --- 6. Feature Engineering (Adjusted for 3-Hour Interval) ---
print("Calculating Rolling Features...")
df = df.sort_values(by=["Device_ID", "Timestamp"]).reset_index(drop=True)

# 24 hours is now exactly 8 rows (8 * 3H = 24H)
df["RX_Power_24h_Mean"] = df.groupby("Device_ID")["Optical_RX_Power_dBm"].transform(lambda x: x.rolling(8, min_periods=1).mean()).round(2)
df["RX_Power_24h_Std"] = df.groupby("Device_ID")["Optical_RX_Power_dBm"].transform(lambda x: x.rolling(8, min_periods=1).std()).fillna(0).round(3)

# 12 hours is now exactly 4 rows (4 * 3H = 12H)
df["Temp_Trend_Slope_12h"] = df.groupby("Device_ID")["Temperature_C"].transform(lambda x: x.diff(4)).fillna(0).round(2)

filename = "synthetic_gpon_hardened_3h_interval.csv"
df.to_csv(filename, index=False)
print(f"Hardened Dataset Complete! Shape: {df.shape}")

numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()['Failure_In_7_Days'].sort_values(ascending=False)
print("\n--- Correlation with Target ---")
print(corr)
