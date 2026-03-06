"""Feature engineering utilities for GPON router telemetry data.

This module provides transformation functions applied to validated telemetry
DataFrames before they are fed into the XGBoost failure-prediction model.
Each function accepts a pandas DataFrame and returns a modified copy with
new or transformed columns.
"""

import pandas as pd


def normalize_voltage(df: pd.DataFrame) -> pd.DataFrame:
    """Convert voltage from millivolts to volts.

    Creates a new column ``Voltage_V`` by dividing ``Voltage_mV`` by 1000
    and drops the original ``Voltage_mV`` column to avoid feature leakage
    from redundant representations.

    Args:
        df: DataFrame containing a ``Voltage_mV`` column with float values
            representing the ONT operating voltage in millivolts.

    Returns:
        DataFrame with ``Voltage_mV`` replaced by ``Voltage_V``.

    Raises:
        KeyError: If ``Voltage_mV`` is not present in the DataFrame.
    """
    if "Voltage_mV" not in df.columns:
        raise KeyError("Column 'Voltage_mV' not found in DataFrame")

    df = df.copy()
    df["Voltage_V"] = df["Voltage_mV"] / 1000.0
    df = df.drop(columns=["Voltage_mV"])
    return df


def compute_rx_power_rolling_features(
    df: pd.DataFrame,
    window: str = "24h",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Compute rolling mean and standard deviation of received optical power.

    Calculates a time-based rolling window over ``Optical_RX_Power_dBm`` to
    capture short-term signal degradation trends.  The DataFrame must contain
    a datetime-like column (default ``timestamp``) which is used as the
    rolling-window index.

    New columns added:
        * ``RX_Power_Rolling_Mean_dBm`` -- rolling mean over the window.
        * ``RX_Power_Rolling_Std_dBm``  -- rolling standard deviation over
          the window.

    Args:
        df: DataFrame containing ``Optical_RX_Power_dBm`` (float) and a
            datetime column identified by *timestamp_col*.
        window: Pandas offset alias defining the rolling window size.
            Defaults to ``"24h"``.
        timestamp_col: Name of the datetime column used for time-based
            rolling.  Defaults to ``"timestamp"``.

    Returns:
        DataFrame with two additional rolling-feature columns.

    Raises:
        KeyError: If ``Optical_RX_Power_dBm`` or *timestamp_col* is missing.
    """
    required_cols = {"Optical_RX_Power_dBm", timestamp_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    df = df.set_index(timestamp_col)

    rolling = df["Optical_RX_Power_dBm"].rolling(window=window, min_periods=1)
    df["RX_Power_Rolling_Mean_dBm"] = rolling.mean()
    df["RX_Power_Rolling_Std_dBm"] = rolling.std().fillna(0.0)

    df = df.reset_index()
    return df
