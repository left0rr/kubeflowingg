"""Pydantic validation schema for GPON router telemetry records.

This module defines the TelemetryRecord model used to validate incoming
telemetry data from GPON optical network terminals (ONTs) before ingestion
into the ML pipeline. Each record represents a single device snapshot used
for predicting router failures within a 7-day horizon.
"""

from pydantic import BaseModel, Field, field_validator


class TelemetryRecord(BaseModel):
    """Validated telemetry snapshot from a single GPON router.

    Attributes:
        Optical_RX_Power_dBm: Received optical signal power in dBm.
            Typical range: -40.0 to 0.0 dBm.
        Optical_TX_Power_dBm: Transmitted optical signal power in dBm.
            Typical range: -10.0 to 10.0 dBm.
        Temperature_C: Internal device temperature in Celsius.
            Valid range: -40.0 to 125.0 C.
        Voltage_mV: Operating voltage in millivolts.
            Valid range: 0.0 to 5000.0 mV.
        Bias_Current_mA: Laser bias current in milliamps.
            Valid range: 0.0 to 200.0 mA.
        Interface_Error_Count: Cumulative interface errors since last reset.
            Must be non-negative.
        Reboot_Count_Last_7D: Number of device reboots in the last 7 days.
            Must be non-negative.
        Connected_Devices: Number of devices connected to this ONT.
            Must be non-negative.
        Device_Age_Days: Age of the device in days since deployment.
            Must be non-negative.
        Maintenance_Count_Last_30D: Maintenance events in the last 30 days.
            Must be non-negative.
        Failure_In_7_Days: Binary target label indicating whether a failure
            occurs within the next 7 days (0 = no failure, 1 = failure).
    """

    Optical_RX_Power_dBm: float = Field(
        ...,
        ge=-40.0,
        le=0.0,
        description="Received optical signal power in dBm",
    )
    Optical_TX_Power_dBm: float = Field(
        ...,
        ge=-10.0,
        le=10.0,
        description="Transmitted optical signal power in dBm",
    )
    Temperature_C: float = Field(
        ...,
        ge=-40.0,
        le=125.0,
        description="Internal device temperature in Celsius",
    )
    Voltage_mV: float = Field(
        ...,
        ge=0.0,
        le=5000.0,
        description="Operating voltage in millivolts",
    )
    Bias_Current_mA: float = Field(
        ...,
        ge=0.0,
        le=200.0,
        description="Laser bias current in milliamps",
    )
    Interface_Error_Count: int = Field(
        ...,
        ge=0,
        description="Cumulative interface errors since last reset",
    )
    Reboot_Count_Last_7D: int = Field(
        ...,
        ge=0,
        description="Number of device reboots in the last 7 days",
    )
    Connected_Devices: int = Field(
        ...,
        ge=0,
        description="Number of devices connected to this ONT",
    )
    Device_Age_Days: int = Field(
        ...,
        ge=0,
        description="Age of the device in days since deployment",
    )
    Maintenance_Count_Last_30D: int = Field(
        ...,
        ge=0,
        description="Maintenance events in the last 30 days",
    )
    Failure_In_7_Days: int = Field(
        ...,
        ge=0,
        le=1,
        description="Binary target: 1 if failure within 7 days, else 0",
    )

    @field_validator("Failure_In_7_Days")
    @classmethod
    def validate_binary_target(cls, v: int) -> int:
        """Ensure the target label is strictly binary."""
        if v not in (0, 1):
            raise ValueError("Failure_In_7_Days must be 0 or 1")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "Optical_RX_Power_dBm": -18.5,
                "Optical_TX_Power_dBm": 2.3,
                "Temperature_C": 42.1,
                "Voltage_mV": 3300.0,
                "Bias_Current_mA": 35.2,
                "Interface_Error_Count": 12,
                "Reboot_Count_Last_7D": 1,
                "Connected_Devices": 4,
                "Device_Age_Days": 730,
                "Maintenance_Count_Last_30D": 0,
                "Failure_In_7_Days": 0,
            }
        }
