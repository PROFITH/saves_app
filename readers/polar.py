from pathlib import Path
from typing import Union

import pandas as pd

from readers.recording import (
    SensorRecording,
    build_recording_frame,
    estimate_sampling_rate,
)

POLAR_ACC_COLUMNS = {"timestamp_iso", "acc_x_mg", "acc_y_mg", "acc_z_mg"}


def is_polar_acc_csv(file_path: Union[str, Path]) -> bool:  # noqa: UP007
    """Return True if the CSV header matches the lab_protocol_timer ACC export."""
    header = pd.read_csv(Path(file_path), nrows=0).columns
    return POLAR_ACC_COLUMNS.issubset({c.strip().lower() for c in header})


def read_polar_acc(file_path: Union[str, Path]) -> SensorRecording:  # noqa: UP007
    """
    Read the Polar acceleration CSV exported by lab_protocol_timer.

    Timestamps are ISO-8601 in UTC, already mapped to the host clock
    (``native_plus_host_utc``); acceleration is converted from mg to g.
    """
    df = pd.read_csv(Path(file_path))
    df.columns = df.columns.str.strip().str.lower()

    missing_columns = POLAR_ACC_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "Polar ACC file is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    times = pd.to_datetime(df["timestamp_iso"], utc=True, format="ISO8601")
    timestamp_unix_ms = (
        times - pd.Timestamp("1970-01-01", tz="UTC")
    ).dt.total_seconds().to_numpy() * 1000.0

    return SensorRecording(
        sensor_type="polar_acc",
        data=build_recording_frame(
            timestamp_unix_ms,
            df["acc_x_mg"].to_numpy() / 1000.0,
            df["acc_y_mg"].to_numpy() / 1000.0,
            df["acc_z_mg"].to_numpy() / 1000.0,
        ),
        sampling_rate_hz=estimate_sampling_rate(timestamp_unix_ms),
    )
