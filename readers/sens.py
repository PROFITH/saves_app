from pathlib import Path
from typing import Union

import pandas as pd

from readers.recording import (
    SensorRecording,
    build_recording_frame,
    estimate_sampling_rate,
)


def read_sens(file_path: Union[str, Path]) -> SensorRecording:  # noqa: UP007
    """
    Read a SENS motion accelerometer CSV file.

    The native ``unixts`` timestamps (Unix milliseconds) are preserved;
    they must not be replaced by a reconstructed time axis.
    """
    df = pd.read_csv(Path(file_path))
    df.columns = df.columns.str.strip().str.lower()

    required_columns = {"unixts", "x", "y", "z"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "SENS file is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    timestamp_unix_ms = pd.to_numeric(df["unixts"], errors="coerce")

    if timestamp_unix_ms.isna().any():
        raise ValueError(
            "SENS file contains invalid values in the 'unixts' column."
        )

    return SensorRecording(
        sensor_type="sens",
        data=build_recording_frame(
            timestamp_unix_ms.to_numpy(),
            df["x"].to_numpy(),
            df["y"].to_numpy(),
            df["z"].to_numpy(),
        ),
        sampling_rate_hz=estimate_sampling_rate(timestamp_unix_ms.to_numpy()),
    )
