from pathlib import Path
from typing import Union

import pandas as pd

from readers.recording import (
    MAGNITUDE_COLUMN,
    TIMESTAMP_COLUMN,
    SensorRecording,
    build_recording_frame,
    estimate_sampling_rate,
)

REQUIRED_COLUMNS = {"unixts", "x", "y", "z"}


def read_sens(file_path: Union[str, Path]) -> SensorRecording:  # noqa: UP007
    """
    Read a SENS motion accelerometer CSV file.

    The native ``unixts`` timestamps (Unix milliseconds) are preserved;
    they must not be replaced by a reconstructed time axis. Any other
    numeric column of the file is kept as an extra channel, named as in
    the file (in lower case).
    """
    df = pd.read_csv(Path(file_path))
    df.columns = df.columns.str.strip().str.lower()
    if not df.columns.is_unique:
        df = df.loc[:, ~df.columns.duplicated()]

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)

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

    reserved = REQUIRED_COLUMNS | {TIMESTAMP_COLUMN, MAGNITUDE_COLUMN.lower(), "vm"}
    extra = {
        name: df[name].to_numpy(dtype=float)
        for name in df.columns
        if name not in reserved and pd.api.types.is_numeric_dtype(df[name])
    }

    return SensorRecording(
        sensor_type="sens",
        data=build_recording_frame(
            timestamp_unix_ms.to_numpy(),
            df["x"].to_numpy(),
            df["y"].to_numpy(),
            df["z"].to_numpy(),
            extra=extra,
        ),
        sampling_rate_hz=estimate_sampling_rate(timestamp_unix_ms.to_numpy()),
    )
