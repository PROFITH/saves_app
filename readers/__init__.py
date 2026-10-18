"""
GUI-independent readers for the sensor formats used in SAVES sessions.

Every reader returns a ``SensorRecording`` whose dataframe holds native
timestamps as ``timestamp_unix_ms`` (UTC) and acceleration in g.
"""

from pathlib import Path
from typing import Optional, Union

from readers.cwa import read_cwa
from readers.matrix import read_matrix
from readers.polar import is_polar_acc_csv, read_polar_acc
from readers.recording import (
    MAGNITUDE_COLUMN,
    TIMESTAMP_COLUMN,
    SensorRecording,
)
from readers.sens import read_sens

SENSOR_FORMATS = ("cwa", "matrix", "sens", "polar_acc")


def infer_format(file_path: Union[str, Path]) -> str:  # noqa: UP007
    """Infer the sensor format from the file extension and CSV header."""
    suffix = Path(file_path).suffix.lower()

    if suffix == ".cwa":
        return "cwa"
    if suffix == ".bin":
        return "matrix"
    if suffix == ".csv":
        return "polar_acc" if is_polar_acc_csv(file_path) else "sens"

    raise ValueError(f"Cannot infer the sensor format of: {file_path}")


def read_sensor(
    file_path: Union[str, Path],  # noqa: UP007
    sensor_format: Optional[str] = None,  # noqa: UP045
    timezone: str = "Europe/Madrid",
) -> SensorRecording:
    """Read any supported sensor file."""
    sensor_format = sensor_format or infer_format(file_path)

    if sensor_format == "cwa":
        return read_cwa(file_path, timezone=timezone)
    if sensor_format == "matrix":
        return read_matrix(file_path)
    if sensor_format == "sens":
        return read_sens(file_path)
    if sensor_format == "polar_acc":
        return read_polar_acc(file_path)

    raise ValueError(
        f"Unsupported sensor format '{sensor_format}'. "
        f"Expected one of: {', '.join(SENSOR_FORMATS)}"
    )


__all__ = [
    "MAGNITUDE_COLUMN",
    "SENSOR_FORMATS",
    "TIMESTAMP_COLUMN",
    "SensorRecording",
    "infer_format",
    "read_cwa",
    "read_matrix",
    "read_polar_acc",
    "read_sens",
    "read_sensor",
]
