"""
GUI-independent readers for the sensor formats used in SAVES sessions.

Every reader returns a ``SensorRecording`` whose dataframe holds native
timestamps as ``timestamp_unix_ms`` (UTC), acceleration in g and every
other channel the file holds (gyroscope, light, temperature, heart rate...)
on the same sample grid. The companion files of a live Polar device (heart
rate, ECG, PPG, PPI, gyroscope, magnetometer) are read as plain dataframes
with ``read_polar_companion``.
"""

from pathlib import Path
from typing import Optional, Union

from readers.cwa import read_cwa
from readers.geneactiv import is_geneactiv_bin, read_geneactiv
from readers.gt3x import read_gt3x
from readers.matrix import read_matrix
from readers.polar import (
    POLAR_COMPANION_CHANNELS,
    POLAR_COMPANION_KINDS,
    infer_hr_source,
    is_polar_acc_csv,
    polar_file_parts,
    read_polar_acc,
    read_polar_companion,
    read_polar_ecg,
    read_polar_gyro,
    read_polar_hr,
    read_polar_mag,
    read_polar_ppg,
    read_polar_ppi,
)
from readers.recording import (
    MAGNITUDE_COLUMN,
    TIMESTAMP_COLUMN,
    SensorRecording,
)
from readers.sens import read_sens

SENSOR_FORMATS = ("cwa", "gt3x", "geneactiv", "matrix", "sens", "polar_acc")

# Sensor types of the session manifest (lab_protocol_timer) and their format.
FORMAT_BY_SENSOR_TYPE = {
    "ax3": "cwa",
    "ax6": "cwa",
    "actigraph_gt3x": "gt3x",
    "geneactiv": "geneactiv",
    "matrix": "matrix",
    "sens": "sens",
    "polar_h10": "polar_acc",
    "verity_sense": "polar_acc",
}


def infer_format(file_path: Union[str, Path]) -> str:  # noqa: UP007
    """Infer the sensor format from the file extension and CSV header."""
    suffix = Path(file_path).suffix.lower()

    if suffix == ".cwa":
        return "cwa"
    if suffix == ".gt3x":
        return "gt3x"
    if suffix == ".bin":
        # GENEActiv and Parmay Matrix both use .bin; GENEActiv files are text.
        return "geneactiv" if is_geneactiv_bin(file_path) else "matrix"
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
    if sensor_format == "gt3x":
        return read_gt3x(file_path, timezone=timezone)
    if sensor_format == "geneactiv":
        return read_geneactiv(file_path, timezone=timezone)
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
    "FORMAT_BY_SENSOR_TYPE",
    "MAGNITUDE_COLUMN",
    "POLAR_COMPANION_CHANNELS",
    "POLAR_COMPANION_KINDS",
    "SENSOR_FORMATS",
    "TIMESTAMP_COLUMN",
    "SensorRecording",
    "infer_format",
    "infer_hr_source",
    "polar_file_parts",
    "read_cwa",
    "read_geneactiv",
    "read_gt3x",
    "read_matrix",
    "read_polar_acc",
    "read_polar_companion",
    "read_polar_ecg",
    "read_polar_gyro",
    "read_polar_hr",
    "read_polar_mag",
    "read_polar_ppg",
    "read_polar_ppi",
    "read_sens",
    "read_sensor",
]
