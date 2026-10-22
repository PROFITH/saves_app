"""
Readers for the CSV files that lab_protocol_timer writes for a live Polar
device (H10 or Verity Sense).

Each stream of a device goes to its own file,
``polar_<kind>_<id>_<stamp>.csv`` (``<id>`` is the sensor ID of the
session, ``<stamp>`` the session stamp ``YYYYMMDD_HHMMSS``), and every row
carries ``timestamp_iso``: the host clock of the timer, to which the
sensor's own clock (``sensor_time_ns``) was mapped. The acceleration file
is the synchronization signal and is read as a ``SensorRecording``. The
companions (heart rate, ECG, PPG, PPI, gyroscope, magnetometer) are plain
dataframes: they are not synchronized on their own but take the clock fit
found with the acceleration of the same device, whose clock they share.
"""

import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from readers.recording import (
    SensorRecording,
    build_recording_frame,
    estimate_sampling_rate,
)

POLAR_ACC_COLUMNS = {"timestamp_iso", "acc_x_mg", "acc_y_mg", "acc_z_mg"}

# Streams of a live device other than the acceleration, with the name each
# file column takes in SAVES (file column -> channel).
POLAR_COMPANION_CHANNELS = {
    "hr": {"hr_bpm": "hr_bpm", "rr_intervals_ms": "rr_intervals_ms", "hr_source": "hr_source"},
    "ecg": {"ecg_uv": "ecg_uv"},
    "ppg": {"ppg0": "ppg0", "ppg1": "ppg1", "ppg2": "ppg2", "ambient": "ambient"},
    "ppi": {
        "ppi_ms": "ppi_ms",
        "hr_bpm": "hr_bpm",
        "error_estimate_ms": "error_estimate_ms",
        "blocker": "blocker",
        "skin_contact": "skin_contact",
        "skin_contact_supported": "skin_contact_supported",
        "index_in_frame": "index_in_frame",
        "hr_source": "hr_source",
    },
    "gyro": {"gyro_x_dps": "gyro_x", "gyro_y_dps": "gyro_y", "gyro_z_dps": "gyro_z"},
    "mag": {"mag_x_gauss": "mag_x", "mag_y_gauss": "mag_y", "mag_z_gauss": "mag_z"},
}
POLAR_COMPANION_KINDS = tuple(POLAR_COMPANION_CHANNELS)
# Files of a live device that hold no signal: the clock fit of every frame
# and the connection log.
POLAR_LOG_KINDS = ("clock", "connection_events")
_TEXT_COLUMNS = {"rr_intervals_ms", "hr_source"}

# Where each heart rate is estimated (the ``hr_source`` vocabulary). SAVES
# computes none of them: every value comes from the device.
HR_SOURCE_H10 = "h10_ecg_firmware"  # the H10 computes it from its ECG
HR_SOURCE_VERITY = "verity_ppg_firmware"  # the Verity Sense computes it from its PPG
HR_SOURCE_PPI = "verity_ppi_firmware"  # Polar's PPI algorithm on the Verity Sense
HR_SOURCE_UNKNOWN = "polar_firmware"  # a Polar of unknown model

_FILE_NAME = re.compile(
    r"^polar_(?P<kind>"
    + "|".join(re.escape(kind) for kind in ("acc", *POLAR_COMPANION_KINDS, *POLAR_LOG_KINDS))
    + r")_(?P<id>.+?)(?:_(?P<stamp>\d{8}_\d+))?$",
    re.IGNORECASE,
)


def polar_file_parts(file_path: Union[str, Path]) -> Optional[tuple[str, str, str]]:  # noqa: UP007, UP045
    """
    ``(kind, id, stamp)`` of a Polar file of lab_protocol_timer
    (``polar_<kind>_<id>_<stamp>.csv``), or None for any other file name.
    ``stamp`` is "" when the name carries none.
    """
    match = _FILE_NAME.match(Path(file_path).stem)
    if match is None:
        return None
    return match.group("kind").lower(), match.group("id"), match.group("stamp") or ""


def is_polar_acc_csv(file_path: Union[str, Path]) -> bool:  # noqa: UP007
    """Return True if the CSV header matches the lab_protocol_timer ACC export."""
    header = pd.read_csv(Path(file_path), nrows=0).columns
    return POLAR_ACC_COLUMNS.issubset({c.strip().lower() for c in header})


def _unix_ms(timestamp_iso: pd.Series, errors: str = "raise") -> np.ndarray:
    """``timestamp_iso`` (UTC) as Unix milliseconds; NaN for invalid values with ``errors="coerce"``."""
    times = pd.to_datetime(timestamp_iso, utc=True, format="ISO8601", errors=errors)
    return (times - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy() * 1000.0


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

    timestamp_unix_ms = _unix_ms(df["timestamp_iso"])

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


def infer_hr_source(hint: str) -> str:
    """
    ``hr_source`` of a heart-rate file written before the column existed,
    guessed from the device type or name of the manifest or from the file
    name: ``polar_h10`` / ``h10`` → the H10 computes it from its ECG,
    ``verity`` / ``sense`` → the Verity Sense computes it from its PPG,
    anything else → a Polar of unknown model.
    """
    text = (hint or "").lower()
    tokens = set(re.split(r"[^a-z0-9]+", text))
    if "h10" in tokens or "polar_h10" in text:
        return HR_SOURCE_H10
    if "verity" in text or "sense" in tokens:
        return HR_SOURCE_VERITY
    return HR_SOURCE_UNKNOWN


def _fill_text(values: pd.Series, default: str) -> pd.Series:
    return values.where(values != "", default)


def read_polar_companion(
    file_path: Union[str, Path],  # noqa: UP007
    kind: Optional[str] = None,  # noqa: UP045
    hr_source_hint: str = "",
) -> pd.DataFrame:
    """
    Read a companion file of a live Polar device into a dataframe.

    Parameters
    ----------
    file_path
        ``polar_<kind>_<id>_<stamp>.csv`` written by lab_protocol_timer.
    kind
        ``hr``, ``ecg``, ``ppg``, ``ppi``, ``gyro`` or ``mag``; taken from
        the file name when omitted.
    hr_source_hint
        Text that names the device (manifest type or name, sensor ID, file
        name), used to fill ``hr_source`` in heart-rate files that lack it.

    Returns
    -------
    pandas.DataFrame
        ``timestamp_unix_ms`` (the ``timestamp_iso`` of each row: host
        clock, UTC), the channels of the stream named as in the SAVES
        export (``POLAR_COMPANION_CHANNELS``) and ``sensor_time_ns`` when
        the file has it. Rows without a valid time are dropped. Older files
        are tolerated: a heart-rate file without ``hr_source`` takes it
        from ``hr_source_hint``, a PPI file without ``hr_source`` gets
        ``verity_ppi_firmware`` and one without ``index_in_frame`` gets
        the pulses of each batch numbered in file order.
    """
    path = Path(file_path)
    if kind is None:
        parts = polar_file_parts(path)
        kind = parts[0] if parts is not None else ""
    if kind not in POLAR_COMPANION_CHANNELS:
        raise ValueError(f"Not a Polar companion file ({', '.join(POLAR_COMPANION_KINDS)}): {path}")

    raw = pd.read_csv(path, dtype={column: str for column in _TEXT_COLUMNS}, keep_default_na=False)
    raw.columns = raw.columns.str.strip().str.lower()
    if "timestamp_iso" not in raw.columns:
        raise ValueError(f"Polar {kind} file is missing the timestamp_iso column: {path}")

    frame = pd.DataFrame({"timestamp_unix_ms": _unix_ms(raw["timestamp_iso"], errors="coerce")})
    for column, channel in POLAR_COMPANION_CHANNELS[kind].items():
        if column in _TEXT_COLUMNS:
            frame[channel] = raw[column].astype(str).str.strip() if column in raw.columns else ""
        elif column in raw.columns:
            frame[channel] = pd.to_numeric(raw[column], errors="coerce")
        else:
            frame[channel] = np.nan
    if "sensor_time_ns" in raw.columns:
        frame["sensor_time_ns"] = pd.to_numeric(raw["sensor_time_ns"], errors="coerce")

    frame = frame[np.isfinite(frame["timestamp_unix_ms"].to_numpy(dtype=float))].reset_index(drop=True)
    if kind == "hr":
        frame["hr_source"] = _fill_text(frame["hr_source"], infer_hr_source(hr_source_hint or path.stem))
    elif kind == "ppi":
        frame["hr_source"] = _fill_text(frame["hr_source"], HR_SOURCE_PPI)
        if "index_in_frame" not in raw.columns:
            frame["index_in_frame"] = frame.groupby("timestamp_unix_ms", sort=False).cumcount()
    return frame


def read_polar_hr(file_path: Union[str, Path], hr_source_hint: str = "") -> pd.DataFrame:  # noqa: UP007
    """Heart rate of the BLE service: ``hr_bpm``, ``rr_intervals_ms`` (``;``-joined, H10 only), ``hr_source``."""
    return read_polar_companion(file_path, "hr", hr_source_hint)


def read_polar_ecg(file_path: Union[str, Path]) -> pd.DataFrame:  # noqa: UP007
    """ECG of the H10: ``ecg_uv`` (microvolts, 130 Hz)."""
    return read_polar_companion(file_path, "ecg")


def read_polar_ppg(file_path: Union[str, Path]) -> pd.DataFrame:  # noqa: UP007
    """PPG of the Verity Sense: ``ppg0``, ``ppg1``, ``ppg2`` and ``ambient`` (raw 22-bit counts, 55 Hz)."""
    return read_polar_companion(file_path, "ppg")


def read_polar_ppi(file_path: Union[str, Path]) -> pd.DataFrame:  # noqa: UP007
    """Pulse-to-pulse intervals of the Verity Sense (Polar's PPI algorithm), one row per pulse."""
    return read_polar_companion(file_path, "ppi")


def read_polar_gyro(file_path: Union[str, Path]) -> pd.DataFrame:  # noqa: UP007
    """Gyroscope of the Verity Sense: ``gyro_x/y/z`` (degrees per second, 52 Hz)."""
    return read_polar_companion(file_path, "gyro")


def read_polar_mag(file_path: Union[str, Path]) -> pd.DataFrame:  # noqa: UP007
    """Magnetometer of the Verity Sense: ``mag_x/y/z`` (gauss, 50 Hz)."""
    return read_polar_companion(file_path, "mag")
