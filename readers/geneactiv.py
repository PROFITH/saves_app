"""
Reader for GENEActiv binary files (.bin).

A GENEActiv .bin file is text (CRLF lines): a header with blocks such as
``Device Identity``, ``Configuration Info`` and ``Calibration Data``, then one
block per page. Each page starts with ``Recorded Data``, carries its own
``Page Time`` and ``Temperature``, and ends with one line of 300 samples
written as hexadecimal. Each sample is 48 bits: x, y and z as signed 12-bit
integers, 10 bits of light, the button bit and a reserved bit.

Acceleration in g is ``(raw * 100 - offset) / gain`` per axis, with the gains
and offsets of the calibration block; light in lux is ``raw * Lux / Volts``.

Sample times are interpolated between consecutive page times, like the
Axivity reader does with sector times, so the real sampling rate is
followed. The device clock stores local time; it is converted to UTC with the
``Time Zone`` of the header (``GMT +01:00``) or with ``timezone``.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from pathlib import Path
from typing import Optional, Union
from zoneinfo import ZoneInfo

import numpy as np

from readers.recording import SensorRecording, build_recording_frame

SAMPLES_PER_PAGE = 300
HEX_PER_SAMPLE = 12
PAGE_HEX_LENGTH = SAMPLES_PER_PAGE * HEX_PER_SAMPLE
SIGNATURE = b"Device Identity"


def is_geneactiv_bin(file_path: Union[str, Path]) -> bool:  # noqa: UP007
    """True if the file starts like a GENEActiv binary file."""
    try:
        with open(file_path, "rb") as f:
            return f.read(len(SIGNATURE)) == SIGNATURE
    except OSError:
        return False


def read_geneactiv(
    file_path: Union[str, Path],  # noqa: UP007
    timezone: str = "Europe/Madrid",
    utc_offset_minutes: Optional[float] = None,  # noqa: UP045
) -> SensorRecording:
    """
    Read a GENEActiv .bin file.

    Returns
    -------
    SensorRecording
        Samples with ``timestamp_unix_ms`` in UTC, calibrated acceleration
        in g, ``light`` (lux), ``button`` (0/1) and ``temperature`` (°C, per
        page).
    """
    path = Path(file_path)
    text = path.read_bytes().decode("latin-1")
    lines = text.split("\n")
    lines = [line.rstrip("\r") for line in lines]

    try:
        first_page = lines.index("Recorded Data")
    except ValueError:
        raise ValueError(f"GENEActiv file has no data pages: {path}") from None

    header = parse_header(lines[:first_page])
    calibration = {
        key: _number(header.get(key))
        for key in ("x gain", "x offset", "y gain", "y offset", "z gain", "z offset", "Volts", "Lux")
    }
    if not all(calibration[k] for k in ("x gain", "y gain", "z gain")):
        raise ValueError(f"GENEActiv file has no calibration gains: {path}")
    nominal_rate = _number(header.get("Measurement Frequency"))

    page_times: list[float] = []
    temperatures: list[float] = []
    hex_pages: list[str] = []
    page_rates: list[float] = []

    i = first_page
    n_lines = len(lines)
    while i < n_lines:
        if lines[i] != "Recorded Data":
            i += 1
            continue
        fields: dict[str, str] = {}
        j = i + 1
        while j < n_lines and lines[j] != "Recorded Data" and len(lines[j]) < PAGE_HEX_LENGTH:
            key, sep, value = lines[j].partition(":")
            if sep:
                fields[key.strip()] = value.strip()
            j += 1
        if j >= n_lines or len(lines[j]) < PAGE_HEX_LENGTH:
            break  # Truncated last page.
        page_time = _parse_time(fields.get("Page Time"))
        if page_time is not None:
            page_times.append(page_time)
            temperatures.append(_number(fields.get("Temperature")) or np.nan)
            page_rates.append(_number(fields.get("Measurement Frequency")) or nominal_rate or 0.0)
            hex_pages.append(lines[j][:PAGE_HEX_LENGTH])
        i = j + 1

    if not hex_pages:
        raise ValueError(f"GENEActiv file contains no complete pages: {path}")

    raw = np.frombuffer("".join(hex_pages).encode("ascii"), dtype=np.uint8)
    nibbles = _hex_to_nibbles(raw).reshape(-1, HEX_PER_SAMPLE).astype(np.int64)

    def field(start: int, length: int) -> np.ndarray:
        value = np.zeros(nibbles.shape[0], dtype=np.int64)
        for k in range(length):
            value = (value << 4) | nibbles[:, start + k]
        return value

    def signed12(values: np.ndarray) -> np.ndarray:
        return np.where(values >= 2048, values - 4096, values)

    x_raw = signed12(field(0, 3))
    y_raw = signed12(field(3, 3))
    z_raw = signed12(field(6, 3))
    tail = field(9, 3)  # 10 bits light, button, reserved
    light_raw = tail >> 2
    button = (tail >> 1) & 1

    x = (x_raw * 100.0 - calibration["x offset"]) / calibration["x gain"]
    y = (y_raw * 100.0 - calibration["y offset"]) / calibration["y gain"]
    z = (z_raw * 100.0 - calibration["z offset"]) / calibration["z gain"]
    volts = calibration["Volts"] or 0.0
    light = light_raw * (calibration["Lux"] or 0.0) / volts if volts else light_raw.astype(float)

    local_s = _sample_times(np.asarray(page_times), np.asarray(page_rates))
    temperature = np.repeat(np.asarray(temperatures, dtype=float), SAMPLES_PER_PAGE)

    offset_seconds = _utc_offset_seconds(local_s[0], header.get("Time Zone"), timezone, utc_offset_minutes)
    timestamp_unix_ms = (local_s - offset_seconds) * 1000.0

    data = build_recording_frame(
        timestamp_unix_ms,
        x,
        y,
        z,
        extra={"light": light, "button": button, "temperature": temperature},
    )

    rate = nominal_rate or float(np.median(page_rates))
    serial = header.get("Device Unique Serial Code", "").strip()
    metadata = {
        "device_type": header.get("Device Type", "GENEActiv").strip(),
        "device_id": serial,
        "serial": serial,
        "firmware": header.get("Device Firmware Version", "").strip(),
        "frequency_hz": rate,
        "accel_range_g": _range_max(header.get("Accelerometer Range")),
        "calibration": calibration,
        "n_pages": len(hex_pages),
        "n_pages_in_header": int(_number(header.get("Number of Pages")) or 0),
        "device_location": header.get("Device Location Code", "").strip(),
        "utc_offset_minutes": offset_seconds / 60.0,
    }

    return SensorRecording(
        sensor_type="geneactiv",
        data=data,
        sampling_rate_hz=rate,
        metadata=metadata,
    )


def parse_header(lines: list[str]) -> dict[str, str]:
    """``Key:Value`` lines of the header (NUL padding removed)."""
    header = {}
    for line in lines:
        key, sep, value = line.partition(":")
        if sep:
            header[key.strip()] = value.replace("\x00", "").strip()
    return header


def read_geneactiv_serial(file_path: Union[str, Path]) -> str:  # noqa: UP007
    """Serial number from the header, or an empty string."""
    try:
        with open(file_path, "rb") as f:
            head = f.read(4096).decode("latin-1")
    except OSError:
        return ""
    match = re.search(r"Device Unique Serial Code:([^\r\n]*)", head)
    return match.group(1).strip() if match else ""


def _number(value: Optional[str]) -> Optional[float]:  # noqa: UP045
    if not value:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return float(match.group()) if match else None


def _range_max(value: Optional[str]) -> Optional[float]:  # noqa: UP045
    numbers = re.findall(r"-?\d+(?:\.\d+)?", value or "")
    return float(numbers[-1]) if numbers else None


def _parse_time(value: Optional[str]) -> Optional[float]:  # noqa: UP045
    """``2013-05-30 10:12:54:500`` → local seconds since 1970."""
    if not value:
        return None
    match = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2}) (\d{1,2}):(\d{2}):(\d{2})(?::(\d{1,3}))?", value)
    if not match:
        return None
    year, month, day, hour, minute, second = (int(g) for g in match.groups()[:6])
    millis = int(match.group(7) or 0)
    moment = datetime(year, month, day, hour, minute, second, tzinfo=dt_timezone.utc)
    return moment.timestamp() + millis / 1000.0


def _hex_to_nibbles(raw: np.ndarray) -> np.ndarray:
    """ASCII hexadecimal digits to their values (0-15)."""
    nibbles = raw.astype(np.int16) - ord("0")
    letters = raw >= ord("A")
    nibbles[letters] = (raw[letters] | 0x20).astype(np.int16) - ord("a") + 10
    if np.any((nibbles < 0) | (nibbles > 15)):
        raise ValueError("GENEActiv page contains characters that are not hexadecimal.")
    return nibbles


def _sample_times(page_times: np.ndarray, page_rates: np.ndarray) -> np.ndarray:
    """Interpolate sample times between page times; nominal rate at gaps."""
    n_pages = page_times.size
    nominal = SAMPLES_PER_PAGE / np.where(page_rates > 0, page_rates, np.nan)
    durations = np.empty(n_pages)
    durations[:-1] = np.diff(page_times)
    durations[-1] = nominal[-1]
    # Use the page-to-page interval only when it is close to the nominal one
    # (otherwise the device paused, or a page is missing).
    plausible = np.abs(durations - nominal) <= 0.1 * nominal
    durations = np.where(plausible, durations, nominal)
    steps = durations / SAMPLES_PER_PAGE
    return (page_times[:, None] + steps[:, None] * np.arange(SAMPLES_PER_PAGE)[None, :]).ravel()


def _utc_offset_seconds(
    first_local_seconds: float,
    header_timezone: Optional[str],  # noqa: UP045
    timezone: str,
    utc_offset_minutes: Optional[float],  # noqa: UP045
) -> float:
    if utc_offset_minutes is not None:
        return float(utc_offset_minutes) * 60.0
    match = re.search(r"([+-])\s*(\d{1,2}):?(\d{2})?", header_timezone or "")
    if match:
        sign = -1.0 if match.group(1) == "-" else 1.0
        return sign * (int(match.group(2)) * 3600 + int(match.group(3) or 0) * 60)
    local = datetime(1970, 1, 1) + timedelta(seconds=float(first_local_seconds))
    offset = local.replace(tzinfo=ZoneInfo(timezone)).utcoffset()
    return offset.total_seconds() if offset is not None else 0.0
