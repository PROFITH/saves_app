"""
Reader for ActiGraph .gt3x files (GT3X+, wGT3X-BT, GT9X Link, CentrePoint).

A .gt3x file is a zip archive with an ``info.txt`` (device, sample rate,
start date, acceleration scale) and the samples, in one of two layouts:

- ``log.bin`` (current firmware): a sequence of records, each with a
  1-byte separator (0x1E), a type, a 4-byte timestamp (seconds since 1970 in
  the device's local time), a 2-byte payload size, the payload and a
  checksum byte. One ``ACTIVITY2`` record (type 0x1A, little-endian int16
  x, y, z) or ``ACTIVITY`` record (type 0x00, 12-bit packed y, x, z) holds
  one second of samples; a ``LUX`` record (type 0x05) holds the light
  reading of its second in lux, as a little-endian unsigned short.
- ``activity.bin`` (older firmware, NHANES layout): one continuous stream of
  12-bit packed y, x, z samples from the start date at the nominal rate.

The format is described in ActiGraph's public GT3X file format
documentation. In idle sleep mode the device stops sampling; following
ActiGraph's convention, the missing seconds are filled with the last
recorded sample and flagged in the ``imputed`` column.

Like the Axivity clock, the ActiGraph clock stores local wall-clock time. It
is converted to UTC with the ``TimeZone`` offset in ``info.txt`` (the offset
when the device was initialized), or with ``timezone`` if the file has none.
"""

from __future__ import annotations

import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union
from zoneinfo import ZoneInfo

import numpy as np

from readers.recording import SensorRecording, build_recording_frame

RECORD_SEPARATOR = 0x1E
RECORD_ACTIVITY = 0x00
RECORD_EVENT = 0x03
RECORD_LUX = 0x05
RECORD_ACTIVITY2 = 0x1A

# .NET ticks (100 ns since 0001-01-01) of the Unix epoch.
_TICKS_AT_UNIX_EPOCH = 621_355_968_000_000_000

# Counts per g when info.txt has no "Acceleration Scale": 256 for the
# GT9X Link (serial prefix MOS), 341 for the GT3X+ and wGT3X-BT families.
_DEFAULT_SCALE_BY_PREFIX = {"MOS": 256.0}
_DEFAULT_SCALE = 341.0


def read_gt3x(
    file_path: Union[str, Path],  # noqa: UP007
    timezone: str = "Europe/Madrid",
    utc_offset_minutes: Optional[float] = None,  # noqa: UP045
    fill_to_last_sample: bool = True,
) -> SensorRecording:
    """
    Read an ActiGraph .gt3x file.

    Parameters
    ----------
    file_path
        Path to the .gt3x file.
    timezone
        IANA time zone used to convert the device clock to UTC when
        ``info.txt`` has no ``TimeZone`` entry.
    utc_offset_minutes
        Fixed offset of the device clock from UTC. Overrides both
        ``info.txt`` and ``timezone``.
    fill_to_last_sample
        Extend the recording with the last sample up to ``Last Sample
        Time`` when the device ended the recording in idle sleep mode.

    Returns
    -------
    SensorRecording
        Samples with ``timestamp_unix_ms`` in UTC, acceleration in g and an
        ``imputed`` column (1 for samples filled during idle sleep or other
        gaps, 0 for recorded samples). Files with LUX records also get
        ``lux``: the reading of each second, kept until the next record
        (NaN before the first one).
    """
    path = Path(file_path)
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if "info.txt" not in names:
            raise ValueError(f"GT3X file has no info.txt: {path}")
        info = parse_info(archive.read("info.txt").decode("utf-8", "replace"))

        rate = float(info.get("Sample Rate", "0") or 0)
        if rate <= 0:
            raise ValueError(f"GT3X file has no valid sample rate: {path}")
        scale = _acceleration_scale(info)

        if "log.bin" in names:
            layout = "log.bin"
            local_s, xyz, imputed, events, lux_records = _read_log_bin(archive.read("log.bin"), rate, scale)
        elif "activity.bin" in names:
            layout = "activity.bin"
            start_local = _ticks_to_local_seconds(info.get("Start Date"))
            if start_local is None:
                raise ValueError(f"GT3X file has no start date: {path}")
            xyz = _unpack_12bit_yxz(archive.read("activity.bin")) / scale
            local_s = start_local + np.arange(xyz.shape[0]) / rate
            imputed = np.zeros(xyz.shape[0], dtype=bool)
            events = []
            lux_records = {}
        else:
            raise ValueError(f"GT3X file has neither log.bin nor activity.bin: {path}")

    if xyz.shape[0] == 0:
        raise ValueError(f"GT3X file contains no acceleration samples: {path}")

    last_sample = _ticks_to_local_seconds(info.get("Last Sample Time"))
    if fill_to_last_sample and last_sample is not None:
        local_s, xyz, imputed = _fill_until(local_s, xyz, imputed, last_sample, rate)

    offset_seconds = _utc_offset_seconds(local_s[0], info.get("TimeZone"), timezone, utc_offset_minutes)
    timestamp_unix_ms = (local_s - offset_seconds) * 1000.0

    extra = {"imputed": imputed.astype(float)}
    if lux_records:
        extra["lux"] = _lux_per_sample(local_s, lux_records)

    data = build_recording_frame(
        timestamp_unix_ms,
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        extra=extra,
    )

    serial = info.get("Serial Number", "")
    metadata = {
        "device_type": info.get("Device Type", ""),
        "device_id": serial,
        "serial": serial,
        "firmware": info.get("Firmware", ""),
        "frequency_hz": rate,
        "accel_scale_counts_per_g": scale,
        "accel_range_g": _float(info.get("Acceleration Max")),
        "layout": layout,
        "n_imputed_samples": int(imputed.sum()),
        "n_lux_records": len(lux_records),
        "idle_sleep_events": events,
        "utc_offset_minutes": offset_seconds / 60.0,
        "info": info,
    }

    return SensorRecording(
        sensor_type="actigraph",
        data=data,
        sampling_rate_hz=rate,
        metadata=metadata,
    )


def parse_info(text: str) -> dict[str, str]:
    """Parse the ``Key: Value`` lines of ``info.txt``."""
    info = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            info[key.strip()] = value.strip()
    return info


def read_gt3x_serial(file_path: Union[str, Path]) -> str:  # noqa: UP007
    """Serial number from ``info.txt``, or an empty string."""
    try:
        with zipfile.ZipFile(file_path) as archive:
            return parse_info(archive.read("info.txt").decode("utf-8", "replace")).get("Serial Number", "")
    except (OSError, KeyError, zipfile.BadZipFile):
        return ""


def _float(value: Optional[str]) -> Optional[float]:  # noqa: UP045
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _acceleration_scale(info: dict[str, str]) -> float:
    scale = _float(info.get("Acceleration Scale"))
    if scale and scale > 0:
        return scale
    serial = info.get("Serial Number", "")
    return _DEFAULT_SCALE_BY_PREFIX.get(serial[:3], _DEFAULT_SCALE)


def _ticks_to_local_seconds(ticks: Optional[str]) -> Optional[float]:  # noqa: UP045
    """Convert .NET ticks of local time to seconds since 1970 (local)."""
    try:
        value = int(ticks) if ticks else 0
    except ValueError:
        return None
    if value <= 0:
        return None
    return (value - _TICKS_AT_UNIX_EPOCH) / 1e7


def _unpack_12bit_yxz(payload: bytes) -> np.ndarray:
    """Decode 12-bit packed samples (big-endian bit stream, axes y, x, z)."""
    data = np.frombuffer(payload, dtype=np.uint8)
    n_samples = (data.size * 8) // 36
    if n_samples == 0:
        return np.zeros((0, 3))
    bits = np.unpackbits(data)[: n_samples * 36].reshape(n_samples * 3, 12)
    values = bits.dot(1 << np.arange(11, -1, -1)).astype(np.int64)
    values = np.where(values >= 2048, values - 4096, values).reshape(n_samples, 3)
    return values[:, [1, 0, 2]].astype(float)


def _read_log_bin(log: bytes, rate: float, scale: float):
    """Samples of ACTIVITY / ACTIVITY2 records with idle-sleep gaps filled, plus the LUX records by second."""
    per_record = round(rate)
    times: list[int] = []
    blocks: list[np.ndarray] = []
    events: list[dict] = []
    lux: dict[int, int] = {}

    offset = 0
    size_total = len(log)
    while offset + 8 <= size_total:
        if log[offset] != RECORD_SEPARATOR:
            # Resynchronize on the next separator after a corrupt record.
            nxt = log.find(bytes([RECORD_SEPARATOR]), offset + 1)
            if nxt < 0:
                break
            offset = nxt
            continue
        record_type = log[offset + 1]
        timestamp = int.from_bytes(log[offset + 2:offset + 6], "little")
        size = int.from_bytes(log[offset + 6:offset + 8], "little")
        payload = log[offset + 8:offset + 8 + size]
        offset += 8 + size + 1
        if len(payload) < size:
            break

        if record_type == RECORD_ACTIVITY2 and size >= 6:
            samples = np.frombuffer(payload[: (size // 6) * 6], dtype="<i2").reshape(-1, 3).astype(float)
        elif record_type == RECORD_ACTIVITY and size > 1:
            samples = _unpack_12bit_yxz(payload)
        else:
            if record_type == RECORD_EVENT and size >= 1 and payload[0] in (0x08, 0x09):
                events.append({
                    "local_unix_s": timestamp,
                    "event": "idle_sleep_start" if payload[0] == 0x08 else "idle_sleep_end",
                })
            elif record_type == RECORD_LUX and size >= 2:
                lux[timestamp] = int.from_bytes(payload[:2], "little")
            # 1-byte activity records carry no samples (USB connection marker).
            continue

        times.append(timestamp)
        blocks.append(samples[:per_record] / scale)

    if not blocks:
        return np.zeros(0), np.zeros((0, 3)), np.zeros(0, dtype=bool), events, lux

    order = np.argsort(np.asarray(times), kind="stable")
    times_arr = np.asarray(times)[order]
    blocks = [blocks[i] for i in order]

    local_parts: list[np.ndarray] = []
    xyz_parts: list[np.ndarray] = []
    imputed_parts: list[np.ndarray] = []
    last_value = blocks[0][0]
    previous_end: Optional[float] = None  # noqa: UP045

    for second, samples in zip(times_arr, blocks):
        start = float(second)
        if previous_end is not None and start > previous_end + 0.5 / rate:
            n_gap = round((start - previous_end) * rate)
            if n_gap > 0:
                local_parts.append(previous_end + np.arange(n_gap) / rate)
                xyz_parts.append(np.repeat(last_value[None, :], n_gap, axis=0))
                imputed_parts.append(np.ones(n_gap, dtype=bool))
        elif previous_end is not None and start < previous_end - 0.5 / rate:
            continue  # Duplicate second.
        n = samples.shape[0]
        if n == 0:
            continue
        local_parts.append(start + np.arange(n) / rate)
        xyz_parts.append(samples)
        imputed_parts.append(np.zeros(n, dtype=bool))
        last_value = samples[-1]
        previous_end = start + n / rate

    return (
        np.concatenate(local_parts),
        np.concatenate(xyz_parts),
        np.concatenate(imputed_parts),
        events,
        lux,
    )


def _lux_per_sample(local_s: np.ndarray, lux: dict[int, int]) -> np.ndarray:
    """Lux of each sample: the last LUX record at or before its second (NaN before the first)."""
    seconds = np.array(sorted(lux), dtype=float)
    values = np.array([lux[int(s)] for s in seconds], dtype=float)
    index = np.searchsorted(seconds, np.floor(local_s + 1e-6), side="right") - 1
    out = values[np.clip(index, 0, None)]
    out[index < 0] = np.nan
    return out


def _fill_until(local_s, xyz, imputed, last_local_s, rate):
    """Repeat the last sample until ``last_local_s`` (idle sleep at the end)."""
    end = local_s[-1] + 1.0 / rate
    n_gap = round((last_local_s - end) * rate)
    if n_gap <= 0:
        return local_s, xyz, imputed
    return (
        np.concatenate([local_s, end + np.arange(n_gap) / rate]),
        np.concatenate([xyz, np.repeat(xyz[-1:], n_gap, axis=0)]),
        np.concatenate([imputed, np.ones(n_gap, dtype=bool)]),
    )


def _utc_offset_seconds(
    first_local_seconds: float,
    info_timezone: Optional[str],  # noqa: UP045
    timezone: str,
    utc_offset_minutes: Optional[float],  # noqa: UP045
) -> float:
    """Offset of the device clock from UTC, in seconds."""
    if utc_offset_minutes is not None:
        return float(utc_offset_minutes) * 60.0
    if info_timezone:
        text = info_timezone.strip()
        sign = -1.0 if text.startswith("-") else 1.0
        parts = text.lstrip("+-").split(":")
        try:
            hours, minutes = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
            return sign * (hours * 3600 + minutes * 60)
        except ValueError:
            pass
    local = datetime(1970, 1, 1) + timedelta(seconds=float(first_local_seconds))
    offset = local.replace(tzinfo=ZoneInfo(timezone)).utcoffset()
    return offset.total_seconds() if offset is not None else 0.0
