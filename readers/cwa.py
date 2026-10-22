"""
Reader for Axivity AX3 / AX6 continuous wave accelerometer files (.cwa).

The binary layout follows the Open Movement specification
(https://github.com/digitalinteraction/openmovement). Each 512-byte data
sector carries its own real-time-clock timestamp, so sample times are
interpolated between sector timestamps rather than reconstructed from the
nominal sampling rate.

The AX3/AX6 clock stores local wall-clock time without a time zone (it is
set from the configuring computer). It is converted to UTC with the
offset that applied at the first sample: the device clock does not follow
daylight-saving changes during a recording.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union
from urllib.parse import parse_qsl
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from readers.recording import SensorRecording, build_recording_frame

SECTOR_SIZE = 512
DATA_PACKET_HEADER = 0x5841  # "AX" little-endian
DATA_PACKET_LENGTH = 508

_SECTOR_DTYPE = np.dtype([
    ("packet_header", "<u2"),
    ("packet_length", "<u2"),
    ("device_fractional", "<u2"),
    ("session_id", "<u4"),
    ("sequence_id", "<u4"),
    ("timestamp_packed", "<u4"),
    ("scale_light", "<u2"),
    ("temperature", "<u2"),
    ("events", "u1"),
    ("battery", "u1"),
    ("rate_code", "u1"),
    ("num_axes_bps", "u1"),
    ("timestamp_offset", "<i2"),
    ("sample_count", "<u2"),
    ("raw", "V480"),
    ("checksum", "<u2"),
])


def read_cwa(
    file_path: Union[str, Path],  # noqa: UP007
    timezone: str = "Europe/Madrid",
    utc_offset_minutes: Optional[float] = None,  # noqa: UP045
) -> SensorRecording:
    """
    Read an Axivity .cwa file.

    Parameters
    ----------
    file_path
        Path to the .cwa file.
    timezone
        IANA time zone of the computer that configured the device. Used to
        convert the device clock to UTC.
    utc_offset_minutes
        Fixed offset of the device clock from UTC (e.g. 120 for CEST). When
        given, it overrides ``timezone``.

    Returns
    -------
    SensorRecording
        Samples with ``timestamp_unix_ms`` in UTC and acceleration in g;
        ``gyro_x/y/z`` (degrees per second) are added for AX6 files with
        gyroscope enabled. ``light_raw`` (the 10-bit reading of the light
        sensor, uncalibrated) and ``temperature_c`` (°C, NaN when the
        sector has no reading) are read once per sector and repeated on
        every sample of their sector.
    """
    path = Path(file_path)
    buffer = np.fromfile(path, dtype=np.uint8)

    header = _parse_header(buffer)
    data_offset = header["data_offset"]

    n_sectors = (buffer.size - data_offset) // SECTOR_SIZE
    if n_sectors <= 0:
        raise ValueError(f"CWA file contains no data sectors: {path}")

    raw_sectors = buffer[data_offset:data_offset + n_sectors * SECTOR_SIZE]
    sectors = raw_sectors.view(_SECTOR_DTYPE)

    words = raw_sectors.view("<u2").reshape(n_sectors, SECTOR_SIZE // 2)
    checksum_ok = words.sum(axis=1, dtype=np.uint64) % 65536 == 0

    structurally_valid = (
        checksum_ok
        & (sectors["packet_header"] == DATA_PACKET_HEADER)
        & (sectors["packet_length"] == DATA_PACKET_LENGTH)
        & (sectors["sample_count"] > 0)
    )

    if not structurally_valid.any():
        raise ValueError(f"CWA file contains no valid data sectors: {path}")

    first = int(np.argmax(structurally_valid))
    rate_code = int(sectors["rate_code"][first])
    num_axes_bps = int(sectors["num_axes_bps"][first])

    valid = (
        structurally_valid
        & (sectors["rate_code"] == rate_code)
        & (sectors["num_axes_bps"] == num_axes_bps)
    )

    sectors = sectors[valid]
    raw_sectors = raw_sectors.reshape(n_sectors, SECTOR_SIZE)[valid]

    frequency = 3200.0 / (1 << (15 - (rate_code & 0x0F)))
    accel_range = 16 >> (rate_code >> 6) if rate_code != 0 else 16

    channels = (num_axes_bps >> 4) & 0x0F
    bytes_per_axis = num_axes_bps & 0x0F

    values, samples_per_sector = _unpack_samples(
        raw_sectors,
        channels,
        bytes_per_axis,
    )

    sample_count = np.minimum(
        sectors["sample_count"].astype(np.int64),
        samples_per_sector,
    )
    keep = (
        np.arange(samples_per_sector)[None, :] < sample_count[:, None]
    )

    # Per-sector scaling: top three bits of scale_light set the
    # accelerometer unit (AX6), the next three the gyroscope range.
    scale_light = sectors["scale_light"].astype(np.int64)
    accel_unit = (1 << (8 + ((scale_light >> 13) & 0x07))).astype(float)
    gyro_code = (scale_light >> 10) & 0x07
    gyro_range = np.where(
        gyro_code != 0,
        8000 // (1 << gyro_code),
        2000,
    ).astype(float)

    accel_axis = 3 if channels >= 6 else 0

    accel = values[:, :, accel_axis:accel_axis + 3] / accel_unit[:, None, None]
    accel = accel[keep]

    extra = {}
    if channels >= 6:
        gyro = values[:, :, 0:3] * (gyro_range[:, None, None] / 32768.0)
        gyro = gyro[keep]
        extra = {
            "gyro_x": gyro[:, 0],
            "gyro_y": gyro[:, 1],
            "gyro_z": gyro[:, 2],
        }

    # Light and temperature are read once per sector (about 1.2 times per
    # second at 100 Hz) and repeated on every sample of the sector.
    temperature_raw = sectors["temperature"].astype(np.int64) & 0x03FF
    extra["light_raw"] = np.repeat(scale_light & 0x03FF, sample_count)
    extra["temperature_c"] = np.repeat(temperature_celsius(temperature_raw), sample_count)

    local_seconds = _sample_times_local_seconds(
        sectors,
        sample_count,
        frequency,
    )

    offset_seconds = _utc_offset_seconds(
        local_seconds[0],
        timezone,
        utc_offset_minutes,
    )

    timestamp_unix_ms = (local_seconds - offset_seconds) * 1000.0

    data = build_recording_frame(
        timestamp_unix_ms,
        accel[:, 0],
        accel[:, 1],
        accel[:, 2],
        extra=extra,
    )

    metadata = {
        **{
            key: value
            for key, value in header.items()
            if key != "data_offset"
        },
        "frequency_hz": frequency,
        "accel_range_g": accel_range,
        "channels": channels,
        "n_sectors": int(n_sectors),
        "n_invalid_sectors": int(n_sectors - valid.sum()),
        "utc_offset_minutes": offset_seconds / 60.0,
    }

    return SensorRecording(
        sensor_type="axivity",
        data=data,
        sampling_rate_hz=frequency,
        metadata=metadata,
    )


def temperature_celsius(raw: np.ndarray) -> np.ndarray:
    """
    Degrees Celsius from the 10-bit temperature reading of a sector.

    Axivity documents the conversion as ``(counts - 171) / 3.413``, the
    same line that Open Movement's and GGIR's readers write as
    ``counts * 75 / 256 - 50``. A reading of 0 means no measurement (NaN).
    """
    raw = np.asarray(raw, dtype=float)
    return np.where(raw > 0, raw * 75.0 / 256.0 - 50.0, np.nan)


def _parse_header(buffer: np.ndarray) -> dict:
    """Parse the 1024-byte "MD" header block."""
    if buffer.size < SECTOR_SIZE or bytes(buffer[0:2]) != b"MD":
        raise ValueError("Not a CWA file: missing 'MD' header.")

    raw = bytes(buffer[0:SECTOR_SIZE])
    packet_length = int.from_bytes(raw[2:4], "little")

    hardware_type = raw[4]
    if hardware_type in (0x00, 0xFF, 0x17):
        device_type = "AX3"
    elif hardware_type == 0x64:
        device_type = "AX6"
    else:
        device_type = hex(hardware_type)

    device_id = int.from_bytes(raw[5:7], "little")
    device_id_upper = int.from_bytes(raw[11:13], "little")
    if device_id_upper != 0xFFFF:
        device_id |= device_id_upper << 16

    annotation = raw[64:512].decode("ascii", "ignore").rstrip("\x20\xff\x00")

    return {
        "device_type": device_type,
        "device_id": device_id,
        "session_id": int.from_bytes(raw[7:11], "little"),
        "firmware_revision": raw[41],
        "annotation": dict(parse_qsl(annotation, keep_blank_values=True)),
        "data_offset": (
            (packet_length + SECTOR_SIZE - 1) // SECTOR_SIZE
        ) * SECTOR_SIZE,
    }


def _unpack_samples(
    raw_sectors: np.ndarray,
    channels: int,
    bytes_per_axis: int,
) -> tuple[np.ndarray, int]:
    """
    Unpack the 480-byte sample payload of every sector.

    Returns an array of shape (sectors, samples_per_sector, channels) with
    raw integer values.
    """
    payload = np.ascontiguousarray(raw_sectors[:, 30:510])

    if bytes_per_axis == 0 and channels == 3:
        # Packed: eezzzzzz zzzzyyyy yyyyyyxx xxxxxxxx
        packed = payload.view("<u4").astype(np.int64)
        exponent = packed >> 30

        def axis(shift: int) -> np.ndarray:
            value = (((packed >> shift) & 0x3FF) ^ 0x200) - 0x200
            return value << exponent

        values = np.stack([axis(0), axis(10), axis(20)], axis=-1)
        return values.astype(float), 120

    if bytes_per_axis == 2 and channels > 0:
        samples_per_sector = 480 // (2 * channels)
        words = payload.view("<i2")[:, :samples_per_sector * channels]
        values = words.reshape(-1, samples_per_sector, channels)
        return values.astype(float), samples_per_sector

    raise ValueError(
        "Unsupported CWA sample format: "
        f"channels={channels}, bytes_per_axis={bytes_per_axis}"
    )


def _decode_timestamps(packed: np.ndarray) -> np.ndarray:
    """
    Decode packed RTC values (YYYYYYMM MMDDDDDh hhhhmmmm mmssssss).

    Returns naive local times in seconds since 1970-01-01, NaN when the
    value does not encode a valid date.
    """
    packed = packed.astype(np.int64)

    parts = pd.DataFrame({
        "year": ((packed >> 26) & 0x3F) + 2000,
        "month": (packed >> 22) & 0x0F,
        "day": (packed >> 17) & 0x1F,
        "hour": (packed >> 12) & 0x1F,
        "minute": (packed >> 6) & 0x3F,
        "second": packed & 0x3F,
    })

    times = pd.to_datetime(parts, errors="coerce")
    seconds = (
        times - pd.Timestamp("1970-01-01")
    ).dt.total_seconds().to_numpy(dtype=float)

    return seconds


def _sample_times_local_seconds(
    sectors: np.ndarray,
    sample_count: np.ndarray,
    frequency: float,
) -> np.ndarray:
    """Interpolate a local timestamp for every sample."""
    timestamps = _decode_timestamps(sectors["timestamp_packed"])
    timestamp_offset = sectors["timestamp_offset"].astype(np.int64)

    device_fractional = sectors["device_fractional"].astype(np.int64)
    has_fraction = (device_fractional & 0x8000) != 0
    fraction_65536 = np.where(
        has_fraction,
        (device_fractional & 0x7FFF) << 1,
        0,
    )

    # Undo the backwards-compatible shim: the stored offset was reduced by
    # the whole samples that the fractional second accounts for.
    timestamp_offset = timestamp_offset + (
        (fraction_65536 * int(frequency)) >> 16
    )
    timestamps = timestamps + fraction_65536 / 65536.0

    sector_start = np.concatenate(([0], np.cumsum(sample_count)[:-1]))
    anchor_index = (sector_start + timestamp_offset).astype(float)

    # Sectors dropped (invalid checksum) or a restarted recording break the
    # sample sequence; interpolating across that gap would stretch the
    # samples before it. Each contiguous run is timed on its own.
    sequence_id = sectors["sequence_id"].astype(np.int64)
    run_breaks = np.flatnonzero(np.diff(sequence_id) != 1) + 1
    run_bounds = np.concatenate(([0], run_breaks, [len(sectors)]))

    times = np.empty(int(sample_count.sum()), dtype=float)

    for run_start, run_end in zip(run_bounds[:-1], run_bounds[1:]):
        first_sample = int(sector_start[run_start])
        last_sample = int(sector_start[run_end - 1] + sample_count[run_end - 1])

        times[first_sample:last_sample] = _interpolate_run(
            np.arange(first_sample, last_sample, dtype=float),
            anchor_index[run_start:run_end],
            timestamps[run_start:run_end],
            frequency,
        )

    return times


def _interpolate_run(
    sample_index: np.ndarray,
    anchor_index: np.ndarray,
    anchor_time: np.ndarray,
    frequency: float,
) -> np.ndarray:
    """Interpolate sample times within one contiguous run of sectors."""
    usable = np.isfinite(anchor_time)
    anchor_index = anchor_index[usable]
    anchor_time = anchor_time[usable]

    if anchor_index.size == 0:
        raise ValueError("CWA file contains a run of sectors without valid timestamps.")

    # Keep anchors that move forward in both sample index and time.
    order = np.argsort(anchor_index, kind="stable")
    anchor_index = anchor_index[order]
    anchor_time = anchor_time[order]
    forward = np.concatenate((
        [True],
        (np.diff(anchor_index) > 0) & (np.diff(anchor_time) > 0),
    ))
    anchor_index = anchor_index[forward]
    anchor_time = anchor_time[forward]

    if anchor_index.size == 1:
        return anchor_time[0] + (sample_index - anchor_index[0]) / frequency

    times = np.interp(sample_index, anchor_index, anchor_time)

    # np.interp clamps outside the anchors: extrapolate with the rate
    # observed between the first and last pairs of anchors.
    head_period = (anchor_time[1] - anchor_time[0]) / (anchor_index[1] - anchor_index[0])
    tail_period = (anchor_time[-1] - anchor_time[-2]) / (anchor_index[-1] - anchor_index[-2])

    before = sample_index < anchor_index[0]
    after = sample_index > anchor_index[-1]
    times[before] = anchor_time[0] - (anchor_index[0] - sample_index[before]) * head_period
    times[after] = anchor_time[-1] + (sample_index[after] - anchor_index[-1]) * tail_period

    return times


def _utc_offset_seconds(
    first_local_seconds: float,
    timezone: str,
    utc_offset_minutes: Optional[float],  # noqa: UP045
) -> float:
    """Offset of the device clock from UTC, in seconds."""
    if utc_offset_minutes is not None:
        return float(utc_offset_minutes) * 60.0

    local = datetime(1970, 1, 1) + timedelta(seconds=float(first_local_seconds))
    offset = local.replace(tzinfo=ZoneInfo(timezone)).utcoffset()

    return offset.total_seconds() if offset is not None else 0.0
