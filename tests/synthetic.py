"""Builders of synthetic sensor files and sessions for the tests."""

from __future__ import annotations

import binascii
import json
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SECTOR = 512


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def clap_signal(times_s: np.ndarray, clap_times_s: list[float], seed: int = 0) -> np.ndarray:
    """
    Static sensor (gravity on z) with sharp impacts at ``clap_times_s``.

    Returns an (N, 3) array in g.
    """
    rng = np.random.default_rng(seed)
    xyz = np.zeros((times_s.size, 3))
    xyz[:, 2] = 1.0
    xyz += rng.normal(0.0, 0.005, xyz.shape)

    for clap in clap_times_s:
        # 20-ms triangular impulse of 4 g on x.
        weight = np.clip(1.0 - np.abs(times_s - clap) / 0.010, 0.0, None)
        xyz[:, 0] += 4.0 * weight

    return xyz


# ---------------------------------------------------------------------------
# Axivity .cwa
# ---------------------------------------------------------------------------

def _pack_timestamp(moment: datetime) -> int:
    return (
        ((moment.year - 2000) & 0x3F) << 26
        | (moment.month & 0x0F) << 22
        | (moment.day & 0x1F) << 17
        | (moment.hour & 0x1F) << 12
        | (moment.minute & 0x3F) << 6
        | (moment.second & 0x3F)
    )


def _pack_dword(raw_xyz: np.ndarray) -> int:
    """Pack three integer values (1/256 g) as 10-bit mantissas + exponent."""
    peak = int(np.max(np.abs(raw_xyz)))
    exponent = 0
    while exponent < 3 and (peak >> exponent) > 511:
        exponent += 1

    value = exponent << 30
    for axis, raw in enumerate(raw_xyz):
        mantissa = int(raw) >> exponent
        mantissa = max(-512, min(511, mantissa))
        value |= (mantissa & 0x3FF) << (10 * axis)
    return value


def write_cwa(
    path: Path,
    start_local: datetime,
    values: np.ndarray,
    frequency: float = 100.0,
    true_frequency: float | None = None,
    packed: bool = True,
    channels: int = 3,
    fractional: bool = False,
    corrupt_sectors: tuple[int, ...] = (),
) -> np.ndarray:
    """
    Write a minimal but valid CWA file.

    ``values`` holds g for 3 channels or (gyro dps, accel g) for 6 channels.
    ``start_local`` is the naive local time of the first sample; samples are
    spaced at ``true_frequency`` (defaults to ``frequency``) so that clock
    drift between the nominal and actual rate can be simulated.

    Returns the naive local time of every sample in seconds since 1970.
    """
    true_frequency = true_frequency or frequency
    n = values.shape[0]
    epoch = datetime(1970, 1, 1)
    t0 = (start_local - epoch).total_seconds()
    sample_times = t0 + np.arange(n) / true_frequency

    rate_code = (1 << 6) | (15 - int(round(np.log2(3200 / frequency))))  # +-8 g

    if packed:
        assert channels == 3
        num_axes_bps = 0x30
        per_sector = 120
        accel_unit = 256
        scale_light = 0
    else:
        num_axes_bps = (channels << 4) | 0x02
        per_sector = 480 // (2 * channels)
        accel_unit = 2048  # scale bits = 3
        scale_light = (3 << 13) | ((2 << 10) if channels >= 6 else 0)  # gyro 2000 dps

    header = bytearray(1024)
    header[0:2] = b"MD"
    header[2:4] = struct.pack("<H", 1020)
    header[4] = 0x17 if packed else 0x64
    header[5:7] = struct.pack("<H", 4321)
    header[7:11] = struct.pack("<I", 99)
    header[11:13] = struct.pack("<H", 0)
    header[36] = rate_code
    annotation = b"_sc=R4H001&_p=wrist"
    header[64:64 + len(annotation)] = annotation

    sectors = []
    for k, start in enumerate(range(0, n, per_sector)):
        chunk = values[start:start + per_sector]
        count = chunk.shape[0]
        t_start = sample_times[start]

        if fractional:
            whole = np.floor(t_start)
            frac_65536 = int(round((t_start - whole) * 65536)) & 0xFFFE
            device_fractional = 0x8000 | (frac_65536 >> 1)
            stamp = epoch + timedelta(seconds=float(whole))
            timestamp_offset = -((frac_65536 * int(frequency)) >> 16)
        else:
            whole = np.ceil(t_start)
            device_fractional = 0
            stamp = epoch + timedelta(seconds=float(whole))
            timestamp_offset = int(round((whole - t_start) * true_frequency))

        sector = bytearray(SECTOR)
        struct.pack_into(
            "<HHHIIIHHBBBBhH",
            sector,
            0,
            0x5841,
            508,
            device_fractional,
            99,
            k,
            _pack_timestamp(stamp),
            scale_light,
            0,
            0,
            0,
            rate_code,
            num_axes_bps,
            timestamp_offset,
            count,
        )

        offset = 30
        if packed:
            for row in chunk:
                raw = np.round(row * accel_unit).astype(int)
                struct.pack_into("<I", sector, offset, _pack_dword(raw))
                offset += 4
        else:
            for row in chunk:
                if channels >= 6:
                    gyro = np.round(row[:3] * 32768.0 / 2000.0)
                    accel = np.round(row[3:6] * accel_unit)
                    ints = np.concatenate([gyro, accel])
                else:
                    ints = np.round(row * accel_unit)
                for value in ints.astype(int):
                    struct.pack_into("<h", sector, offset, int(value))
                    offset += 2

        words = struct.unpack("<255H", bytes(sector[:510]))
        struct.pack_into("<H", sector, 510, (-sum(words)) & 0xFFFF)

        if k in corrupt_sectors:
            sector[100] ^= 0xFF

        sectors.append(bytes(sector))

    path.write_bytes(bytes(header) + b"".join(sectors))
    return sample_times


# ---------------------------------------------------------------------------
# CSV formats
# ---------------------------------------------------------------------------

def write_sens_csv(path: Path, timestamp_unix_ms: np.ndarray, xyz: np.ndarray) -> None:
    pd.DataFrame({
        "unixts": np.round(timestamp_unix_ms).astype(np.int64),
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "z": xyz[:, 2],
    }).to_csv(path, index=False)


def write_polar_acc_csv(path: Path, timestamp_unix_ms: np.ndarray, xyz_g: np.ndarray) -> None:
    times = pd.to_datetime(timestamp_unix_ms, unit="ms", utc=True)
    pd.DataFrame({
        "timestamp_iso": times.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "participant_id": "R4H001",
        "activity_index": 1,
        "phase_name": "static_prep",
        "acc_x_mg": xyz_g[:, 0] * 1000.0,
        "acc_y_mg": xyz_g[:, 1] * 1000.0,
        "acc_z_mg": xyz_g[:, 2] * 1000.0,
    }).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Parmay Matrix .bin
# ---------------------------------------------------------------------------

def write_matrix_bin(
    path: Path,
    start_unix_s: int,
    xyz_g: np.ndarray,
    samples_per_package: int = 100,
    acc_range: int = 8,
    corrupt_packages: tuple[int, ...] = (),
) -> None:
    """Write a Matrix file with one-second packages of accelerometer data."""
    packages = []
    n_packages = int(np.ceil(xyz_g.shape[0] / samples_per_package))

    for k in range(n_packages):
        chunk = xyz_g[k * samples_per_package:(k + 1) * samples_per_package]
        raw = np.round(chunk / acc_range * 0x7FFF).astype(int)
        payload = b"".join(struct.pack("hhh", *row) for row in raw)

        t_start = start_unix_s + k
        t_end = t_start + 1
        body = struct.pack("IIIIII", t_start, t_end, len(chunk), 0, 0, 0) + payload
        crc = binascii.crc32(body)
        if k in corrupt_packages:
            crc ^= 0xFFFF
        packages.append(b"MDTCPACK" + struct.pack("I", crc) + body)

    header = b"MDTC" + struct.pack("IHH", n_packages, acc_range, 0)
    path.write_bytes(bytes(512) + header + b"".join(packages))


# ---------------------------------------------------------------------------
# Session manifest
# ---------------------------------------------------------------------------

def iso(unix_s: float) -> str:
    return (
        datetime.fromtimestamp(unix_s, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def write_manifest(
    path: Path,
    session_start_s: float,
    session_end_s: float,
    windows: list[tuple[str, str, float]],
    schema_version: str = "1.1",
) -> None:
    """
    Write a manifest with one activity and the given sync windows.

    ``windows`` holds (event_id, phase, window_start_unix_s); windows last
    10 s as in lab_protocol_timer.
    """
    start_key, end_key = (
        ("window_start_time_utc", "window_end_time_utc")
        if schema_version == "1.1"
        else ("window_start", "window_end")
    )

    manifest = {
        "schema_version": schema_version,
        "session_id": "session_test",
        "created_by": "tests",
        "session_start_time_utc": iso(session_start_s),
        "session_end_time_utc": iso(session_end_s),
        "participants": [
            {"participant_id": "R4H001", "redcap_event_name": "da_0__visita_inici_arm_1"},
        ],
        "activities": [
            {
                "activity_index": 1,
                "activity_start_time_utc": iso(session_start_s + 30),
                "activity_end_time_utc": iso(session_end_s - 30),
                "duration_seconds": session_end_s - session_start_s - 60,
            }
        ],
        "sync_windows": [
            {
                "event_id": event_id,
                "event_type": "manual_sync",
                "activity_index": 1,
                "phase": phase,
                start_key: iso(window_start),
                end_key: iso(window_start + 10),
            }
            for event_id, phase, window_start in windows
        ],
        "video": {"files": []},
        "sensors": {},
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
