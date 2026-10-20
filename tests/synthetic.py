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


def _dotnet_ticks(local: datetime) -> int:
    return int((local - datetime(1, 1, 1)).total_seconds()) * 10_000_000


def _pack_12bit_yxz(counts_xyz: np.ndarray) -> bytes:
    """Pack samples as a big-endian stream of 12-bit y, x, z values."""
    values = counts_xyz[:, [1, 0, 2]].astype(np.int64).ravel() & 0xFFF
    bits = ((values[:, None] >> np.arange(11, -1, -1)) & 1).astype(np.uint8).ravel()
    return np.packbits(bits).tobytes()


def write_gt3x(
    path: Path,
    start_local: datetime,
    xyz_g: np.ndarray,
    rate: int = 30,
    scale: float = 256.0,
    idle_seconds: tuple[int, ...] = (),
    record_type: int = 0x1A,
    layout: str = "log.bin",
    timezone_offset: str = "01:00:00",
    trailing_idle_s: int = 0,
) -> None:
    """
    Write an ActiGraph .gt3x zip. ``xyz_g`` holds the samples of the
    recorded seconds only; ``idle_seconds`` are seconds without a record
    (idle sleep), bracketed by idle-sleep events.
    """
    import zipfile

    counts = np.round(xyz_g * scale).astype(np.int64)
    n_recorded = counts.shape[0] // rate
    total_seconds = n_recorded + len(idle_seconds)
    last_local = start_local + timedelta(seconds=total_seconds + trailing_idle_s)
    info = "\n".join([
        "Serial Number: TAS1H00000001",
        "Device Type: wGT3XBT",
        "Firmware: 1.9.2",
        f"Sample Rate: {rate}",
        f"Start Date: {_dotnet_ticks(start_local)}",
        f"Stop Date: {_dotnet_ticks(last_local)}",
        f"Last Sample Time: {_dotnet_ticks(last_local)}",
        f"TimeZone: {timezone_offset}",
        f"Acceleration Scale: {scale}",
        "Acceleration Min: -8.0",
        "Acceleration Max: 8.0",
    ]) + "\n"

    def record(kind: int, local_s: int, payload: bytes) -> bytes:
        head = struct.pack("<BBIH", 0x1E, kind, local_s, len(payload))
        checksum = 0xFF
        for byte in head + payload:
            checksum ^= byte
        return head + payload + bytes([(~checksum) & 0xFF])

    start_s = int((start_local - datetime(1970, 1, 1)).total_seconds())
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("info.txt", info)
        if layout == "activity.bin":
            archive.writestr("activity.bin", _pack_12bit_yxz(counts))
            return
        log = bytearray()
        k = 0
        idle = set(idle_seconds)
        for second in range(total_seconds):
            t = start_s + second
            if second in idle:
                if second - 1 not in idle:
                    log += record(0x03, t, b"\x08")
                continue
            if second - 1 in idle:
                log += record(0x03, t, b"\x09")
            chunk = counts[k * rate:(k + 1) * rate]
            k += 1
            if record_type == 0x1A:
                payload = chunk.astype("<i2").tobytes()
            else:
                payload = _pack_12bit_yxz(chunk)
            log += record(record_type, t, payload)
        log += record(0x1A, start_s + total_seconds, b"Z")  # USB marker, no samples
        archive.writestr("log.bin", bytes(log))


def write_geneactiv_bin(
    path: Path,
    start_local: datetime,
    xyz_g: np.ndarray,
    rate: float = 100.0,
    gains=(25875, 25734, 25538),
    offsets=(439, -662, -3056),
    time_zone: str = "GMT +01:00",
    truncate_last_page: bool = False,
) -> None:
    """Write a GENEActiv .bin text file with pages of 300 samples."""
    n_pages = xyz_g.shape[0] // 300
    header = [
        "Device Identity",
        "Device Unique Serial Code:045678",
        "Device Type:GENEActiv",
        "",
        "Device Capabilities",
        "Accelerometer Range:-8 to 8",
        "",
        "Configuration Info",
        f"Measurement Frequency:{rate:g} Hz",
        f"Start Time:{start_local:%Y-%m-%d %H:%M:%S}:000",
        f"Time Zone:{time_zone}",
        "",
        "Subject Info",
        "Device Location Code:left wrist",
        "Subject Notes:" + "\x00" * 10,
        "",
        "Calibration Data",
        f"x gain:{gains[0]}",
        f"x offset:{offsets[0]}",
        f"y gain:{gains[1]}",
        f"y offset:{offsets[1]}",
        f"z gain:{gains[2]}",
        f"z offset:{offsets[2]}",
        "Volts:300",
        "Lux:800",
        "",
        "Memory Status",
        f"Number of Pages:{n_pages}",
        "",
    ]
    raw = np.column_stack([
        np.round((xyz_g[:, axis] * gains[axis] + offsets[axis]) / 100.0) for axis in range(3)
    ]).astype(np.int64)
    light = np.arange(xyz_g.shape[0]) % 1024
    button = (np.arange(xyz_g.shape[0]) // 300) % 2
    lines = list(header)
    for p in range(n_pages):
        page_time = start_local + timedelta(seconds=p * 300 / rate)
        rows = raw[p * 300:(p + 1) * 300] & 0xFFF
        tail = (light[p * 300:(p + 1) * 300] << 2) | (button[p * 300:(p + 1) * 300] << 1)
        hexline = "".join(
            f"{x:03X}{y:03X}{z:03X}{t:03X}" for (x, y, z), t in zip(rows, tail)
        )
        if truncate_last_page and p == n_pages - 1:
            hexline = hexline[:1000]
        lines += [
            "Recorded Data",
            "Device Unique Serial Code:045678",
            f"Sequence Number:{p}",
            f"Page Time:{page_time:%Y-%m-%d %H:%M:%S}:{page_time.microsecond // 1000:03d}",
            "Unassigned:",
            "Temperature:24.5",
            "Battery voltage:4.1",
            "Device Status:Recording",
            f"Measurement Frequency:{rate:g}",
            hexline,
        ]
    path.write_bytes("\r\n".join(lines).encode("latin-1"))


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
