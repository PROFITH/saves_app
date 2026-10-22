"""
Reader for Parmay Matrix binary files (.bin).

The file starts with 512 bytes of remarks and an ``MDTC`` header (number
of packages, accelerometer range in g, gyroscope range in dps). Each
``MDTCPACK`` package carries a CRC32, a start and an end time (Unix
seconds) and four blocks of little-endian int16 samples: accelerometer
(x, y, z), gyroscope (x, y, z), temperature (body surface and ambient, in
tenths of a degree) and heart rate (``hr_raw`` and ``hr`` in bpm, 0 when
the firmware has no estimate). The block layout and the meaning of the
temperature and heart-rate pairs follow GGIRread's ``readParmayMatrix``.

Samples within a package are spread evenly between its start and end
times. The blocks with fewer samples than the accelerometer (temperature
and heart rate, about one per second) are repeated on the accelerometer
grid of the package. Packages with an invalid CRC32 are skipped.
"""

import binascii
import struct
from pathlib import Path
from typing import Union

import numpy as np

from readers.recording import (
    SensorRecording,
    build_recording_frame,
    estimate_sampling_rate,
)

REMARKS_SIZE = 512
PACKAGE_KEY = b"MDTCPACK"
# The heart rate is estimated by the device firmware from its optical (PPG) sensor.
HR_SOURCE = "matrix_ppg_firmware"

_FILE_HEADER = struct.Struct("<4sIHH")
_PACKAGE_HEADER = struct.Struct("<8sIIIIIII")
# Blocks of a package in order, with the int16 values of one sample.
_BLOCKS = (("acc", 3), ("gyro", 3), ("temp", 2), ("hr", 2))


def _calc_sensor_val(value: int, sensor_range: int) -> float:
    denominator = 0x7FFF if value > 0 else 0x8000
    return float(value * sensor_range / denominator)


def _scale(values: np.ndarray, sensor_range: float) -> np.ndarray:
    """int16 counts to physical units: positive counts span 0x7FFF, negative ones 0x8000."""
    values = np.asarray(values, dtype=float)
    return np.where(values > 0, values * sensor_range / 0x7FFF, values * sensor_range / 0x8000)


def _spread(block: np.ndarray, n_samples: int) -> np.ndarray:
    """Rows of ``block`` repeated on a grid of ``n_samples`` (NaN when the block is empty)."""
    if block.shape[0] == 0:
        return np.full((n_samples, block.shape[1]), np.nan)
    index = (np.arange(n_samples) * block.shape[0]) // n_samples
    return block[index]


def read_matrix(file_path: Union[str, Path]) -> SensorRecording:  # noqa: UP007
    """
    Read a Parmay Matrix binary file (.bin).

    Returns
    -------
    SensorRecording
        ``timestamp_unix_ms`` and acceleration in g on the package grid,
        plus every block the file holds: ``gyro_x/y/z`` (degrees per
        second), ``temperature_c`` (body surface, °C) and
        ``temperature_ambient_c`` (°C), ``hr_bpm`` (the firmware's heart
        rate from its optical sensor, NaN when it gives none) and
        ``hr_raw`` (the raw value stored next to it, without a documented
        unit). ``metadata["hr_source"]`` names the origin of ``hr_bpm``.
    """
    with open(file_path, "rb") as f:
        file_buf = f.read()

    # 1. Remarks and general file header
    if len(file_buf) < REMARKS_SIZE + _FILE_HEADER.size:
        raise ValueError("El archivo binario de Matrix está incompleto o dañado.")

    recognition, n_packages_declared, acc_range, gyro_range = _FILE_HEADER.unpack_from(file_buf, REMARKS_SIZE)
    if recognition != b"MDTC":
        raise ValueError("Formato de archivo no válido: firma MDTC no encontrada.")

    # 2. Packages
    timestamps: list[np.ndarray] = []
    blocks: dict[str, list[np.ndarray]] = {name: [] for name, _ in _BLOCKS}
    present = dict.fromkeys(blocks, False)
    n_crc_errors = 0
    position = REMARKS_SIZE + _FILE_HEADER.size

    for _ in range(n_packages_declared):
        position = file_buf.find(PACKAGE_KEY, position)
        if position < 0 or position + _PACKAGE_HEADER.size > len(file_buf):
            break

        _key, crc32, t_start, t_end, *sizes = _PACKAGE_HEADER.unpack_from(file_buf, position)
        payload_start = position + _PACKAGE_HEADER.size
        payload_end = payload_start + sum(n * 2 * width for n, (_, width) in zip(sizes, _BLOCKS))
        if payload_end > len(file_buf):
            break

        package = file_buf[position:payload_end]
        position = payload_end
        if binascii.crc32(package[len(PACKAGE_KEY) + 4:]) != crc32:
            n_crc_errors += 1
            continue

        n_samples = max(sizes)
        if n_samples <= 0:
            continue

        raw = {}
        offset = payload_start
        for count, (name, width) in zip(sizes, _BLOCKS):
            raw[name] = np.frombuffer(file_buf, dtype="<i2", count=count * width, offset=offset).reshape(count, width)
            offset += count * 2 * width
            present[name] |= count > 0

        step_ms = (t_end - t_start) * 1000.0 / n_samples
        timestamps.append(t_start * 1000.0 + np.arange(n_samples) * step_ms)
        acc = _spread(_scale(raw["acc"], acc_range), n_samples)
        blocks["acc"].append(acc if raw["acc"].shape[0] else np.zeros((n_samples, 3)))
        blocks["gyro"].append(_spread(_scale(raw["gyro"], gyro_range), n_samples))
        blocks["temp"].append(_spread(raw["temp"].astype(float) / 10.0, n_samples))
        blocks["hr"].append(_spread(raw["hr"].astype(float), n_samples))

    if not timestamps:
        raise ValueError("No se pudieron extraer registros válidos del archivo Matrix.")

    timestamp_unix_ms = np.concatenate(timestamps)
    acc = np.concatenate(blocks["acc"])

    extra = {}
    metadata = {
        "accel_range_g": acc_range,
        "gyro_range_dps": gyro_range,
        "n_packages_declared": n_packages_declared,
        "n_crc_errors": n_crc_errors,
    }
    if present["gyro"]:
        gyro = np.concatenate(blocks["gyro"])
        extra.update({"gyro_x": gyro[:, 0], "gyro_y": gyro[:, 1], "gyro_z": gyro[:, 2]})
    if present["temp"]:
        temperature = np.concatenate(blocks["temp"])
        extra["temperature_c"] = temperature[:, 0]
        extra["temperature_ambient_c"] = temperature[:, 1]
    if present["hr"]:
        heart = np.concatenate(blocks["hr"])
        extra["hr_bpm"] = np.where(heart[:, 1] > 0, heart[:, 1], np.nan)
        extra["hr_raw"] = heart[:, 0]
        metadata["hr_source"] = HR_SOURCE

    # Median interval: robust to the gaps left by skipped packages.
    sampling_rate = estimate_sampling_rate(timestamp_unix_ms) if timestamp_unix_ms.size > 1 else 100.0

    return SensorRecording(
        sensor_type="matrix",
        data=build_recording_frame(timestamp_unix_ms, acc[:, 0], acc[:, 1], acc[:, 2], extra=extra),
        sampling_rate_hz=float(sampling_rate),
        metadata=metadata,
    )
