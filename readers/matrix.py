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


def _calc_sensor_val(value: int, sensor_range: int) -> float:
    denominator = 0x7FFF if value > 0 else 0x8000
    return float(value * sensor_range / denominator)


def read_matrix(file_path: Union[str, Path]) -> SensorRecording:  # noqa: UP007
    """
    Read a Parmay Matrix binary file (.bin).

    The device stores packages with a start and end time (Unix seconds);
    samples within a package are spread evenly between both times. Packages
    with an invalid CRC32 are skipped.
    """
    with open(file_path, "rb") as f:
        file_buf = bytearray(f.read())

    # 1. Remarks
    if len(file_buf) < REMARKS_SIZE:
        raise ValueError("El archivo binario de Matrix está incompleto o dañado.")
    file_buf = file_buf[REMARKS_SIZE:]

    # 2. General file header
    struct_file_header = struct.Struct("4sIHH")
    if len(file_buf) < struct_file_header.size:
        raise ValueError("No se pudo leer la cabecera principal del archivo Matrix.")

    header_recogni, header_packet_num, acc_range, _gyro_range = (
        struct_file_header.unpack_from(file_buf, 0)
    )
    file_buf = file_buf[struct_file_header.size:]

    if header_recogni.decode("utf-8", "ignore") != "MDTC":
        raise ValueError("Formato de archivo no válido: firma MDTC no encontrada.")

    # 3. Package structures
    struct_pkg_header = struct.Struct("8sIIIIIII")
    struct_acc = struct.Struct("hhh")
    struct_gyro = struct.Struct("hhh")
    struct_temp = struct.Struct("hh")
    struct_hr = struct.Struct("hh")

    timestamps: list[float] = []
    acc_rows: list[tuple[float, float, float]] = []
    n_crc_errors = 0

    # 4. Packages
    for _ in range(header_packet_num):
        try:
            pkg_offset = file_buf.index(PACKAGE_KEY)
        except ValueError:
            break
        file_buf = file_buf[pkg_offset:]

        if len(file_buf) < struct_pkg_header.size:
            break

        (
            rec_str,
            crc32,
            t_start,
            t_end,
            size_acc,
            size_gyro,
            size_temp,
            size_hr,
        ) = struct_pkg_header.unpack_from(file_buf, 0)

        payload_len = (
            size_acc * struct_acc.size
            + size_gyro * struct_gyro.size
            + size_temp * struct_temp.size
            + size_hr * struct_hr.size
        )

        pkg_total_len = struct_pkg_header.size + payload_len
        if len(file_buf) < pkg_total_len:
            break

        payload_buf = file_buf[struct_pkg_header.size:pkg_total_len]
        calculated_crc = binascii.crc32(file_buf[len(rec_str) + 4:pkg_total_len])
        file_buf = file_buf[pkg_total_len:]

        if calculated_crc != crc32:
            n_crc_errors += 1
            continue

        max_samples = max(size_acc, size_gyro, size_temp, size_hr)
        if max_samples <= 0:
            continue

        acc_data = []
        offset = 0
        for _ in range(size_acc):
            x, y, z = struct_acc.unpack_from(payload_buf, offset)
            acc_data.append((
                _calc_sensor_val(x, acc_range),
                _calc_sensor_val(y, acc_range),
                _calc_sensor_val(z, acc_range),
            ))
            offset += struct_acc.size

        step_ms = ((t_end - t_start) * 1000.0) / max_samples
        acc_scale = max_samples / size_acc if size_acc > 0 else 0

        for i in range(max_samples):
            timestamps.append(t_start * 1000.0 + i * step_ms)

            acc_idx = int(i / acc_scale) if acc_scale > 0 else 0
            acc_idx = min(acc_idx, len(acc_data) - 1) if acc_data else 0

            acc_rows.append(acc_data[acc_idx] if acc_data else (0.0, 0.0, 0.0))

    if not acc_rows:
        raise ValueError("No se pudieron extraer registros válidos del archivo Matrix.")

    acc = np.asarray(acc_rows, dtype=float)
    timestamp_unix_ms = np.asarray(timestamps, dtype=float)

    # Median interval: robust to the gaps left by skipped packages.
    sampling_rate = (
        estimate_sampling_rate(timestamp_unix_ms) if len(timestamps) > 1 else 100.0
    )

    return SensorRecording(
        sensor_type="matrix",
        data=build_recording_frame(timestamp_unix_ms, acc[:, 0], acc[:, 1], acc[:, 2]),
        sampling_rate_hz=float(sampling_rate),
        metadata={
            "accel_range_g": acc_range,
            "n_packages_declared": header_packet_num,
            "n_crc_errors": n_crc_errors,
        },
    )
