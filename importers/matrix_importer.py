import binascii
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from mad_gui import BaseImporter


class MatrixImporter(BaseImporter):
    loadable_file_type = "*.bin"

    @classmethod
    def name(cls) -> str:
        return "Parmay Matrix binary (*.bin)"

    @staticmethod
    def _calc_sensor_val(value: int, sensor_range: int) -> float:
        denominator = 0x7FFF if value > 0 else 0x8000
        return float(value * sensor_range / denominator)

    def _parse_binary(self, file_path: Path) -> tuple[pd.DataFrame, float]:
        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        file_buf = bytearray(raw_bytes)
        pkg_key = b"MDTCPACK"
        remarks_size = 512

        # 1. Extraer Remarks
        if len(file_buf) < remarks_size:
            raise ValueError("El archivo binario de Matrix está incompleto o dañado.")
        file_buf = file_buf[remarks_size:]

        # 2. Extraer cabecera general del archivo
        struct_file_header = struct.Struct("4sIHH")
        if len(file_buf) < struct_file_header.size:
            raise ValueError("No se pudo leer la cabecera principal del archivo Matrix.")

        header_recogni, header_packet_num, acc_range, _gyro_range = (
            struct_file_header.unpack_from(file_buf, 0)
        )
        file_buf = file_buf[struct_file_header.size :]

        if header_recogni.decode("utf-8", "ignore") != "MDTC":
            raise ValueError("Formato de archivo no válido: firma MDTC no encontrada.")

        # 3. Estructuras de desempaquetado por paquete
        struct_pkg_header = struct.Struct("8sIIIIIII")
        struct_acc = struct.Struct("hhh")
        struct_gyro = struct.Struct("hhh")
        struct_temp = struct.Struct("hh")
        struct_hr = struct.Struct("hh")

        records: list[dict[str, float]] = []
        timestamps: list[int] = []

        # 4. Procesar cada paquete de datos
        for _ in range(header_packet_num):
            try:
                pkg_offset = file_buf.index(pkg_key)
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

            acc_bytes = size_acc * struct_acc.size
            gyro_bytes = size_gyro * struct_gyro.size
            temp_bytes = size_temp * struct_temp.size
            hr_bytes = size_hr * struct_hr.size
            payload_len = acc_bytes + gyro_bytes + temp_bytes + hr_bytes

            pkg_total_len = struct_pkg_header.size + payload_len
            if len(file_buf) < pkg_total_len:
                break

            # Verificación de CRC32
            payload_buf = file_buf[struct_pkg_header.size : pkg_total_len]
            calculated_crc = binascii.crc32(file_buf[len(rec_str) + 4 : pkg_total_len])
            file_buf = file_buf[pkg_total_len:]

            if calculated_crc != crc32:
                continue

            max_samples = max(size_acc, size_gyro, size_temp, size_hr)
            if max_samples <= 0:
                continue

            # Desempaquetar señales
            acc_data = []
            offset = 0
            for _ in range(size_acc):
                x, y, z = struct_acc.unpack_from(payload_buf, offset)
                acc_data.append(
                    (
                        self._calc_sensor_val(x, acc_range),
                        self._calc_sensor_val(y, acc_range),
                        self._calc_sensor_val(z, acc_range),
                    )
                )
                offset += struct_acc.size

            # Mapear muestras del paquete
            step_ms = ((t_end - t_start) * 1000.0) / max_samples
            acc_scale = max_samples / size_acc if size_acc > 0 else 0

            for i in range(max_samples):
                t_calc = int(t_start * 1000 + i * step_ms)
                timestamps.append(t_calc)

                acc_idx = int(i / acc_scale) if acc_scale > 0 else 0
                acc_idx = min(acc_idx, len(acc_data) - 1) if acc_data else 0

                ax, ay, az = acc_data[acc_idx] if acc_data else (0.0, 0.0, 0.0)
                vm = np.sqrt(ax**2 + ay**2 + az**2)

                records.append({
                    "acc_x": ax,
                    "acc_y": ay,
                    "acc_z": az,
                    "Vector Magnitude": vm,
                })

        if not records:
            raise ValueError("No se pudieron extraer registros válidos del archivo Matrix.")

        df = pd.DataFrame(records)
        mean_dt_ms = pd.Series(timestamps).diff().mean()
        sampling_rate = 1000.0 / mean_dt_ms if mean_dt_ms and not np.isnan(mean_dt_ms) else 100.0

        return df, float(sampling_rate)

    def load_sensor_data(self, file_path: str) -> dict:
        path_bin = Path(file_path)
        df_sensor, fs = self._parse_binary(path_bin)

        return {
            "Parmay Matrix (acceleration)": {
                "sensor_data": df_sensor,
                "sampling_rate_hz": fs,
            }
        }