import struct
from pathlib import Path

import pandas as pd
from mad_gui import BaseImporter
from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication

from importers.matrix_importer import MatrixImporter
from importers.multi_device_dialog import MultiDeviceLoadDialog
from importers.sens_importer import SensImporter


class UniversalStudyImporter(BaseImporter):
    loadable_file_type = "*.*"

    @classmethod
    def name(cls) -> str:
        return "Carga Universal de Sensores (SENS, Matrix, N dispositivos)"

    def _get_sensor_t0(self, file_path: str, sensor_type: str) -> float:
        """Obtiene el timestamp inicial en ms de forma rápida."""
        if sensor_type == "SENS":
            df_head = pd.read_csv(file_path, nrows=2)
            df_head.columns = df_head.columns.str.strip().str.lower()
            return float(df_head["unixts"].iloc[0])
        elif sensor_type == "Matrix":
            with open(file_path, "rb") as f:
                f.seek(512 + 12)
                first_pkg = f.read(20)
                _, _, t_start_sec = struct.unpack_from("8sII", first_pkg, 0)
                return float(t_start_sec * 1000.0)
        return 0.0

    def _trigger_video_load(self, video_path: str):
        """Inyecta y reproduce el vídeo directamente en la ventana principal de mad-gui."""
        if not video_path or not Path(video_path).exists():
            return

        # Localizamos la instancia activa de MainWindow en la aplicación Qt
        app = QApplication.instance()
        main_win = None
        for widget in app.topLevelWidgets():
            if hasattr(widget, "load_video") and hasattr(widget, "global_data"):
                main_win = widget
                break

        if main_win:
            main_win.global_data.video_file = video_path
            main_win.load_video(video_path)

    def _trigger_session_setup(self, video_path: str, sync_path: str):
        app = QApplication.instance()
        main_win = None
        for widget in app.topLevelWidgets():
            if hasattr(widget, "load_video") and hasattr(widget, "global_data"):
                main_win = widget
                break

        if not main_win:
            return

        if video_path and Path(video_path).exists():
            main_win.global_data.video_file = video_path
            main_win.load_video(video_path)

        if sync_path and Path(sync_path).exists():
            main_win.global_data.sync_file = sync_path

    def load_sensor_data(self, file_path: str) -> dict:
        dialog = MultiDeviceLoadDialog()
        if dialog.exec_() != MultiDeviceLoadDialog.Accepted:
            raise ValueError("Operación de carga cancelada.")

        devices, time_window, video_path, sync_path = dialog.get_configured_data()
        if not devices:
            raise ValueError("No se especificó ningún archivo de dispositivo válido.")

        # Obtener referencia a MainWindow
        app = QApplication.instance()
        main_win = None
        for widget in app.topLevelWidgets():
            if hasattr(widget, "load_video") and hasattr(widget, "global_data"):
                main_win = widget
                break

        if main_win:
            print(f"[DEBUG combined_importer] Guardando en global_data -> video: '{video_path}', sync: '{sync_path}'")
            if video_path and Path(video_path).exists():
                main_win.global_data.video_file = video_path
                main_win.load_video(video_path)

            if sync_path and Path(sync_path).exists():
                # ASIGNAR ANTES DE DEVOLVER LOS DATOS
                main_win.global_data.sync_file = sync_path

        sens_loader = SensImporter()
        matrix_loader = MatrixImporter()

        data_dict = {}
        sens_idx = 1
        mat_idx = 1

        for dev in devices:
            dtype = dev["type"]
            dpath = dev["path"]

            if dtype == "SENS":
                res = sens_loader.load_sensor_data(dpath)
                key = (
                    f"SENS #{sens_idx}"
                    if len([d for d in devices if d["type"] == "SENS"]) > 1
                    else "SENS (acceleration)"
                )
                sens_idx += 1
            else:
                res = matrix_loader.load_sensor_data(dpath)
                key = (
                    f"Matrix #{mat_idx}"
                    if len([d for d in devices if d["type"] == "Matrix"]) > 1
                    else "Matrix (acceleration)"
                )
                mat_idx += 1

            raw_data = list(res.values())[0]
            df_sensor = raw_data["sensor_data"]
            fs = raw_data["sampling_rate_hz"]

            if time_window is not None:
                w_start_ms, w_end_ms = time_window
                t0_sensor_ms = self._get_sensor_t0(dpath, dtype)

                idx_start = max(0, int(((w_start_ms - t0_sensor_ms) / 1000.0) * fs))
                idx_end = min(len(df_sensor), int(((w_end_ms - t0_sensor_ms) / 1000.0) * fs))

                if idx_start < idx_end:
                    df_sensor = df_sensor.iloc[idx_start:idx_end].reset_index(drop=True)

            data_dict[key] = {
                "sensor_data": df_sensor,
                "sampling_rate_hz": fs,
            }

        return data_dict