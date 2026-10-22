from pathlib import Path

import pandas as pd
from mad_gui import BaseImporter

from readers.matrix import read_matrix


class MatrixImporter(BaseImporter):
    loadable_file_type = "*.bin"

    @classmethod
    def name(cls) -> str:
        return "Parmay Matrix binary (*.bin)"

    def _parse_binary(self, file_path: Path) -> tuple[pd.DataFrame, float]:
        # Parsing lives in readers.matrix so it can run without the GUI.
        recording = read_matrix(file_path)

        df = pd.DataFrame({
            "acc_x": recording.data["x"],
            "acc_y": recording.data["y"],
            "acc_z": recording.data["z"],
            "Vector Magnitude": recording.data["Vector Magnitude"],
        })

        return df, recording.sampling_rate_hz

    def load_sensor_data(self, file_path: str) -> dict:
        path_bin = Path(file_path)
        df_sensor, fs = self._parse_binary(path_bin)

        return {
            "Parmay Matrix (acceleration)": {
                "sensor_data": df_sensor,
                "sampling_rate_hz": fs,
            }
        }
