from pathlib import Path

import numpy as np
import pandas as pd
from mad_gui import BaseImporter


class SensImporter(BaseImporter):
    loadable_file_type = "*.csv"

    @classmethod
    def name(cls) -> str:
        return "SENS file"

    def load_sensor_data(self, file_path: str) -> dict:
        path_csv = Path(file_path)
        
        # Leemos el archivo y normalizamos nombres de columnas
        df = pd.read_csv(path_csv)
        df.columns = df.columns.str.strip().str.lower()

        # Cálculo de la magnitud vectorial: sqrt(x^2 + y^2 + z^2)
        vm_values = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
        
        sensor_data = pd.DataFrame({
            "x": df['x'],
            "y": df['y'],
            "z": df['z'],
            "Vector Magnitude": vm_values
        }).reset_index(drop=True)
        
        # Frecuencia de muestreo en Hz estimada desde la diferencia de unixts (ms)
        mean_diff_ms = df['unixts'].diff().mean()
        sampling_rate_hz = 1000.0 / mean_diff_ms

        # Estructura que mad-gui necesita para graficar
        return {
            "SENS motion (acceleration)": {
                "sensor_data": sensor_data,
                "sampling_rate_hz": sampling_rate_hz,
            }
        }