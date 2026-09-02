from pathlib import Path

import numpy as np
import pandas as pd
from mad_gui import BaseImporter, start_gui
from mad_gui.plot_tools.labels.base_label import BaseRegionLabel


class ActivityLabel(BaseRegionLabel):
    name = "SENS motion (acceleration)"
    min_height = 0
    max_height = 1
    descriptions = {f"Activity {i}": None for i in range(1, 20)} #noQA


class SensImporter(BaseImporter):
    loadable_file_type = "*.csv"

    @classmethod
    def name(cls) -> str:
        return "SENS file"

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.df = None

    def load_sensor_data(self, file_path: str) -> dict:
        path_csv = Path(file_path)
        
        df = pd.read_csv(path_csv)
        df.columns = df.columns.str.strip().str.lower()
        self.df = df.copy()

        vm_values = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
        sensor_data = pd.DataFrame({
            "x": df['x'],
            "y": df['y'],
            "z": df['z'],
            "Vector Magnitude": vm_values
            })
        sensor_data = sensor_data.reset_index(drop=True)
        
        mean_diff_ms = df['unixts'].diff().mean()
        sampling_rate_hz = 1000.0 / mean_diff_ms

        data = {
           "SENS motion (acceleration)": {
               "sensor_data": sensor_data,
               "sampling_rate_hz": sampling_rate_hz,
           }
        }
        return data

    def load_annotations(self, file_path: str) -> dict:
        marker_path = Path(file_path)
        annotations = []
        
        if marker_path.exists():
            marker_df = pd.read_csv(marker_path, sep=';', skipinitialspace=True)
            marker_df.columns = marker_df.columns.str.strip()

            marker_df['Phone timestamp'] = pd.to_datetime(marker_df['Phone timestamp']).dt.tz_localize(None)
            
            time_col = 'local' if 'local' in self.df.columns else 'utc'
            acc_times = pd.to_datetime(self.df[time_col]).dt.tz_localize(None)

            start_time = None
            activity_count = 1

            for _, row in marker_df.iterrows():
                timestamp = row['Phone timestamp']
                marker_type = row['Marker start/stop'].strip()
                
                if "START" in marker_type:
                    start_time = timestamp
                elif "STOP" in marker_type and start_time is not None:
                    start_idx = int((acc_times - start_time).abs().argsort().iloc[0])
                    stop_idx = int((acc_times - timestamp).abs().argsort().iloc[0])

                    annotations.append({
                        "identifier": activity_count - 1,
                        "start": start_idx,
                        "end": stop_idx,
                        "description": f"Activity {activity_count}"
                    })
                    activity_count += 1
                    start_time = None
                    
            if annotations:
                ann_df = pd.DataFrame(annotations)
                ann_df['start'] = ann_df['start'].astype(int)
                ann_df['end'] = ann_df['end'].astype(int)
                ann_df['identifier'] = range(len(ann_df))
            else:
                ann_df = pd.DataFrame(columns=["identifier", "start", "end", "description"])
            
            ann_df = ann_df[["identifier", "start", "end", "description"]]
            
            # ESTRUCTURA CORRECTA: 
            # Clave del plot -> Diccionario de anotaciones donde la clave es el 'name' de la etiqueta
            return {
                "SENS motion (acceleration)": {
                    "SENS motion (acceleration)": ann_df
                }
            }
        else:
            print(f"Marker file not found: {marker_path}")
            return {}


def main():
    start_gui(
        plugins=[SensImporter],
        labels=[ActivityLabel]
    )


if __name__ == "__main__":
    main()