from pathlib import Path

import numpy as np
import pandas as pd
from mad_gui import BaseImporter


class SensImporter(BaseImporter):
    """
    Importer for SENS accelerometer CSV files.

    The importer preserves the native sensor timestamps so that the
    synchronization pipeline can later relate sensor events to video time.
    """

    loadable_file_type = "*.csv"

    @classmethod
    def name(cls) -> str:
        """Return the human-readable importer name."""
        return "SENS file"

    def load_sensor_data(self, file_path: str) -> dict:
        """
        Load a SENS accelerometer CSV file.

        Parameters
        ----------
        file_path
            Path to the SENS CSV file.

        Returns
        -------
        dict
            Dictionary containing the accelerometer data and estimated
            sampling frequency.

        Notes
        -----
        The original ``unixts`` timestamps are preserved in the output
        dataframe as ``timestamp_unix_ms``. These timestamps are part of
        the sensor's native time domain and must not be replaced by a
        reconstructed time axis.
        """
        path_csv = Path(file_path)

        # ------------------------------------------------------------------
        # Load and normalize the input data
        # ------------------------------------------------------------------
        df = pd.read_csv(path_csv)
        df.columns = df.columns.str.strip().str.lower()

        # ------------------------------------------------------------------
        # Validate required columns
        # ------------------------------------------------------------------
        required_columns = {"unixts", "x", "y", "z"}
        missing_columns = required_columns.difference(df.columns)

        if missing_columns:
            raise ValueError(
                "SENS file is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        # ------------------------------------------------------------------
        # Preserve native timestamps
        # ------------------------------------------------------------------
        timestamp_unix_ms = pd.to_numeric(
            df["unixts"],
            errors="coerce",
        )

        if timestamp_unix_ms.isna().any():
            raise ValueError(
                "SENS file contains invalid values in the 'unixts' column."
            )

        # ------------------------------------------------------------------
        # Accelerometer vector magnitude
        # ------------------------------------------------------------------
        vm_values = np.sqrt(
            df["x"] ** 2
            + df["y"] ** 2
            + df["z"] ** 2
        )

        sensor_data = pd.DataFrame({
            "timestamp_unix_ms": timestamp_unix_ms,
            "x": df["x"],
            "y": df["y"],
            "z": df["z"],
            "Vector Magnitude": vm_values,
        }).reset_index(drop=True)

        # ------------------------------------------------------------------
        # Estimate sampling frequency from native timestamps
        # ------------------------------------------------------------------
        timestamp_diff_ms = timestamp_unix_ms.diff()

        median_diff_ms = timestamp_diff_ms.median()

        if not np.isfinite(median_diff_ms) or median_diff_ms <= 0:
            raise ValueError(
                "Unable to estimate the SENS sampling frequency from "
                "the 'unixts' timestamps."
            )

        sampling_rate_hz = 1000.0 / median_diff_ms

        return {
            "SENS motion (acceleration)": {
                "sensor_data": sensor_data,
                "sampling_rate_hz": sampling_rate_hz,
            }
        }