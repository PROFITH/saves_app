import pandas as pd

from models import Session, SyncWindow


class SyncWindowExtractor:
    """
    Extract sensor samples corresponding to synchronization windows.

    Synchronization windows are defined by UTC timestamps in the session
    manifest, while sensor samples retain their native Unix timestamps
    in milliseconds.
    """

    def extract(
        self,
        session: Session,
        sensor_data: pd.DataFrame,
        timestamp_column: str = "timestamp_unix_ms",
    ) -> dict:
        """
        Extract samples falling inside each synchronization window.

        Parameters
        ----------
        session
            Parsed SAVES session.

        sensor_data
            Sensor dataframe containing native timestamps.

        timestamp_column
            Name of the column containing Unix timestamps in milliseconds.

        Returns
        -------
        dict
            Dictionary mapping each synchronization event ID to the
            corresponding subset of sensor samples.

        Raises
        ------
        ValueError
            If the timestamp column is missing or contains invalid values.
        """
        if timestamp_column not in sensor_data.columns:
            raise ValueError(
                f"Sensor data does not contain the required timestamp "
                f"column: '{timestamp_column}'"
            )

        timestamps = pd.to_numeric(
            sensor_data[timestamp_column],
            errors="coerce",
        )

        if timestamps.isna().any():
            raise ValueError(
                f"Sensor data contains invalid values in "
                f"'{timestamp_column}'."
            )

        sensor_data = sensor_data.copy()
        sensor_data[timestamp_column] = timestamps

        extracted_windows = {}

        for window in session.sync_windows:
            extracted_windows[window.event_id] = self._extract_window(
                sensor_data,
                window,
                timestamp_column,
            )

        return extracted_windows

    @staticmethod
    def _extract_window(
        sensor_data: pd.DataFrame,
        window: SyncWindow,
        timestamp_column: str,
    ) -> pd.DataFrame:
        """
        Extract samples from one synchronization window.
        """

        # Convert manifest UTC timestamps to Unix milliseconds.
        window_start_ms = window.window_start.timestamp() * 1000.0
        window_end_ms = window.window_end.timestamp() * 1000.0

        mask = (
            (sensor_data[timestamp_column] >= window_start_ms)
            & (sensor_data[timestamp_column] <= window_end_ms)
        )

        return sensor_data.loc[mask].copy().reset_index(drop=True)