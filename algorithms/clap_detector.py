import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class ClapDetector:
    """
    Detect impulsive acceleration events in a synchronization window.

    The detector does not assume how many claps are present or where they
    should occur. It identifies candidate peaks in the acceleration
    magnitude using their prominence relative to the surrounding signal.
    """

    def detect(
        self,
        sensor_data: pd.DataFrame,
        timestamp_column: str = "timestamp_unix_ms",
        signal_column: str = "Vector Magnitude",
    ) -> dict:
        """
        Detect candidate impulsive events.

        Parameters
        ----------
        sensor_data
            Accelerometer samples from one synchronization window.

        timestamp_column
            Column containing native Unix timestamps in milliseconds.

        signal_column
            Column containing acceleration magnitude.

        Returns
        -------
        dict
            Detection results including all candidate peaks.

        Raises
        ------
        ValueError
            If required columns are missing or the input is empty.
        """
        self._validate_input(
            sensor_data,
            timestamp_column,
            signal_column,
        )

        signal = pd.to_numeric(
            sensor_data[signal_column],
            errors="coerce",
        ).to_numpy(dtype=float)

        timestamps = pd.to_numeric(
            sensor_data[timestamp_column],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = np.isfinite(signal) & np.isfinite(timestamps)

        signal = signal[valid]
        timestamps = timestamps[valid]

        if len(signal) < 3:
            return {
                "detected": False,
                "candidates": [],
            }

        # --------------------------------------------------------------
        # Estimate the typical temporal spacing between samples.
        # --------------------------------------------------------------
        timestamp_diffs = np.diff(timestamps)

        median_dt_ms = np.median(timestamp_diffs)

        if not np.isfinite(median_dt_ms) or median_dt_ms <= 0:
            raise ValueError(
                "Unable to determine the sensor sampling interval."
            )

        # --------------------------------------------------------------
        # Robust signal scale.
        # --------------------------------------------------------------
        median = np.median(signal)

        mad = np.median(
            np.abs(signal - median)
        )

        robust_sd = 1.4826 * mad

        # --------------------------------------------------------------
        # Candidate peak detection.
        #
        # We use prominence rather than an absolute acceleration
        # threshold. This makes the detector respond to local impulses
        # rather than simply selecting the largest value.
        # --------------------------------------------------------------
        if robust_sd > 0:
            min_prominence = 3.0 * robust_sd
        else:
            min_prominence = 0.05

        peaks, properties = find_peaks(
            signal,
            prominence=min_prominence,
        )

        candidates = []

        for i, peak_index in enumerate(peaks):
            candidates.append(
                {
                    "timestamp_unix_ms": int(
                        timestamps[peak_index]
                    ),
                    "peak_value": float(
                        signal[peak_index]
                    ),
                    "prominence": float(
                        properties["prominences"][i]
                    ),
                    "sample_index": int(peak_index),
                }
            )

        # Strongest candidate first.
        candidates.sort(
            key=lambda candidate: candidate["prominence"],
            reverse=True,
        )

        return {
            "detected": len(candidates) > 0,
            "candidates": candidates,
            "median_sample_interval_ms": float(
                median_dt_ms
            ),
            "median_signal": float(median),
            "robust_sd": float(robust_sd),
            "min_prominence": float(min_prominence),
        }

    @staticmethod
    def _validate_input(
        sensor_data: pd.DataFrame,
        timestamp_column: str,
        signal_column: str,
    ) -> None:
        """Validate the dataframe required by the detector."""

        if sensor_data.empty:
            raise ValueError(
                "Cannot detect events in an empty sensor window."
            )

        required_columns = {
            timestamp_column,
            signal_column,
        }

        missing_columns = required_columns.difference(
            sensor_data.columns
        )

        if missing_columns:
            raise ValueError(
                "Sensor data is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )