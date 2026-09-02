import numpy as np

from models import SyncAnchorPair


class SyncModel:
    """
    Estimate the temporal relationship between sensor and video timelines.

    The model is:

        video_time = intercept + slope * sensor_relative_time

    where sensor_relative_time is measured in seconds from the first
    synchronization anchor.

    The slope represents clock-rate differences between the sensor
    timeline and the video timeline.
    """

    def fit(self, anchors: list[SyncAnchorPair]) -> dict:
        """
        Fit an affine synchronization model.

        Parameters
        ----------
        anchors:
            Synchronization correspondences between sensor and video.

        Returns
        -------
        dict
            Estimated model parameters and quality-control metrics.
        """

        if len(anchors) < 2:
            raise ValueError(
                "At least two synchronization anchors are required "
                "to estimate offset and drift."
            )

        anchors = sorted(
            anchors,
            key=lambda anchor: anchor.sensor_timestamp_unix_ms,
        )

        sensor_time = np.array(
            [
                anchor.sensor_timestamp_unix_ms / 1000.0
                for anchor in anchors
            ],
            dtype=float,
        )

        video_time = np.array(
            [
                anchor.video_timestamp_seconds
                for anchor in anchors
            ],
            dtype=float,
        )

        sensor_origin = sensor_time[0]
        sensor_relative_time = sensor_time - sensor_origin

        coefficients = np.polyfit(
            sensor_relative_time,
            video_time,
            deg=1,
        )

        slope = float(coefficients[0])
        intercept = float(coefficients[1])

        predicted_video_time = (
            intercept + slope * sensor_relative_time
        )

        residuals = video_time - predicted_video_time

        rmse_seconds = float(
            np.sqrt(np.mean(residuals ** 2))
        )

        drift_ppm = (slope - 1.0) * 1_000_000.0

        return {
            "sensor_origin_unix_seconds": float(sensor_origin),
            "intercept_seconds": intercept,
            "slope": slope,
            "drift_ppm": drift_ppm,
            "rmse_seconds": rmse_seconds,
            "residuals_seconds": residuals.tolist(),
            "n_anchors": len(anchors),
            "anchors": anchors,
        }