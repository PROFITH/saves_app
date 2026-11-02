from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

TIMESTAMP_COLUMN = "timestamp_unix_ms"
MAGNITUDE_COLUMN = "Vector Magnitude"


@dataclass
class SensorRecording:
    """
    Raw samples of one sensor file in a GUI-independent representation.

    ``data`` always contains ``timestamp_unix_ms`` (native sensor time,
    converted to Unix milliseconds in UTC), the acceleration axes in g
    (``x``, ``y``, ``z``) and ``Vector Magnitude``. Readers may add further
    channels (e.g. gyroscope).
    """

    sensor_type: str
    data: pd.DataFrame
    sampling_rate_hz: float
    metadata: dict = field(default_factory=dict)

    @property
    def start_unix_ms(self) -> float:
        """Timestamp of the first sample in Unix milliseconds."""
        return float(self.data[TIMESTAMP_COLUMN].iloc[0])

    @property
    def end_unix_ms(self) -> float:
        """Timestamp of the last sample in Unix milliseconds."""
        return float(self.data[TIMESTAMP_COLUMN].iloc[-1])


def build_recording_frame(
    timestamp_unix_ms: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    extra: Optional[dict] = None,  # noqa: UP045
) -> pd.DataFrame:
    """Assemble the standard sample dataframe used by every reader."""
    frame = pd.DataFrame({
        TIMESTAMP_COLUMN: np.asarray(timestamp_unix_ms, dtype=float),
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y, dtype=float),
        "z": np.asarray(z, dtype=float),
    })

    for name, values in (extra or {}).items():
        frame[name] = np.asarray(values, dtype=float)

    frame[MAGNITUDE_COLUMN] = np.sqrt(
        frame["x"] ** 2 + frame["y"] ** 2 + frame["z"] ** 2
    )

    return frame


def estimate_sampling_rate(timestamp_unix_ms: np.ndarray) -> float:
    """Estimate the sampling rate from native timestamps (median interval)."""
    diffs = np.diff(np.asarray(timestamp_unix_ms, dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]

    if diffs.size == 0:
        raise ValueError(
            "Unable to estimate the sampling rate from the timestamps."
        )

    return 1000.0 / float(np.median(diffs))
