from dataclasses import dataclass


@dataclass(frozen=True)
class SyncAnchorPair:
    """
    Represents a synchronization correspondence between sensor and video.

    The sensor timestamp is expressed in native Unix milliseconds.
    The video timestamp is expressed in seconds from the beginning
    of the video.
    """

    event_id: str

    sensor_timestamp_unix_ms: int
    video_timestamp_seconds: float

    sensor_selection_rule: str
    video_selection_method: str

    @property
    def sensor_timestamp_seconds(self) -> float:
        """Return the sensor timestamp in Unix seconds."""
        return self.sensor_timestamp_unix_ms / 1000.0