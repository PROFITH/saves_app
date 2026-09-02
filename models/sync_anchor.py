from dataclasses import dataclass


@dataclass(frozen=True)
class SyncAnchor:
    """
    Represents a detected synchronization anchor in sensor data.

    A synchronization anchor identifies the precise sensor timestamp
    corresponding to a synchronization event detected within a protocol
    synchronization window.

    The video timestamp is intentionally not included yet. It will be
    added when SAVES establishes the correspondence between the sensor
    event and the video timeline.
    """

    event_id: str
    event_type: str
    phase: str

    sensor_timestamp_unix_ms: int

    selection_rule: str
    prominence: float

    peak_value: float

    candidate_count: int

    @property
    def sensor_datetime(self):
        """
        Return the sensor timestamp as a UTC datetime.

        The conversion is provided as a convenience only. The original
        Unix timestamp remains the authoritative sensor time.
        """
        from datetime import datetime, timezone

        return datetime.fromtimestamp(
            self.sensor_timestamp_unix_ms / 1000.0,
            tz=timezone.utc,
        )