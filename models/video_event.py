from dataclasses import dataclass


@dataclass(frozen=True)
class VideoEvent:
    """
    Represents a synchronization event identified in the video.

    The timestamp is the decoder-reported presentation timestamp of
    the selected video frame, expressed in seconds from the beginning
    of the video.
    """

    event_id: str
    frame_index: int
    timestamp_seconds: float
    selection_method: str

    @property
    def timestamp_milliseconds(self) -> float:
        """Return the video timestamp in milliseconds."""
        return self.timestamp_seconds * 1000.0