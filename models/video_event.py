from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VideoEvent:
    """
    Represents a synchronization event identified in the video.

    The timestamp is the selected video presentation position, expressed
    in seconds from the beginning of the video. The frame index is optional
    because the synchronization workflow must not reconstruct timestamps
    from the nominal frame rate.
    """

    event_id: str
    timestamp_seconds: float
    selection_method: str
    frame_index: Optional[int] = None  # noqa: UP045

    @property
    def timestamp_milliseconds(self) -> float:
        """Return the video timestamp in milliseconds."""
        return self.timestamp_seconds * 1000.0