from typing import Optional

from models import VideoEvent


class VideoEventCapture:
    """
    Capture synchronization events from the full MaD-GUI video timeline.

    The caller supplies the current video position reported by the video player,
    in milliseconds from the beginning of the video. No timestamp is derived
    from the nominal video frame rate.
    """

    def capture(
        self,
        event_id: str,
        position_milliseconds: int,
        selection_method: str = "mad_gui_manual",
        frame_index: Optional[int] = None,  # noqa: UP045
    ) -> VideoEvent:
        """Create a VideoEvent from the player's current position."""
        if not event_id:
            raise ValueError("event_id must not be empty")

        if position_milliseconds < 0:
            raise ValueError("position_milliseconds must be non-negative")

        if not selection_method:
            raise ValueError("selection_method must not be empty")

        return VideoEvent(
            event_id=event_id,
            timestamp_seconds=position_milliseconds / 1000.0,
            selection_method=selection_method,
            frame_index=frame_index,
        )

    def capture_from_seconds(
        self,
        event_id: str,
        position_seconds: float,
        selection_method: str = "mad_gui_manual",
        frame_index: Optional[int] = None,  # noqa: UP045
    ) -> VideoEvent:
        """Create a VideoEvent from a video position expressed in seconds."""
        if position_seconds < 0:
            raise ValueError("position_seconds must be non-negative")

        position_milliseconds = round(position_seconds * 1000.0)

        return self.capture(
            event_id=event_id,
            position_milliseconds=position_milliseconds,
            selection_method=selection_method,
            frame_index=frame_index,
        )