from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SyncWindow:
    """Defines a time interval in which a synchronization event is expected."""

    event_id: str
    event_type: str
    activity_index: int
    phase: str
    window_start: datetime
    window_end: datetime

    @property
    def duration_seconds(self) -> float:
        """Return the duration of the synchronization window in seconds."""
        return (self.window_end - self.window_start).total_seconds()