from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

# Typical delay (s) from the go of the cue to the clap (reaction time) or
# to the landing of the jump (reaction, countermovement and flight); used
# to look for the event around the cue.
EVENT_DELAY_SECONDS = {"clap": 0.35, "jump": 0.9}


@dataclass(frozen=True)
class SyncWindow:
    """Defines a time interval in which a synchronization event is expected."""

    event_id: str
    event_type: str
    activity_index: int
    phase: str
    window_start: datetime
    window_end: datetime

    # "clap" or "jump" (manifest >= 1.1 written by lab_protocol_timer).
    sync_kind: str = "clap"
    # Host time of the go of the 3-2-1 cue, when the timer gave one.
    cue_time: Optional[datetime] = None  # noqa: UP045

    @property
    def duration_seconds(self) -> float:
        """Return the duration of the synchronization window in seconds."""
        return (self.window_end - self.window_start).total_seconds()

    @property
    def expected_time(self) -> Optional[datetime]:  # noqa: UP045
        """Host time at which the clap or landing most likely happened."""
        if self.cue_time is None:
            return None
        delay = EVENT_DELAY_SECONDS.get(self.sync_kind, EVENT_DELAY_SECONDS["clap"])
        return self.cue_time + timedelta(seconds=delay)
