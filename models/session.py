from dataclasses import dataclass, field
from datetime import datetime

from .activity import Activity
from .sync_window import SyncWindow


@dataclass(frozen=True)
class Participant:
    """Participant metadata associated with a study session."""

    participant_id: str
    redcap_event_name: str


@dataclass(frozen=True)
class Session:
    """Represents the metadata and timing information of one study session."""

    schema_version: str
    session_id: str
    created_by: str

    start_time: datetime
    end_time: datetime

    participants: list[Participant] = field(default_factory=list)
    activities: list[Activity] = field(default_factory=list)
    sync_windows: list[SyncWindow] = field(default_factory=list)

    video_files: list[str] = field(default_factory=list)

    sensors: dict = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Return the total duration of the session in seconds."""
        return (self.end_time - self.start_time).total_seconds()