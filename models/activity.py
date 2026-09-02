from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Activity:
    """Represents one activity performed during a study session."""

    activity_index: int
    start_time: datetime
    end_time: datetime
    duration_seconds: float