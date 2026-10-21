from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class Activity:
    """Represents one activity performed during a study session."""

    activity_index: int
    start_time: datetime
    end_time: datetime
    duration_seconds: float

    # Optional protocol metadata (manifest schema >= 1.1).
    activity_name: Optional[str] = None  # noqa: UP045
    compendium_code: Optional[str] = None  # noqa: UP045
    planned_duration_seconds: Optional[float] = None  # noqa: UP045
    met: Optional[float] = None  # noqa: UP045
    rpe: Optional[float] = None  # noqa: UP045
