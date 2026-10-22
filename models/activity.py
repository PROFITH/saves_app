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

    # Attempt (1, or more when the activity was repeated), the kind of
    # segment ("activity", or "calibration" for the calibration box) and
    # the lab_activity instance it was sent to in REDCap by
    # lab_protocol_timer.
    attempt: int = 1
    activity_kind: str = "activity"
    redcap_instance: Optional[int] = None  # noqa: UP045
