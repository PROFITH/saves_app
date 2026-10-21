"""
Persistent state of one synchronization project.

A project lives in the session folder as ``saves_project.json``: which
files belong to the session, where each sensor was worn, the video claps
marked by hand, the manual corrections of sensor claps and the steps
counted on the video. Reopening the folder restores the work exactly where
it was left.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_FILE = "saves_project.json"
PROJECT_VERSION = 1

LOCATIONS = (
    "Muñeca no dominante",
    "Muñeca dominante",
    "Muslo",
    "Pecho",
    "Brazo",
    "Cadera",
    "Calzado",
    "Otra",
)


@dataclass
class SensorEntry:
    """One sensor file of the session."""

    sensor_id: str
    path: str
    sensor_format: str
    location: str = "Otra"
    serial: str = ""
    enabled: bool = True


@dataclass
class VideoEntry:
    """One video file and, when known, the host time of its first frame."""

    path: str
    recording_start_unix_s: Optional[float] = None  # noqa: UP045


@dataclass
class SyncProject:
    """Everything needed to rebuild the synchronization of a session."""

    folder: str
    manifest_path: Optional[str] = None  # noqa: UP045
    sensors: list[SensorEntry] = field(default_factory=list)
    videos: list[VideoEntry] = field(default_factory=list)
    reference: str = "auto"
    timezone: str = "Europe/Madrid"
    # event_id -> seconds from the start of the (first) video.
    video_events: dict[str, float] = field(default_factory=dict)
    # sensor_id -> event_id -> sensor timestamp (Unix ms) chosen by hand.
    manual_anchors: dict[str, dict[str, float]] = field(default_factory=dict)
    # sensor_id -> event_ids whose clap is left out of the fit.
    excluded_anchors: dict[str, list[str]] = field(default_factory=dict)
    # Steps marked on the video: seconds from the start of the (first) video
    # and the foot ("L", "R" or "" when not told apart), sorted by time.
    steps: list[dict] = field(default_factory=list)
    version: int = PROJECT_VERSION

    @property
    def project_path(self) -> Path:
        return Path(self.folder) / PROJECT_FILE

    def sensor(self, sensor_id: str) -> SensorEntry:
        for entry in self.sensors:
            if entry.sensor_id == sensor_id:
                return entry
        raise KeyError(sensor_id)

    def unique_sensor_id(self, base: str) -> str:
        """Return ``base`` or ``base_2``, ``base_3``... not used yet."""
        taken = {s.sensor_id for s in self.sensors}
        if base not in taken:
            return base
        n = 2
        while f"{base}_{n}" in taken:
            n += 1
        return f"{base}_{n}"

    def add_sensor(self, entry: SensorEntry) -> SensorEntry:
        """Add a sensor unless the same file is already in the project."""
        resolved = str(Path(entry.path).resolve())
        for existing in self.sensors:
            if str(Path(existing.path).resolve()) == resolved:
                return existing
        entry.sensor_id = self.unique_sensor_id(entry.sensor_id)
        self.sensors.append(entry)
        return entry

    def remove_sensor(self, sensor_id: str) -> None:
        self.sensors = [s for s in self.sensors if s.sensor_id != sensor_id]
        self.manual_anchors.pop(sensor_id, None)
        self.excluded_anchors.pop(sensor_id, None)
        if self.reference == sensor_id:
            self.reference = "auto"

    def set_manual_anchor(self, sensor_id: str, event_id: str, unix_ms: float) -> None:
        self.manual_anchors.setdefault(sensor_id, {})[event_id] = float(unix_ms)
        self.set_excluded(sensor_id, event_id, False)

    def clear_manual_anchor(self, sensor_id: str, event_id: str) -> None:
        self.manual_anchors.get(sensor_id, {}).pop(event_id, None)

    def set_excluded(self, sensor_id: str, event_id: str, excluded: bool) -> None:
        events = set(self.excluded_anchors.get(sensor_id, []))
        if excluded:
            events.add(event_id)
        else:
            events.discard(event_id)
        self.excluded_anchors[sensor_id] = sorted(events)

    def is_excluded(self, sensor_id: str, event_id: str) -> bool:
        return event_id in self.excluded_anchors.get(sensor_id, [])

    # ------------------------------------------------------------------
    # Steps counted on the video
    # ------------------------------------------------------------------

    def add_step(self, video_s: float, foot: str = "") -> dict:
        """Mark a step; a second mark on the same frame replaces the first."""
        step = {"video_s": round(float(video_s), 4), "foot": foot}
        self.steps = [s for s in self.steps if abs(s["video_s"] - step["video_s"]) > 1e-3]
        self.steps.append(step)
        self.steps.sort(key=lambda s: s["video_s"])
        return step

    def remove_step_near(self, video_s: float, within_s: float = 0.5) -> Optional[dict]:  # noqa: UP045
        """Remove the step closest to ``video_s`` if it is near enough."""
        if not self.steps:
            return None
        nearest = min(self.steps, key=lambda s: abs(s["video_s"] - video_s))
        if abs(nearest["video_s"] - video_s) > within_s:
            return None
        self.steps.remove(nearest)
        return nearest

    def step_times(self) -> list[float]:
        return [s["video_s"] for s in self.steps]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        data = asdict(self)
        folder = Path(self.folder)
        # Paths inside the session folder are stored relative to it so the
        # folder can be moved or opened from another computer.
        data["manifest_path"] = _relative(self.manifest_path, folder)
        for item in data["sensors"] + data["videos"]:
            item["path"] = _relative(item["path"], folder)
        return data

    def save(self) -> Path:
        path = self.project_path
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, folder: str | Path) -> SyncProject:
        folder = Path(folder)
        data = json.loads((folder / PROJECT_FILE).read_text(encoding="utf-8"))
        project = cls(
            folder=str(folder),
            manifest_path=_absolute(data.get("manifest_path"), folder),
            sensors=[SensorEntry(**{**s, "path": _absolute(s["path"], folder)}) for s in data.get("sensors", [])],
            videos=[VideoEntry(**{**v, "path": _absolute(v["path"], folder)}) for v in data.get("videos", [])],
            reference=data.get("reference", "auto"),
            timezone=data.get("timezone", "Europe/Madrid"),
            video_events={k: float(v) for k, v in data.get("video_events", {}).items()},
            manual_anchors={
                sid: {eid: float(ms) for eid, ms in events.items()}
                for sid, events in data.get("manual_anchors", {}).items()
            },
            excluded_anchors={sid: list(e) for sid, e in data.get("excluded_anchors", {}).items()},
            steps=sorted(
                (
                    {"video_s": float(s["video_s"]), "foot": str(s.get("foot", ""))}
                    for s in data.get("steps", [])
                    if "video_s" in s
                ),
                key=lambda s: s["video_s"],
            ),
        )
        return project

    @classmethod
    def exists_in(cls, folder: str | Path) -> bool:
        return (Path(folder) / PROJECT_FILE).exists()


def _relative(path: Optional[str], folder: Path) -> Optional[str]:  # noqa: UP045
    if not path:
        return path
    p = Path(path)
    try:
        return p.resolve().relative_to(folder.resolve()).as_posix()
    except ValueError:
        return str(p)


def _absolute(path: Optional[str], folder: Path) -> Optional[str]:  # noqa: UP045
    if not path:
        return path
    p = Path(path)
    return str(p if p.is_absolute() else folder / p)
