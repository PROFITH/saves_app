"""
Persistent state of one synchronization project.

A project lives in the session folder, one file per technician
(``saves_project_<code>.json``): which files belong to the session, where
each sensor was worn, the video claps marked by hand, the manual corrections
of sensor claps, the events counted on the video (steps, sit-to-stands or
any other kind), the technician's task checklist and which plots are
hidden. Reopening the folder with the same code restores that technician's
work exactly where it was left; a second technician works apart, so a video
can be reviewed by two people independently. ``saves_project.json`` is the
file of the versions before technicians; it is read when nobody has a file
of their own yet.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_FILE = "saves_project.json"
PROJECT_VERSION = 2
TECHNICIAN_FILE = re.compile(r"^saves_project_([A-Z0-9_-]{1,12})\.json$")

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

# Checklist: every project has the synchronization and one task per kind of
# event counted on the video. Only the technician marks a task finished.
SYNC_TASK = "sync"
NOT_STARTED, STARTED, FINISHED = "not_started", "started", "finished"
TASK_STATES = (NOT_STARTED, STARTED, FINISHED)
TASK_STATE_TEXT = {NOT_STARTED: "No iniciada", STARTED: "Iniciada", FINISHED: "Finalizada"}

STEPS = "steps"
SIT_TO_STAND = "sit_to_stand"


@dataclass
class EventType:
    """A kind of event counted on the video, such as steps."""

    key: str
    name: str
    # Left and right can be told apart (keys A and D).
    sides: bool = False
    # What the technician marks, after "Pulsa S en…".
    hint: str = ""
    # Text of the mark button of the video, before "[S]".
    button: str = ""

    @property
    def mark_hint(self) -> str:
        return self.hint or f"cada «{self.name}»"

    @property
    def button_text(self) -> str:
        return self.button or f"＋ {self.name}"

    @property
    def lower_name(self) -> str:
        """Name inside a sentence (``Pasos`` → ``pasos``, ``TUG`` stays)."""
        name = self.name
        return name[0].lower() + name[1:] if len(name) > 1 and name[1].islower() else name


BUILT_IN_EVENT_TYPES = {
    STEPS: EventType(STEPS, "Pasos", sides=True, hint="cada apoyo del pie", button="👣 Paso"),
    SIT_TO_STAND: EventType(
        SIT_TO_STAND, "Sentarse y levantarse",
        hint="cada vez que se queda de pie del todo (test de 30 s)", button="🪑 Se levanta",
    ),
}
DEFAULT_EVENT_TYPES = (STEPS, SIT_TO_STAND)


def default_event_types() -> list[EventType]:
    return [EventType(**asdict(BUILT_IN_EVENT_TYPES[k])) for k in DEFAULT_EVENT_TYPES]


def _event_type_from(raw: dict) -> EventType:
    key = str(raw["key"])
    built_in = BUILT_IN_EVENT_TYPES.get(key)
    return EventType(
        key=key,
        name=str(raw.get("name") or (built_in.name if built_in else key)),
        sides=bool(raw.get("sides", built_in.sides if built_in else False)),
        # The texts of the built-in kinds follow the version of the app.
        hint=built_in.hint if built_in else str(raw.get("hint", "")),
        button=built_in.button if built_in else str(raw.get("button", "")),
    )


def event_type_key(name: str, taken: set[str] = frozenset()) -> str:  # type: ignore[assignment]
    """Key for a new kind of event, from its name (``Escalones`` → ``escalones``)."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    base = re.sub(r"[^a-z0-9]+", "_", plain.lower()).strip("_") or "eventos"
    key, n = base, 2
    while key in taken:
        key, n = f"{base}_{n}", n + 1
    return key


def normalize_technician(code: str) -> str:
    """Technician code as stored (``jhm`` → ``JHM``), or "" when not valid."""
    plain = unicodedata.normalize("NFKD", code or "").encode("ascii", "ignore").decode().strip().upper()
    return plain if re.fullmatch(r"[A-Z0-9_-]{1,12}", plain) else ""


def project_file(technician: str = "") -> str:
    return f"saves_project_{technician}.json" if technician else PROJECT_FILE


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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
    # Code of the technician whose work this is ("" before technicians).
    technician: str = ""
    # Kinds of events counted on the video, and the events of each kind:
    # seconds from the start of the (first) video and the side ("L", "R"
    # or "" when not told apart), sorted by time.
    event_types: list[EventType] = field(default_factory=default_event_types)
    events: dict[str, list[dict]] = field(default_factory=dict)
    # task key -> {"state", "started_at", "finished_at", "by"}.
    tasks: dict[str, dict] = field(default_factory=dict)
    # Sensors whose plot is hidden (they still take part in the fit).
    hidden_plots: list[str] = field(default_factory=list)
    version: int = PROJECT_VERSION

    @property
    def project_path(self) -> Path:
        return Path(self.folder) / project_file(self.technician)

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
        if sensor_id in self.hidden_plots:
            self.hidden_plots.remove(sensor_id)
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

    def set_plot_hidden(self, sensor_id: str, hidden: bool) -> None:
        if hidden and sensor_id not in self.hidden_plots:
            self.hidden_plots.append(sensor_id)
        elif not hidden and sensor_id in self.hidden_plots:
            self.hidden_plots.remove(sensor_id)

    # ------------------------------------------------------------------
    # Events counted on the video
    # ------------------------------------------------------------------

    def event_type(self, key: str) -> EventType:
        for kind in self.event_types:
            if kind.key == key:
                return kind
        raise KeyError(key)

    def add_event_type(self, name: str, sides: bool = False) -> EventType:
        """A new kind of event, named by the technician."""
        name = name.strip()
        for kind in self.event_types:
            if kind.name.lower() == name.lower():
                return kind
        kind = EventType(event_type_key(name, {k.key for k in self.event_types}), name, sides=sides)
        self.event_types.append(kind)
        return kind

    def remove_event_type(self, key: str) -> None:
        """Remove a kind of event; only one with no events marked."""
        if self.events.get(key):
            raise ValueError("Esta categoría ya tiene eventos marcados")
        self.event_types = [k for k in self.event_types if k.key != key]
        self.events.pop(key, None)
        self.tasks.pop(key, None)

    def events_of(self, key: str) -> list[dict]:
        return self.events.setdefault(key, [])

    def add_event(self, key: str, video_s: float, side: str = "") -> dict:
        """Mark an event; a second mark on the same frame replaces the first."""
        event = {"video_s": round(float(video_s), 4), "side": side}
        events = [e for e in self.events_of(key) if abs(e["video_s"] - event["video_s"]) > 1e-3]
        events.append(event)
        events.sort(key=lambda e: e["video_s"])
        self.events[key] = events
        return event

    def remove_event_near(self, key: str, video_s: float, within_s: float = 0.5) -> Optional[dict]:  # noqa: UP045
        """Remove the event closest to ``video_s`` if it is near enough."""
        events = self.events_of(key)
        if not events:
            return None
        nearest = min(events, key=lambda e: abs(e["video_s"] - video_s))
        if abs(nearest["video_s"] - video_s) > within_s:
            return None
        events.remove(nearest)
        return nearest

    def event_times(self, key: str) -> list[float]:
        return [e["video_s"] for e in self.events_of(key)]

    # Steps, the first kind of event (kept for the callers that count steps).

    @property
    def steps(self) -> list[dict]:
        return self.events_of(STEPS)

    @steps.setter
    def steps(self, value: list[dict]) -> None:
        self.events[STEPS] = list(value)

    def add_step(self, video_s: float, foot: str = "") -> dict:
        return self.add_event(STEPS, video_s, foot)

    def remove_step_near(self, video_s: float, within_s: float = 0.5) -> Optional[dict]:  # noqa: UP045
        return self.remove_event_near(STEPS, video_s, within_s)

    def step_times(self) -> list[float]:
        return self.event_times(STEPS)

    # ------------------------------------------------------------------
    # Task checklist
    # ------------------------------------------------------------------

    def task_keys(self) -> list[str]:
        return [SYNC_TASK, *(k.key for k in self.event_types)]

    def task_name(self, key: str) -> str:
        if key == SYNC_TASK:
            return "Sincronización"
        try:
            return f"Conteo de {self.event_type(key).lower_name}"
        except KeyError:
            return key

    def task_state(self, key: str) -> str:
        return self.tasks.get(key, {}).get("state", NOT_STARTED)

    def work_on(self, key: str) -> str:
        """
        Record a change of the technician in a task.

        A task not started becomes started, and a finished one goes back to
        started: "finished" always means that the technician confirmed the
        task after its last change. Returns "started", "reopened" or "".
        """
        state = self.task_state(key)
        if state == STARTED:
            return ""
        self.set_task_state(key, STARTED)
        return "started" if state == NOT_STARTED else "reopened"

    def set_task_state(self, key: str, state: str) -> None:
        if state not in TASK_STATES:
            raise ValueError(state)
        task = dict(self.tasks.get(key, {}))
        task["state"] = state
        task["by"] = self.technician
        if state == NOT_STARTED:
            task.pop("started_at", None)
            task.pop("finished_at", None)
        else:
            task.setdefault("started_at", _now())
            if state == FINISHED:
                task["finished_at"] = _now()
            else:
                task.pop("finished_at", None)
        self.tasks[key] = task

    def pending_tasks(self) -> list[str]:
        """Tasks not finished yet, in checklist order."""
        return [k for k in self.task_keys() if self.task_state(k) != FINISHED]

    def progress_summary(self) -> str:
        """One line with the state of every task, for other technicians."""
        parts = []
        for key in self.task_keys():
            name = "Sincronización" if key == SYNC_TASK else self.event_type(key).name
            text = f"{name}: {TASK_STATE_TEXT[self.task_state(key)].lower()}"
            if key != SYNC_TASK and self.events.get(key):
                text += f" ({len(self.events[key])})"
            parts.append(text)
        return " · ".join(parts)

    def has_sync_work(self) -> bool:
        return bool(self.video_events or self.manual_anchors or any(self.excluded_anchors.values()))

    def copy_sync_from(self, other: SyncProject) -> None:
        """Start from the synchronization of another technician (not the counts)."""
        self.manifest_path = other.manifest_path
        self.sensors = [SensorEntry(**asdict(s)) for s in other.sensors]
        self.videos = [VideoEntry(**asdict(v)) for v in other.videos]
        self.reference = other.reference
        self.timezone = other.timezone
        self.video_events = dict(other.video_events)
        self.manual_anchors = {sid: dict(a) for sid, a in other.manual_anchors.items()}
        self.excluded_anchors = {sid: list(e) for sid, e in other.excluded_anchors.items()}
        if other.task_state(SYNC_TASK) != NOT_STARTED:
            # Still the other technician's work: "by" says who did it.
            self.tasks[SYNC_TASK] = {**other.tasks[SYNC_TASK], "copied_from": other.technician}

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
    def load(cls, folder: str | Path, technician: str = "") -> SyncProject:
        """The work of ``technician`` in ``folder`` (the file before technicians when "")."""
        folder = Path(folder)
        data = json.loads((folder / project_file(technician)).read_text(encoding="utf-8"))
        kinds = [
            _event_type_from(k) for k in data.get("event_types", []) if isinstance(k, dict) and k.get("key")
        ] or default_event_types()
        events = {
            key: _events(raw, "side") for key, raw in (data.get("events") or {}).items() if isinstance(raw, list)
        }
        if "steps" in data and STEPS not in events:
            # Version 1: only steps, with the side in "foot".
            events[STEPS] = _events(data["steps"], "foot")
        project = cls(
            folder=str(folder),
            manifest_path=_absolute(data.get("manifest_path"), folder),
            sensors=[SensorEntry(**{**s, "path": _absolute(s["path"], folder)}) for s in data.get("sensors", [])],
            videos=[VideoEntry(**{**v, "path": _absolute(v["path"], folder)}) for v in data.get("videos", [])],
            reference=data.get("reference", "auto"),
            timezone=data.get("timezone", "Europe/Madrid"),
            video_events={k: float(v) for k, v in data.get("video_events", {}).items()},
            manual_anchors={
                sid: {eid: float(ms) for eid, ms in events_.items()}
                for sid, events_ in data.get("manual_anchors", {}).items()
            },
            excluded_anchors={sid: list(e) for sid, e in data.get("excluded_anchors", {}).items()},
            technician=technician,
            event_types=kinds,
            events=events,
            tasks={k: dict(v) for k, v in (data.get("tasks") or {}).items() if isinstance(v, dict)},
            hidden_plots=[str(s) for s in data.get("hidden_plots", [])],
        )
        return project

    @classmethod
    def exists_in(cls, folder: str | Path, technician: str = "") -> bool:
        return (Path(folder) / project_file(technician)).exists()

    @staticmethod
    def technicians_in(folder: str | Path) -> list[str]:
        """Codes of the technicians with work saved in ``folder``."""
        try:
            names = [p.name for p in Path(folder).iterdir() if p.is_file()]
        except OSError:
            return []
        return sorted(m.group(1) for m in map(TECHNICIAN_FILE.match, names) if m)


def _events(raw: list, side_key: str) -> list[dict]:
    return sorted(
        (
            {"video_s": float(e["video_s"]), "side": str(e.get(side_key, e.get("side", "")) or "")}
            for e in raw
            if isinstance(e, dict) and "video_s" in e
        ),
        key=lambda e: e["video_s"],
    )


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
