"""
Synchronized dataset of a session: every sensor on one clock, labelled.

After synchronizing, the export writes one file per sensor with its samples
on the host clock of the protocol app and, for every sample, the activity,
attempt and phase of the protocol; the activities and phases with their
times; the steps and any other events counted on the video; the
technician's task checklist; and the heart rate of the Polar sensors with
the same labels. Every file shares the time column ``time_unix_ms`` (host
clock), so they can be joined directly.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from studio.project import STEPS, SYNC_TASK, TASK_STATE_TEXT, SyncProject

if TYPE_CHECKING:
    from studio.engine import SyncEngine

# Seconds of signal kept before the start and after the end of the session.
EXPORT_MARGIN_S = 60.0
GZIP = {"method": "gzip", "compresslevel": 1}

PHASE_NAMES = {
    "static_prep": "quieto antes",
    "activity_main": "actividad",
    "static_post": "quieto después",
    "transition_lap": "transición",
    "paused": "pausa",
}

README = """Exportación sincronizada de SAVES Studio
==========================================

Todas las horas están en el reloj del ordenador del cronómetro
(lab_protocol_timer), en milisegundos Unix (time_unix_ms), así que los
archivos se pueden unir directamente por esa columna.

signals/<sensor>.csv.gz
    Una fila por muestra del sensor, ya sincronizada:
    time_unix_ms         hora sincronizada (reloj del cronómetro)
    time_native_unix_ms  hora que escribió el propio sensor
    x, y, z              aceleración en g
    activity_index       número de la actividad del protocolo
    activity             nombre de la actividad
    attempt              intento (1, o más si se repitió)
    phase                static_prep (quieto antes), activity_main
                         (actividad), static_post (quieto después),
                         transition_lap (transición) o paused (pausa)

heart_rate/<dispositivo>.csv
    Frecuencia cardiaca y RR del Polar, con las mismas etiquetas.

activities.csv   Actividades con su inicio y fin, MET, RPE y lo contado en
                 el vídeo en cada una (steps_video, <tipo>_video).
phases.csv       Cada fase de cada actividad con su inicio y fin.
steps.csv        Pasos marcados en el vídeo (si se han contado), con el
                 pie (foot: L izquierdo, R derecho, vacío si no se
                 distinguió).
steps_per_activity.csv
                 Pasos por actividad y cadencia (pasos/min).
events_<tipo>.csv
                 Otros eventos contados en el vídeo, uno por fila, con
                 las mismas columnas que steps.csv (event en lugar de
                 step y side en lugar de foot). Por ejemplo,
                 events_sit_to_stand.csv: cada vez que se queda de pie
                 en el test de sentarse y levantarse de 30 s.
events_<tipo>_per_activity.csv
                 Esos eventos por actividad: <tipo>_main (en la
                 actividad), <tipo>_all_phases (con las fases quietas) y
                 <tipo>_per_min (por minuto de actividad).
tasks.csv        Tareas del técnico: la sincronización y cada conteo, con
                 su estado (not_started no iniciada, started iniciada,
                 finished finalizada), quién la hizo y cuándo.
sync_report.csv  Calidad de la sincronización de cada sensor.
anchors.csv      Palmadas y saltos encontrados en cada sensor.
qc.json          Todos los ajustes (desfase, deriva, error) en detalle.
"""


def _parse_utc(text: str) -> Optional[float]:  # noqa: UP045
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _iso(unix_s: float) -> str:
    return datetime.fromtimestamp(unix_s, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def read_phases(folder: str | Path, session) -> pd.DataFrame:
    """
    Phases of the protocol with their start and end (host Unix s).

    Taken from the timer's log (session_events.csv): each ``phase_started``
    opens a phase that lasts until the next one; a pause is a phase of its
    own. Without the log, the activities of the manifest are the only
    phases.
    """
    columns = ["activity_index", "attempt", "phase", "start_s", "end_s"]
    session_end = session.end_time.timestamp()
    path = Path(folder) / "session_events.csv"
    rows: list[dict] = []
    if path.exists():
        try:
            events = pd.read_csv(path, dtype=str, keep_default_na=False)
        except (OSError, ValueError, pd.errors.ParserError):
            events = pd.DataFrame()
        state: Optional[dict] = None  # noqa: UP045
        before_pause: Optional[dict] = None  # noqa: UP045
        for event in events.to_dict("records"):
            t = _parse_utc(event.get("time_utc", ""))
            if t is None:
                continue
            name = event.get("event", "")
            if name in ("phase_started", "session_paused", "session_resumed",
                        "session_finished", "session_cancelled", "protocol_completed"):
                if state is not None:
                    state["end_s"] = t
                    if state["end_s"] > state["start_s"]:
                        rows.append(state)
                if name == "phase_started":
                    state = {
                        "activity_index": _int(event.get("activity_index")),
                        "attempt": _int(event.get("attempt")) or 1,
                        "phase": event.get("phase") or "",
                        "start_s": t,
                    }
                elif name == "session_paused":
                    before_pause = state
                    state = {**(state or {"activity_index": None, "attempt": None}), "phase": "paused", "start_s": t}
                elif name == "session_resumed" and before_pause is not None:
                    state = {**before_pause, "start_s": t}
                    before_pause = None
                else:
                    state = None
        if state is not None:
            state["end_s"] = max(session_end, state["start_s"])
            if state["end_s"] > state["start_s"]:
                rows.append(state)
    if not rows:
        rows = [
            {
                "activity_index": a.activity_index,
                "attempt": 1,
                "phase": "activity_main",
                "start_s": a.start_time.timestamp(),
                "end_s": a.end_time.timestamp(),
            }
            for a in session.activities
        ]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values("start_s", kind="stable").reset_index(drop=True)


def _int(value) -> Optional[int]:  # noqa: UP045
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def label_times(times_s: np.ndarray, phases: pd.DataFrame, names: dict[int, str]) -> dict[str, object]:
    """Activity, attempt and phase of each time (host Unix s)."""
    n = times_s.size
    activity = pd.array([pd.NA] * n, dtype="Int64")
    attempt = pd.array([pd.NA] * n, dtype="Int64")
    phase = np.full(n, "", dtype=object)
    name = np.full(n, "", dtype=object)
    if n and len(phases):
        starts = phases["start_s"].to_numpy(dtype=float)
        ends = phases["end_s"].to_numpy(dtype=float)
        index = np.searchsorted(starts, times_s, side="right") - 1
        inside = (index >= 0) & (times_s < ends[np.clip(index, 0, None)])
        rows = index[inside]
        act = phases["activity_index"].to_numpy(dtype=object)[rows]
        att = phases["attempt"].to_numpy(dtype=object)[rows]
        activity[inside] = pd.array([pd.NA if pd.isna(v) else v for v in act], dtype="Int64")
        attempt[inside] = pd.array([pd.NA if pd.isna(v) else v for v in att], dtype="Int64")
        phase[inside] = phases["phase"].to_numpy(dtype=object)[rows]
        name[inside] = ["" if pd.isna(v) else names.get(int(v), "") for v in act]
    return {"activity_index": activity, "activity": name, "attempt": attempt, "phase": phase}


def export_dataset(engine: SyncEngine, out_dir: str | Path, margin_s: float = EXPORT_MARGIN_S) -> list[Path]:
    """Write the labelled, synchronized dataset; returns the files written."""
    out = Path(out_dir)
    (out / "signals").mkdir(parents=True, exist_ok=True)
    session = engine.session
    folder = Path(engine.project.folder)
    names = {a.activity_index: a.activity_name or f"Actividad {a.activity_index}" for a in session.activities}
    phases = read_phases(folder, session)
    start = session.start_time.timestamp() - margin_s
    end = session.end_time.timestamp() + margin_s
    written: list[Path] = []

    for sid in engine.active_sensors():
        signal = engine.signals[sid]
        n0 = float(engine.display_to_sensor(sid, start))
        n1 = float(engine.display_to_sensor(sid, end))
        i0 = int(np.searchsorted(signal.t_ms, min(n0, n1)))
        i1 = int(np.searchsorted(signal.t_ms, max(n0, n1), side="right"))
        native = signal.t_ms[i0:i1]
        host = np.asarray(engine.sensor_to_display(sid, native), dtype=float)
        frame = pd.DataFrame({
            "time_unix_ms": np.round(host * 1000.0).astype(np.int64),
            "time_native_unix_ms": np.round(native).astype(np.int64),
            "x": np.round(signal.channels["x"][i0:i1].astype(float), 4),
            "y": np.round(signal.channels["y"][i0:i1].astype(float), 4),
            "z": np.round(signal.channels["z"][i0:i1].astype(float), 4),
            **label_times(host, phases, names),
        })
        path = out / "signals" / f"{_safe(sid)}.csv.gz"
        frame.to_csv(path, index=False, compression=GZIP)
        written.append(path)

    written += _export_heart_rate(folder, out, phases, names, start, end)

    phase_rows = phases.copy()
    phase_rows["activity"] = ["" if pd.isna(v) else names.get(int(v), "") for v in phase_rows["activity_index"]]
    phase_rows["phase_label"] = [PHASE_NAMES.get(p, p) for p in phase_rows["phase"]]
    phase_rows["start_utc"] = [_iso(t) for t in phase_rows["start_s"]]
    phase_rows["end_utc"] = [_iso(t) for t in phase_rows["end_s"]]
    phase_rows["start_unix_ms"] = np.round(phase_rows["start_s"] * 1000).astype(np.int64)
    phase_rows["end_unix_ms"] = np.round(phase_rows["end_s"] * 1000).astype(np.int64)
    phase_rows["duration_s"] = (phase_rows["end_s"] - phase_rows["start_s"]).round(3)
    phase_rows = phase_rows[["activity_index", "activity", "attempt", "phase", "phase_label", "start_utc",
                             "end_utc", "start_unix_ms", "end_unix_ms", "duration_s"]]
    written.append(out / "phases.csv")
    phase_rows.to_csv(written[-1], index=False)

    counted = {}
    for kind in engine.project.event_types:
        table = event_table(engine, kind.key, phases, names)
        if not len(table):
            continue
        counted[kind.key] = table
        stem = "steps" if kind.key == STEPS else f"events_{_safe(kind.key)}"
        written.append(out / f"{stem}.csv")
        table.to_csv(written[-1], index=False)
        written.append(out / f"{stem}_per_activity.csv")
        events_per_activity(engine, kind.key, table, phases, names).to_csv(written[-1], index=False)

    written.append(out / "activities.csv")
    activity_table(engine, folder, counted).to_csv(written[-1], index=False)

    written.append(out / "tasks.csv")
    task_table(engine.project).to_csv(written[-1], index=False)

    written.append(out / "LEEME.txt")
    written[-1].write_text(readme(engine.project), encoding="utf-8")
    return written


def readme(project: SyncProject) -> str:
    """The guide to the files, headed by who exported them and what was left."""
    head = f"Exportado el {datetime.now().astimezone():%Y-%m-%d %H:%M}"
    if project.technician:
        head += f" por el técnico {project.technician}"
    head += ".\n"
    pending = project.pending_tasks()
    if pending:
        head += "Tareas sin finalizar al exportar: " + "; ".join(
            f"{project.task_name(k)} ({TASK_STATE_TEXT[project.task_state(k)].lower()})" for k in pending
        ) + ".\n"
    else:
        head += "Todas las tareas estaban finalizadas.\n"
    return f"{head}\n{README}"


def task_table(project: SyncProject) -> pd.DataFrame:
    """The technician's checklist: one row per task."""
    rows = []
    for key in project.task_keys():
        task = project.tasks.get(key, {})
        state = project.task_state(key)
        rows.append({
            "task": key,
            "name": project.task_name(key),
            "state": state,
            "state_label": TASK_STATE_TEXT[state],
            "technician": task.get("by", "") if state != "not_started" else "",
            "started_at": task.get("started_at", ""),
            "finished_at": task.get("finished_at", ""),
            "events": "" if key == SYNC_TASK else len(project.events.get(key, [])),
            "copied_from": task.get("copied_from", ""),
        })
    return pd.DataFrame(rows)


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text) or "sensor"


def _export_heart_rate(folder: Path, out: Path, phases, names, start: float, end: float) -> list[Path]:
    """Polar heart rate and RR files of the session, labelled."""
    written = []
    candidates = sorted(folder.glob("polar_hr_*.csv")) + sorted(folder.glob("*/polar_hr_*.csv"))
    for path in candidates:
        try:
            data = pd.read_csv(path, dtype={"rr_intervals_ms": str}, keep_default_na=False)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        if "timestamp_iso" not in data.columns or data.empty:
            continue
        times = pd.to_datetime(data["timestamp_iso"], utc=True, format="ISO8601", errors="coerce")
        valid = times.notna().to_numpy()
        data = data.loc[valid].reset_index(drop=True)
        seconds = times[valid].astype("int64").to_numpy() / 1e9
        keep = (seconds >= start) & (seconds <= end)
        data, seconds = data.loc[keep].reset_index(drop=True), seconds[keep]
        frame = pd.DataFrame({
            "time_unix_ms": np.round(seconds * 1000.0).astype(np.int64),
            "hr_bpm": pd.to_numeric(data.get("hr_bpm"), errors="coerce"),
            "rr_intervals_ms": data.get("rr_intervals_ms", ""),
            **label_times(seconds, phases, names),
        })
        device = re.sub(r"_\d{8}_\d+$", "", path.stem[len("polar_hr_"):]) or "polar"
        (out / "heart_rate").mkdir(parents=True, exist_ok=True)
        target = out / "heart_rate" / f"{_safe(device)}.csv"
        frame.to_csv(target, index=False)
        written.append(target)
    return written


def _event_columns(project: SyncProject, key: str) -> tuple[str, Optional[str]]:  # noqa: UP045
    """Names of the number and side columns of a kind of event (side None if it has none)."""
    if key == STEPS:
        return "step", "foot"
    try:
        sides = project.event_type(key).sides
    except KeyError:
        sides = False
    return "event", ("side" if sides else None)


def event_table(engine: SyncEngine, key: str, phases: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    """Events of one kind marked on the video, on the host clock and labelled."""
    number, side = _event_columns(engine.project, key)
    columns = [number, "video_seconds", "time_unix_ms", "time_utc", *([side] if side else []),
               "activity_index", "activity", "attempt", "phase"]
    events = engine.project.events.get(key, [])
    if not events:
        return pd.DataFrame(columns=columns)
    video = np.array([e["video_s"] for e in events], dtype=float)
    host = np.atleast_1d(np.asarray(engine.video_to_display(video), dtype=float))
    frame = pd.DataFrame({
        number: np.arange(1, video.size + 1),
        "video_seconds": np.round(video, 4),
        "time_unix_ms": np.round(host * 1000.0).astype(np.int64),
        "time_utc": [_iso(t) for t in host],
        **({side: [e.get("side", "") for e in events]} if side else {}),
        **label_times(host, phases, names),
    })
    return frame[columns]


def step_table(engine: SyncEngine, phases: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    """Steps marked on the video, on the host clock and labelled."""
    return event_table(engine, STEPS, phases, names)


def _count_columns(key: str) -> tuple[str, str, str]:
    if key == STEPS:
        return "steps_main", "steps_all_phases", "cadence_steps_min"
    return f"{key}_main", f"{key}_all_phases", f"{key}_per_min"


def events_per_activity(engine: SyncEngine, key: str, events: pd.DataFrame, phases: pd.DataFrame,
                        names: dict[int, str]) -> pd.DataFrame:
    """Events of each activity (main phase and all its phases) and their rate per minute."""
    main_col, all_col, rate_col = _count_columns(key)
    video_range = engine.video_display_range()
    rows = []
    for activity in engine.session.activities:
        index = activity.activity_index
        main = phases[(phases["activity_index"] == index) & (phases["phase"] == "activity_main")]
        main_s = float((main["end_s"] - main["start_s"]).sum()) if len(main) else activity.duration_seconds
        in_activity = events[events["activity_index"] == index]
        in_main = in_activity[in_activity["phase"] == "activity_main"]
        covered = None
        if video_range is not None and len(main):
            overlap = sum(
                max(0.0, min(e, video_range[1]) - max(s, video_range[0]))
                for s, e in zip(main["start_s"], main["end_s"])
            )
            covered = round(100.0 * overlap / main_s, 1) if main_s > 0 else None
        rows.append({
            "activity_index": index,
            "activity": names.get(index, ""),
            main_col: int(len(in_main)),
            all_col: int(len(in_activity)),
            "main_duration_s": round(main_s, 3),
            rate_col: round(60.0 * len(in_main) / main_s, 1) if main_s > 0 else None,
            "video_coverage_pct": covered,
        })
    return pd.DataFrame(rows)


def steps_per_activity(engine: SyncEngine, steps: pd.DataFrame, phases: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    """Steps of each activity (main phase and all its phases) and cadence."""
    return events_per_activity(engine, STEPS, steps, phases, names)


def activity_table(engine: SyncEngine, folder: Path, counted: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Activities with their times, the timer's markers and the events counted on the video."""
    rows = []
    for a in engine.session.activities:
        rows.append({
            "activity_index": a.activity_index,
            "activity": a.activity_name,
            "compendium_code": a.compendium_code,
            "met": a.met,
            "rpe": a.rpe,
            "start_utc": _iso(a.start_time.timestamp()),
            "end_utc": _iso(a.end_time.timestamp()),
            "start_unix_ms": int(round(a.start_time.timestamp() * 1000)),
            "end_unix_ms": int(round(a.end_time.timestamp() * 1000)),
            "duration_s": a.duration_seconds,
            "planned_duration_s": a.planned_duration_seconds,
        })
    frame = pd.DataFrame(rows)
    markers_path = folder / "activity_markers.csv"
    if markers_path.exists() and len(frame):
        try:
            markers = pd.read_csv(markers_path, keep_default_na=False)
        except (OSError, ValueError, pd.errors.ParserError):
            markers = pd.DataFrame()
        extra = [c for c in ("attempt", "completion", "end_reason", "paused_seconds", "rpe_before",
                             "resting_hr_bpm", "incident") if c in markers.columns]
        if "activity_index" in markers.columns and extra:
            last = markers.drop_duplicates("activity_index", keep="last")[["activity_index", *extra]]
            frame = frame.merge(last, on="activity_index", how="left")
    for key, events in counted.items():
        if len(events) and len(frame):
            main = events[events["phase"] == "activity_main"].groupby("activity_index").size()
            frame[f"{key}_video"] = [int(main.get(i, 0)) for i in frame["activity_index"]]
    return frame
