"""
What SAVES Studio sends to REDCap: the steps and the sit-to-stands counted
on the video in each attempt of each activity.

They go to the ``lab_activity`` instance that lab_protocol_timer created
for that attempt, each with the code of the technician who counted, and
only once the count is finished. The time of every event stays in the
exported files (``steps.csv``, ``events_sit_to_stand.csv``): in REDCap they
would be hundreds of rows per session that nobody analyses there.

The instance of each attempt is in the session manifest
(``redcap_instance``). Sessions saved before it was written there take it
from the records lab_protocol_timer kept in the folder
(``redcap_records*.json``).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from studio.export import event_table, events_per_activity, read_phases
from studio.project import FINISHED, SIT_TO_STAND, STEPS, SYNC_TASK, SyncProject

if TYPE_CHECKING:
    from models import Participant
    from studio.engine import SyncEngine

DEFAULT_API_URL = "https://redcap.ugr.es/api/"
INSTRUMENT = "lab_activity"
# Fields of lab_activity that take each count: the number and who counted.
COUNT_FIELDS = {
    STEPS: ("la_steps_video", "la_steps_video_by"),
    SIT_TO_STAND: ("la_reps_video", "la_reps_video_by"),
}
# Share of the main phase of an attempt the video has to cover for its
# count to be sent (a count of 0 outside the video means nothing).
MIN_VIDEO_COVERAGE_PCT = 95.0
CSV_NAME = "redcap_conteos.csv"


class RedcapError(Exception):
    """Something the technician has to read: what failed and what to do."""


class TokenRejected(RedcapError):
    """REDCap does not accept the token: it has to be typed again."""


def api_url() -> str:
    return os.environ.get("REDCAP_API_URL", "").strip() or DEFAULT_API_URL


def env_token() -> str:
    return os.environ.get("REDCAP_TOKEN", "").strip()


# ----------------------------------------------------------------------
# Which instance is each attempt
# ----------------------------------------------------------------------

def _records_files(folder: Path) -> list[Path]:
    """Records kept by lab_protocol_timer: the sent ones first, newest first."""
    sent = sorted(folder.glob("redcap_records_enviados_*.json"), reverse=True)
    pending = [folder / "redcap_records.json"] if (folder / "redcap_records.json").exists() else []
    return sent + pending


def attempt_instances(session, folder: str | Path, participant: Participant) -> tuple[dict[tuple[int, int], int], str]:
    """
    The ``lab_activity`` instance of each attempt, keyed by (activity, attempt).

    Returns the instances and where they come from ("" when they cannot be
    known: then nothing should be sent).
    """
    activities = [a for a in session.activities if a.activity_kind != "calibration"]
    if activities and all(a.redcap_instance for a in activities):
        return {(a.activity_index, a.attempt): int(a.redcap_instance) for a in activities}, "session_manifest.json"
    for path in _records_files(Path(folder)):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(rows, list):
            continue
        timed = []
        for row in rows:
            if not isinstance(row, dict) or row.get("redcap_repeat_instrument") != INSTRUMENT:
                continue
            if str(row.get("record_id")) != participant.participant_id:
                continue
            if str(row.get("redcap_event_name")) != participant.redcap_event_name or not row.get("la_start"):
                continue
            try:
                timed.append((int(row["redcap_repeat_instance"]), str(row.get("la_name", ""))))
            except (KeyError, TypeError, ValueError):
                continue
        timed.sort()
        # Both follow the order of the session: the attempts with a time
        # span, one by one. The names must agree, or nothing is assumed.
        names = [a.activity_name or f"Actividad {a.activity_index}" for a in activities]
        if len(timed) == len(activities) and [name for _, name in timed] == names:
            return {
                (a.activity_index, a.attempt): instance for a, (instance, _) in zip(activities, timed)
            }, path.name
    return {}, ""


# ----------------------------------------------------------------------
# What is sent
# ----------------------------------------------------------------------

@dataclass
class CountRecords:
    """The records of one participant, and what was left out and why."""

    participant: Participant
    technician: str
    records: list[dict] = field(default_factory=list)
    # Finished counts that were prepared, and how many attempts got a value
    # from each.
    kinds: list[str] = field(default_factory=list)
    sent_counts: dict[str, int] = field(default_factory=dict)
    left_out: list[str] = field(default_factory=list)
    instances_from: str = ""
    # What REDCap must already have in each instance: this session's
    # attempt, sent by lab_protocol_timer.
    session_id: str = ""
    expected_names: dict[int, str] = field(default_factory=dict)


def countable_kinds(project: SyncProject) -> list[str]:
    """Kinds of events REDCap has fields for, in checklist order."""
    keys = {k.key for k in project.event_types}
    return [k for k in COUNT_FIELDS if k in keys]


def finished_kinds(project: SyncProject) -> list[str]:
    return [k for k in countable_kinds(project) if project.task_state(k) == FINISHED]


def count_records(engine: SyncEngine, participant: Participant) -> CountRecords:
    """
    The ``lab_activity`` rows with the finished counts of the technician.

    Steps go for every attempt the video covers (0 is a count too: a still
    activity). Sit-to-stands go only for the attempts where some were
    marked, so the rest of the activities stay empty. Raises RedcapError
    when nothing can be sent and the technician has to do something first.
    """
    project = engine.project
    technician = project.technician
    if not technician:
        raise RedcapError("Abre la sesión con tu código de técnico: REDCap guarda quién hizo cada conteo.")
    kinds = finished_kinds(project)
    if not kinds:
        names = " o ".join(f"«{project.task_name(k)}»" for k in countable_kinds(project)) or "un conteo"
        raise RedcapError(f"No hay ningún conteo finalizado que enviar: finaliza {names} en la lista de tareas.")
    if project.task_state(SYNC_TASK) != FINISHED:
        raise RedcapError(
            "Finaliza antes la sincronización: sin ella no se sabe con seguridad en qué actividad cae cada evento."
        )
    instances, source = attempt_instances(engine.session, project.folder, participant)
    if not instances:
        raise RedcapError(
            "No se sabe qué instancia de lab_activity es cada actividad. La sesión se guardó con una versión "
            "antigua de lab_protocol_timer y en la carpeta no está redcap_records.json, o no coincide con el manifiesto."
        )

    folder = Path(project.folder)
    session = engine.session
    phases = read_phases(folder, session)
    names = {a.activity_index: a.activity_name or f"Actividad {a.activity_index}" for a in session.activities}
    result = CountRecords(participant=participant, technician=technician, kinds=kinds, instances_from=source,
                          session_id=session.session_id)
    rows: dict[int, dict] = {}
    for key in kinds:
        value_field, by_field = COUNT_FIELDS[key]
        events = event_table(engine, key, phases, names)
        per = events_per_activity(engine, key, events, phases, names)
        main_col = "steps_main" if key == STEPS else f"{key}_main"
        sent = 0
        for item in per.to_dict("records"):
            instance = instances.get((int(item["activity_index"]), int(item["attempt"])))
            if instance is None:
                continue
            label = names.get(int(item["activity_index"]), "")
            if int(item["attempt"]) > 1:
                label += f" (intento {int(item['attempt'])})"
            count = int(item[main_col])
            if key == SIT_TO_STAND and count == 0:
                continue
            coverage = item.get("video_coverage_pct")
            if coverage is None or pd.isna(coverage) or float(coverage) < MIN_VIDEO_COVERAGE_PCT:
                shown = "nada" if coverage is None or pd.isna(coverage) else f"el {float(coverage):.0f} %"
                result.left_out.append(f"{project.task_name(key)} de «{label}»: el vídeo cubre {shown} de la actividad")
                continue
            row = rows.setdefault(instance, {
                "record_id": participant.participant_id,
                "redcap_event_name": participant.redcap_event_name,
                "redcap_repeat_instrument": INSTRUMENT,
                "redcap_repeat_instance": str(instance),
            })
            row[value_field] = str(count)
            row[by_field] = technician
            result.expected_names[instance] = names.get(int(item["activity_index"]), "")
            sent += 1
        result.sent_counts[key] = sent
    result.records = [rows[i] for i in sorted(rows)]
    return result


CSV_COLUMNS = ["record_id", "redcap_event_name", "redcap_repeat_instrument", "redcap_repeat_instance",
               "la_steps_video", "la_steps_video_by", "la_reps_video", "la_reps_video_by"]


def write_import_csv(records: list[dict], path: str | Path) -> Path:
    """The records as a file for REDCap's Data Import Tool (empty cells change nothing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records, columns=CSV_COLUMNS).fillna("")
    frame.to_csv(path, index=False)
    return path


# ----------------------------------------------------------------------
# REDCap API
# ----------------------------------------------------------------------

def _post(url: str, data: list[tuple[str, str]], timeout: float = 30.0) -> str:
    request = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(data).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        try:
            body = json.loads(body).get("error", body)
        except (ValueError, AttributeError):
            pass
        if error.code in (401, 403):
            raise TokenRejected(f"REDCap no acepta el token (respuesta {error.code}): {body}") from error
        raise RedcapError(f"REDCap respondió {error.code}: {body}{_missing_fields_hint(body)}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        reason = getattr(error, "reason", error)
        raise RedcapError(f"No se pudo conectar con REDCap ({url}): {reason}") from error


def _missing_fields_hint(text: str) -> str:
    if any(f in text for pair in COUNT_FIELDS.values() for f in pair):
        return ("\nEn REDCap faltan los campos de los conteos del vídeo: importa el diccionario v6 "
                "(RUN4HEALTH_DataDictionary_2026-10-21_v6.csv).")
    return ""


def current_rows(token: str, participant: Participant, url: Optional[str] = None) -> dict[int, dict]:  # noqa: UP045
    """The lab_activity instances REDCap has for the participant's visit: attempt and counts."""
    fields = ["record_id", "la_session_id", "la_name", *[f for pair in COUNT_FIELDS.values() for f in pair]]
    data = [
        ("token", token), ("content", "record"), ("format", "json"), ("type", "flat"),
        ("records[0]", participant.participant_id), ("events[0]", participant.redcap_event_name),
        ("returnFormat", "json"),
        *[(f"fields[{i}]", f) for i, f in enumerate(fields)],
    ]
    text = _post(url or api_url(), data)
    try:
        rows = json.loads(text)
    except ValueError as error:
        raise RedcapError(f"REDCap no devolvió datos legibles: {text[:200]}") from error
    if isinstance(rows, dict) and rows.get("error"):
        raise RedcapError(f"REDCap: {rows['error']}{_missing_fields_hint(str(rows['error']))}")
    found = {}
    for row in rows if isinstance(rows, list) else []:
        if row.get("redcap_repeat_instrument") != INSTRUMENT:
            continue
        try:
            found[int(row["redcap_repeat_instance"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return found


def check_instances(prepared: CountRecords, existing: dict[int, dict]) -> list[str]:
    """
    Why the records must not be sent: REDCap does not have the attempts of
    this session in those instances (sent by lab_protocol_timer), so the
    counts would land on another session's activities or on empty ones.
    """
    who = f"{prepared.participant.participant_id} ({prepared.participant.redcap_event_name})"
    missing = [i for i in prepared.expected_names if i not in existing]
    if missing:
        return [(
            f"REDCap todavía no tiene las actividades de esta sesión en el registro {who}. Envíalas antes desde "
            "lab_protocol_timer («Reintentar REDCap», o «Enviar a REDCap una sesión guardada…» en la pantalla de inicio)."
        )]
    problems = []
    for instance, name in sorted(prepared.expected_names.items()):
        row = existing[instance]
        session_id = str(row.get("la_session_id", "") or "")
        if prepared.session_id and session_id and session_id != prepared.session_id:
            problems.append(f"La instancia {instance} es de otra sesión ({session_id}), no de {prepared.session_id}.")
        elif str(row.get("la_name", "") or "") != name:
            problems.append(f"La instancia {instance} es «{row.get('la_name', '')}», no «{name}».")
    return problems


def counts_of_others(prepared: CountRecords, existing: dict[int, dict]) -> dict[str, set[str]]:
    """Task key -> codes of other technicians whose counts these records would replace."""
    others: dict[str, set[str]] = {}
    for record in prepared.records:
        current = existing.get(int(record["redcap_repeat_instance"]), {})
        for key, (value_field, by_field) in COUNT_FIELDS.items():
            if value_field not in record:
                continue
            by = str(current.get(by_field, "") or "").strip()
            has_value = str(current.get(value_field, "") or "").strip() != ""
            if has_value and by and by != prepared.technician:
                others.setdefault(key, set()).add(by)
    return others


def import_records(token: str, records: list[dict], url: Optional[str] = None) -> int:  # noqa: UP045
    """Sends the records; returns how many REDCap records were updated."""
    data = [
        ("token", token), ("content", "record"), ("format", "json"), ("type", "flat"),
        ("overwriteBehavior", "normal"), ("data", json.dumps(records, ensure_ascii=False)),
        ("returnContent", "count"), ("returnFormat", "json"),
    ]
    text = _post(url or api_url(), data)
    try:
        answer = json.loads(text)
    except ValueError:
        answer = {}
    if isinstance(answer, dict) and answer.get("error"):
        raise RedcapError(f"REDCap: {answer['error']}{_missing_fields_hint(str(answer['error']))}")
    try:
        return int(answer.get("count", 0))
    except (AttributeError, TypeError, ValueError):
        return 0


def mark_sent(project: SyncProject, kinds: list[str]) -> None:
    """Records in the checklist that these counts are in REDCap."""
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    for key in kinds:
        task = dict(project.tasks.get(key, {}))
        task["redcap_sent_at"] = now
        task["redcap_sent_by"] = project.technician
        project.tasks[key] = task


def needs_sending(project: SyncProject, key: str) -> bool:
    """A finished count with events that REDCap does not have yet, or not since its last change."""
    if key not in COUNT_FIELDS or project.task_state(key) != FINISHED or not project.events.get(key):
        return False
    task = project.tasks.get(key, {})
    sent, finished = task.get("redcap_sent_at", ""), task.get("finished_at", "")
    if not sent:
        return True
    try:
        return datetime.fromisoformat(sent) < datetime.fromisoformat(finished)
    except ValueError:
        return False
