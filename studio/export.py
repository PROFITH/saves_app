"""
Synchronized dataset of a session: every sensor on one clock, labelled.

After synchronizing, the export writes one file per sensor with its samples
on the host clock of the protocol app and, for every sample, the activity,
attempt and phase of the protocol, together with every other channel the
sensor file holds (gyroscope, light, temperature, heart rate); the other
signals of the live Polar devices (ECG, PPG, gyroscope, magnetometer,
heart rate and pulse intervals), on the clock fit of the device's
acceleration; the activities and phases with their times; the steps and
any other events counted on the video; and the technician's task
checklist. Every file shares the time column ``time_unix_ms`` (host
clock), so they can be joined directly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from readers.polar import (
    POLAR_COMPANION_CHANNELS,
    POLAR_COMPANION_KINDS,
    polar_file_parts,
    read_polar_companion,
)
from studio.discovery import SKIP_DIRS
from studio.project import STEPS, SYNC_TASK, TASK_STATE_TEXT, SyncProject
from studio.signals import SignalData

if TYPE_CHECKING:
    from studio.engine import SyncEngine

# Seconds of signal kept before the start and after the end of the session.
EXPORT_MARGIN_S = 60.0
GZIP = {"method": "gzip", "compresslevel": 1}
# Decimals written for the channels that are not integers: degrees Celsius
# and lux with 2, everything else (g, dps, gauss) with 4.
_DECIMALS = {"temperature_c": 2, "temperature_ambient_c": 2, "light": 2}
_TEXT_CHANNELS = {"rr_intervals_ms", "hr_source"}

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
archivos se pueden unir directamente por esa columna. Cada dispositivo
tiene un solo reloj: el ajuste (desfase y deriva) que SAVES encuentra con
las palmadas y saltos de su acelerómetro se aplica igual a todas las
demás señales del mismo dispositivo.

Etiquetas, al final de cada archivo con muestras (vacías fuera del
protocolo):
    activity_index       número de la actividad del protocolo
    activity             nombre de la actividad
    attempt              intento (1, o más si se repitió)
    phase                static_prep (quieto antes), activity_main
                         (actividad), static_post (quieto después),
                         transition_lap (transición) o paused (pausa)

signals/<sensor>.csv.gz
    Una fila por muestra del acelerómetro del sensor, ya sincronizada,
    con todas las señales que guarda su archivo (solo las columnas que
    tiene ese dispositivo):
    time_unix_ms         hora sincronizada (reloj del cronómetro)
    time_native_unix_ms  hora que escribió el propio sensor
    x, y, z              aceleración en g
    gyro_x, gyro_y, gyro_z
                         giroscopio en grados por segundo (Axivity AX6,
                         Parmay Matrix)
    light_raw            Axivity: luz en cuentas del sensor (0-1023), sin
                         calibrar
    light                GENEActiv: luz en lux
    lux                  ActiGraph: luz en lux, un valor por segundo que
                         se mantiene hasta el siguiente registro (solo si
                         el archivo trae registros LUX)
    temperature_c        temperatura en °C: Axivity (cuentas × 75/256 −
                         50, la fórmula del fabricante; vacía cuando el
                         sector no trae lectura), GENEActiv (la de cada
                         página) o Parmay Matrix (la de la superficie del
                         cuerpo)
    temperature_ambient_c
                         Parmay Matrix: temperatura ambiente en °C
    button               GENEActiv: botón pulsado (1) o no (0)
    imputed              ActiGraph: 1 en las muestras rellenadas mientras
                         el dispositivo dormía (idle sleep), 0 en las
                         grabadas
    hr_bpm, hr_source, hr_raw
                         Parmay Matrix: frecuencia cardiaca en latidos por
                         minuto que calcula el firmware del dispositivo
                         con su sensor óptico (vacía cuando no da valor),
                         su origen (matrix_ppg_firmware) y el valor crudo
                         que el dispositivo guarda junto a ella (el
                         fabricante no documenta su unidad)
    Las demás columnas numéricas de un CSV de SENS motion se conservan
    con su nombre en minúsculas.
    Las señales que el dispositivo mide menos veces que el acelerómetro
    (luz y temperatura de Axivity, una vez por sector de ~1 s;
    temperatura de GENEActiv, una por página de 300 muestras;
    temperatura y frecuencia cardiaca de Matrix, ~1 por segundo) se
    repiten en todas las muestras de su sector, página o paquete.

Dispositivos Polar (H10 y Verity Sense), grabados en directo por el
cronómetro. Su acelerómetro es signals/<sensor>.csv.gz; sus demás señales
van en archivos aparte con el mismo <sensor> y el mismo ajuste de reloj.
En ellos time_native_unix_ms es la hora del cronómetro que escribió la
app al recibir cada muestra, antes del ajuste.
signals/<sensor>_ecg.csv.gz
    H10: ecg_uv, electrocardiograma en microvoltios (130 Hz).
signals/<sensor>_ppg.csv.gz
    Verity Sense: ppg0, ppg1, ppg2, cuentas crudas (22 bits) de los tres
    canales verdes del sensor óptico, y ambient, el canal de luz
    ambiente (55 Hz).
signals/<sensor>_gyro.csv.gz
    Verity Sense: gyro_x, gyro_y, gyro_z en grados por segundo (52 Hz).
signals/<sensor>_mag.csv.gz
    Verity Sense: mag_x, mag_y, mag_z en gauss (50 Hz).
heart_rate/<sensor>.csv
    Frecuencia cardiaca del servicio BLE del dispositivo, una fila por
    segundo (cada 5 s en el Verity Sense mientras mide PPI): hr_bpm
    (latidos por minuto), rr_intervals_ms (intervalos RR en ms separados
    por «;», solo en el H10; vacío en el Verity Sense) y hr_source.
heart_rate/<sensor>_ppi.csv
    Verity Sense: intervalos entre pulsos del algoritmo PPI de Polar, una
    fila por pulso: ppi_ms (intervalo en ms), hr_bpm (la frecuencia que
    da el algoritmo), error_estimate_ms (error estimado del intervalo),
    blocker (1 cuando el movimiento invalida la medida), skin_contact y
    skin_contact_supported (contacto con la piel; poco fiables en el
    Verity), index_in_frame y hr_source. El dispositivo entrega los
    pulsos en lotes y el primero llega unos 25 s después de empezar:
    cuando el sensor no fecha cada pulso, todos los de un lote comparten
    time_unix_ms (la llegada del lote) e index_in_frame (0, 1, 2…) los
    ordena dentro del lote.

hr_source dice de dónde sale cada frecuencia cardiaca (SAVES no calcula
ninguna; todas vienen del dispositivo):
    h10_ecg_firmware     el H10 la calcula de su ECG
    verity_ppg_firmware  el Verity Sense la calcula de su PPG
    verity_ppi_firmware  el algoritmo PPI del Verity Sense (archivo _ppi)
    matrix_ppg_firmware  el firmware del Parmay Matrix, con su sensor
                         óptico
    polar_firmware       un Polar de modelo desconocido
    En archivos antiguos del cronómetro sin esa columna se deduce del
    tipo de dispositivo del manifiesto o del nombre del archivo.

Si el acelerómetro de un Polar no está entre los sensores sincronizados
(falta su archivo o se dejó fuera de la sincronización), sus demás
archivos se exportan igualmente con la hora del cronómetro tal como la
escribió la app, sin ajuste, y con el <id> del nombre del archivo; el
encabezado de este LEEME dice cuáles.

activities.csv   Actividades, una fila por intento (attempt), con su
                 inicio y fin, MET, RPE, la instancia de lab_activity en
                 REDCap (redcap_instance) y lo contado en el vídeo en cada
                 una (steps_video, <tipo>_video).
phases.csv       Cada fase de cada actividad con su inicio y fin.
steps.csv        Pasos marcados en el vídeo (si se han contado), con el
                 pie (foot: L izquierdo, R derecho, vacío si no se
                 distinguió).
steps_per_activity.csv
                 Pasos de cada intento de cada actividad y cadencia
                 (pasos/min).
events_<tipo>.csv
                 Otros eventos contados en el vídeo, uno por fila, con
                 las mismas columnas que steps.csv (event en lugar de
                 step y side en lugar de foot). Por ejemplo,
                 events_sit_to_stand.csv: cada vez que se queda de pie
                 en el test de sentarse y levantarse de 30 s.
events_<tipo>_per_activity.csv
                 Esos eventos en cada intento de cada actividad:
                 <tipo>_main (en la actividad), <tipo>_all_phases (con
                 las fases quietas) y <tipo>_per_min (por minuto de
                 actividad).
tasks.csv        Tareas del técnico: la sincronización y cada conteo, con
                 su estado (not_started no iniciada, started iniciada,
                 finished finalizada), quién la hizo y cuándo, y cuándo
                 se envió el conteo a REDCap (redcap_sent_at).
redcap_conteos.csv
                 Lo que «Enviar a REDCap» manda a lab_activity (si se ha
                 usado): los pasos (la_steps_video) y las veces que se
                 levanta en el test de sentarse y levantarse
                 (la_reps_video) de cada intento, con el técnico que
                 contó. Sirve también para subirlo a mano con Data
                 Import Tool de REDCap.
sync_report.csv  Calidad de la sincronización de cada sensor.
anchors.csv      Palmadas y saltos encontrados en cada sensor.
qc.json          Todos los ajustes (desfase, deriva, error) en detalle.
<sensor>_aligned.csv.gz
                 Solo si se pidió: la grabación completa del sensor, con
                 todas sus señales, timestamp_native_unix_ms y
                 timestamp_aligned_unix_ms (la hora sincronizada).
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
                "attempt": a.attempt,
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


def _rounded(name: str, values) -> object:
    """A channel as a CSV column: integers stay integers (empty when missing), the rest rounded to their unit."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or (np.abs(finite).max() < 2**53 and np.all(finite == np.round(finite))):
        try:
            return pd.array(values, dtype="Int64")
        except (TypeError, ValueError):
            pass
    return np.round(values, _DECIMALS.get(name, 4))


def signal_columns(signal: SignalData, i0: int = 0, i1: Optional[int] = None) -> dict[str, object]:  # noqa: UP045
    """
    Every channel of a signal but ``vm``, between two sample indices: x, y,
    z first, then the extras in the order the reader gave them, each
    rounded to its unit; a heart rate is followed by its ``hr_source``.
    """
    i1 = signal.t_ms.size if i1 is None else i1
    columns: dict[str, object] = {}
    for name in ("x", "y", "z", *signal.extra_channels):
        columns[name] = _rounded(name, signal.channels[name][i0:i1])
        if name == "hr_bpm" and signal.metadata.get("hr_source"):
            columns["hr_source"] = str(signal.metadata["hr_source"])
    return columns


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
            **signal_columns(signal, i0, i1),
            **label_times(host, phases, names),
        })
        path = out / "signals" / f"{_safe(sid)}.csv.gz"
        frame.to_csv(path, index=False, compression=GZIP)
        written.append(path)

    polar_files, identity_fit = export_polar_companions(engine, out, phases, names, start, end)
    written += polar_files

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
    written[-1].write_text(readme(engine.project, identity_fit), encoding="utf-8")
    return written


def readme(project: SyncProject, identity_fit: Sequence[str] = ()) -> str:
    """The guide to the files, headed by who exported them, what was left and which Polar devices kept the host clock."""
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
    if identity_fit:
        head += (
            "Dispositivos Polar sin acelerómetro sincronizado, exportados con la hora del cronómetro "
            "tal como la escribió la app (sin ajuste): " + ", ".join(identity_fit) + ".\n"
        )
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
            "redcap_sent_at": task.get("redcap_sent_at", ""),
        })
    return pd.DataFrame(rows)


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text) or "sensor"


# ----------------------------------------------------------------------
# Companion files of the live Polar devices
# ----------------------------------------------------------------------

def polar_companion_files(folder: str | Path) -> list[tuple[str, str, str, Path]]:
    """``(kind, id, stamp, path)`` of every Polar companion file in ``folder`` and its direct subfolders."""
    folder = Path(folder)
    candidates = sorted(folder.glob("polar_*.csv"))
    for sub in sorted(p for p in folder.iterdir() if p.is_dir() and p.name not in SKIP_DIRS):
        candidates += sorted(sub.glob("polar_*.csv"))
    found = []
    for path in candidates:
        parts = polar_file_parts(path)
        if parts is not None and parts[0] in POLAR_COMPANION_KINDS:
            found.append((*parts, path))
    return found


def _same_device(acc: tuple[str, str, str], companion: tuple[str, str, str, Path]) -> bool:
    """The companion belongs to the device of the acceleration file: same ``<id>`` (and ``<stamp>``, when both have one)."""
    return companion[1] == acc[1] and (not acc[2] or not companion[2] or companion[2] == acc[2])


def _manifest_sensors(manifest_path: Optional[str]) -> dict:  # noqa: UP045
    if not manifest_path:
        return {}
    try:
        sensors = json.loads(Path(manifest_path).read_text(encoding="utf-8")).get("sensors")
    except (OSError, ValueError, AttributeError):
        return {}
    return sensors if isinstance(sensors, dict) else {}


def _device_hint(sensor: Optional[dict], *names: str) -> str:  # noqa: UP045
    """Text naming a Polar device for ``infer_hr_source``: the manifest type and name, then the IDs."""
    fields = [str(sensor.get(key) or "") for key in ("type", "device_type", "device_name")] if sensor else []
    return " ".join(part for part in (*fields, *names) if part)


def _unique(target: Path, used: set[Path]) -> Path:
    """``target``, or ``<name>_2``... when this export already wrote it (two devices sharing one ``<id>``)."""
    extension = ".csv.gz" if target.name.endswith(".csv.gz") else target.suffix
    base = target.name[:-len(extension)]
    candidate, n = target, 2
    while candidate in used:
        candidate, n = target.with_name(f"{base}_{n}{extension}"), n + 1
    used.add(candidate)
    return candidate


def _write_companion(
    path: Path,
    kind: str,
    name: str,
    fit: Callable,
    hint: str,
    out: Path,
    phases: pd.DataFrame,
    names: dict[int, str],
    start: float,
    end: float,
    used: set[Path],
) -> list[Path]:
    """One companion file on the host clock (``fit``: native Unix ms -> host s), labelled; [] if unreadable or empty."""
    try:
        data = read_polar_companion(path, kind, hr_source_hint=hint)
    except (OSError, ValueError, pd.errors.ParserError):
        return []
    if data.empty:
        return []
    native_ms = data["timestamp_unix_ms"].to_numpy(dtype=float)
    host = np.asarray(fit(native_ms), dtype=float)
    keep = (host >= start) & (host <= end)
    data, host, native_ms = data.loc[keep].reset_index(drop=True), host[keep], native_ms[keep]

    columns: dict[str, object] = {"time_unix_ms": np.round(host * 1000.0).astype(np.int64)}
    if kind in ("hr", "ppi"):
        target = out / "heart_rate" / (f"{_safe(name)}.csv" if kind == "hr" else f"{_safe(name)}_ppi.csv")
    else:
        columns["time_native_unix_ms"] = np.round(native_ms).astype(np.int64)
        target = out / "signals" / f"{_safe(name)}_{kind}.csv.gz"
    for channel in POLAR_COMPANION_CHANNELS[kind].values():
        columns[channel] = data[channel] if channel in _TEXT_CHANNELS else _rounded(channel, data[channel])
    frame = pd.DataFrame({**columns, **label_times(host, phases, names)})

    target = _unique(target, used)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, compression=GZIP if target.name.endswith(".gz") else None)
    return [target]


def export_polar_companions(
    engine: SyncEngine,
    out: Path,
    phases: pd.DataFrame,
    names: dict[int, str],
    start: float,
    end: float,
) -> tuple[list[Path], list[str]]:
    """
    The other signals of every live Polar device (heart rate, PPI, ECG,
    PPG, gyroscope, magnetometer) on the host clock, labelled.

    A device synchronized through its acceleration lends its clock fit to
    its companion files (those with the ``<id>`` and ``<stamp>`` of the
    acceleration file name) and names them with its sensor ID. The
    companions of a device whose acceleration is not among the
    synchronized sensors (no file, disabled or left out) keep the host
    clock as written (identity fit) and are named by their ``<id>``;
    those IDs are returned so the LEEME can say so.
    """
    companions = polar_companion_files(engine.project.folder)
    manifest_sensors = _manifest_sensors(engine.project.manifest_path)
    written: list[Path] = []
    used: set[Path] = set()
    claimed: set[Path] = set()

    for sid in engine.active_sensors():
        try:
            entry = engine.project.sensor(sid)
        except KeyError:
            continue
        acc = polar_file_parts(entry.path)
        if acc is None or acc[0] != "acc":
            continue
        hint = _device_hint(manifest_sensors.get(sid) or manifest_sensors.get(acc[1]), sid, acc[1])

        def fit(native_ms, sid=sid):
            return engine.sensor_to_display(sid, native_ms)

        for companion in companions:
            kind, _token, _stamp, path = companion
            if path in claimed or not _same_device(acc, companion):
                continue
            claimed.add(path)
            written += _write_companion(path, kind, sid, fit, hint, out, phases, names, start, end, used)

    identity: list[str] = []
    for kind, token, _stamp, path in companions:
        if path in claimed:
            continue
        hint = _device_hint(manifest_sensors.get(token), token, path.stem)
        files = _write_companion(
            path, kind, token, lambda ms: np.asarray(ms, dtype=float) / 1000.0, hint, out, phases, names, start, end, used
        )
        written += files
        if files and token not in identity:
            identity.append(token)
    return written, identity


# ----------------------------------------------------------------------
# Events counted on the video
# ----------------------------------------------------------------------

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


def _of_attempt(frame: pd.DataFrame, index: int, attempt: int) -> pd.Series:
    """Rows of ``frame`` that belong to one attempt of an activity."""
    same = (frame["activity_index"] == index) & (frame["attempt"] == attempt)
    return same.fillna(False).astype(bool)


def events_per_activity(engine: SyncEngine, key: str, events: pd.DataFrame, phases: pd.DataFrame,
                        names: dict[int, str]) -> pd.DataFrame:
    """Events of each attempt of each activity (main phase and all its phases) and their rate per minute."""
    main_col, all_col, rate_col = _count_columns(key)
    video_range = engine.video_display_range()
    rows = []
    for activity in engine.session.activities:
        index, attempt = activity.activity_index, activity.attempt
        main = phases[_of_attempt(phases, index, attempt) & (phases["phase"] == "activity_main")]
        main_s = float((main["end_s"] - main["start_s"]).sum()) if len(main) else activity.duration_seconds
        in_activity = events[_of_attempt(events, index, attempt)] if len(events) else events
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
            "attempt": attempt,
            "redcap_instance": activity.redcap_instance,
            "activity": names.get(index, ""),
            main_col: int(len(in_main)),
            all_col: int(len(in_activity)),
            "main_duration_s": round(main_s, 3),
            rate_col: round(60.0 * len(in_main) / main_s, 1) if main_s > 0 else None,
            "video_coverage_pct": covered,
        })
    return pd.DataFrame(rows)


def steps_per_activity(engine: SyncEngine, steps: pd.DataFrame, phases: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    """Steps of each attempt of each activity (main phase and all its phases) and cadence."""
    return events_per_activity(engine, STEPS, steps, phases, names)


def activity_table(engine: SyncEngine, folder: Path, counted: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Activities with their times, the timer's markers and the events counted on the video."""
    rows = []
    for a in engine.session.activities:
        rows.append({
            "activity_index": a.activity_index,
            "attempt": a.attempt,
            "redcap_instance": a.redcap_instance,
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
        keys = [c for c in ("activity_index", "attempt") if c in markers.columns]
        extra = [c for c in ("completion", "end_reason", "paused_seconds", "rpe_before",
                             "resting_hr_bpm", "incident") if c in markers.columns]
        if "activity_index" in keys and extra:
            last = markers.drop_duplicates(keys, keep="last")[[*keys, *extra]]
            frame = frame.merge(last, on=keys, how="left")
    for key, events in counted.items():
        if len(events) and len(frame):
            main = events[events["phase"] == "activity_main"].groupby(["activity_index", "attempt"]).size()
            frame[f"{key}_video"] = [int(main.get((i, a), 0)) for i, a in zip(frame["activity_index"], frame["attempt"])]
    return frame
