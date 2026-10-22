"""
Find the files of a session and guess what each one is.

lab_protocol_timer writes the manifest, the Polar CSVs, the video and a
``video_sync_session_*.csv`` with the host time at which each video started.
Offline sensor files (Axivity .cwa, ActiGraph .gt3x, GENEActiv or Parmay
Matrix .bin, SENS .csv) are copied next to them, or dropped on the window
from anywhere. When the manifest lists the sensors (serial, placement),
that information wins over guesses from the file name.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from readers import infer_format
from readers.cwa import _parse_header
from readers.geneactiv import read_geneactiv_serial
from readers.gt3x import read_gt3x_serial
from readers.polar import POLAR_COMPANION_KINDS, POLAR_LOG_KINDS
from studio.project import SensorEntry, VideoEntry

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
SENSOR_SUFFIXES = (".cwa", ".gt3x", ".bin", ".csv")
SKIP_DIRS = {"saves_sync", "saves_export", ".saves_cache", "__pycache__"}
# Files of a live Polar device other than its acceleration (heart rate, ECG,
# PPG, PPI, gyroscope, magnetometer, clock fit, connection log): exported
# with the device's clock fit, never sensors of their own.
POLAR_COMPANION_PREFIXES = tuple(f"polar_{kind}_" for kind in (*POLAR_COMPANION_KINDS, *POLAR_LOG_KINDS))

FORMAT_LABELS = {
    "cwa": "Axivity",
    "gt3x": "ActiGraph",
    "geneactiv": "GENEActiv",
    "matrix": "Parmay Matrix",
    "sens": "SENS motion",
    "polar_acc": "Polar (ACC)",
}

# Words in file names that give away where a sensor was worn.
_LOCATION_HINTS = (
    (("thigh", "muslo", "leg", "pierna"), "Muslo"),
    (("wnd", "nondom", "non_dom", "nodom", "no_dom", "izq", "left", "nd"), "Muñeca no dominante"),
    (("wd", "dom", "dominante", "dominant", "der", "derecha", "right"), "Muñeca dominante"),
    (("wrist", "muneca", "mun"), "Muñeca no dominante"),
    (("shoe", "calzado", "zapatilla", "foot", "pie", "cordon"), "Calzado"),
    (("chest", "pecho", "h10"), "Pecho"),
    (("arm", "brazo", "verity"), "Brazo"),
    (("hip", "cadera", "waist"), "Cadera"),
)

_SENSOR_ID_BY_FORMAT = {
    "cwa": "ax",
    "gt3x": "gt3x",
    "geneactiv": "geneactiv",
    "matrix": "matrix",
    "sens": "sens",
    "polar_acc": "h10",
}


@dataclass
class Discovery:
    """What was found in a folder."""

    manifest_path: Optional[str] = None  # noqa: UP045
    videos: list[VideoEntry] = field(default_factory=list)
    sensors: list[SensorEntry] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return text.lower()


def guess_location(file_name: str, sensor_format: str) -> str:
    """Guess the body location from tokens in the file name."""
    stem = re.sub(r"non?[_\- ]?domin\w*", "nondom", _normalize(Path(file_name).stem))
    tokens = set(re.split(r"[^a-z0-9]+", stem))
    if sensor_format == "polar_acc":
        # Polar H10 on the chest unless the name says Verity Sense (arm).
        return "Brazo" if tokens.intersection(("verity", "arm", "brazo")) else "Pecho"
    for words, location in _LOCATION_HINTS:
        if tokens.intersection(words):
            return location
    if sensor_format == "sens":
        return "Muslo"
    if sensor_format in ("matrix", "geneactiv"):
        return "Muñeca no dominante"
    return "Otra"


def read_serial(path: Path, sensor_format: str) -> str:
    """Read the device serial from the file header when the format has one."""
    if sensor_format == "gt3x":
        return read_gt3x_serial(path)
    if sensor_format == "geneactiv":
        return read_geneactiv_serial(path)
    if sensor_format != "cwa":
        return ""
    try:
        header = _parse_header(np.fromfile(path, dtype=np.uint8, count=1024))
    except (OSError, ValueError):
        return ""
    return f"{header['device_type']}-{header['device_id']}"


def classify_file(path: Path) -> Optional[str]:  # noqa: UP045
    """Return the sensor format of ``path``, or None if it is not a sensor file."""
    if path.suffix.lower() not in SENSOR_SUFFIXES:
        return None
    if path.suffix.lower() == ".csv":
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                header = f.readline().strip().lower()
        except OSError:
            return None
        columns = {c.strip() for c in header.split(",")}
        if {"timestamp_iso", "acc_x_mg"}.issubset(columns):
            return "polar_acc"
        if {"unixts", "x", "y", "z"}.issubset(columns):
            return "sens"
        return None
    try:
        return infer_format(path)
    except ValueError:
        return None


def sensor_entry_for(path: Path, sensor_format: Optional[str] = None) -> Optional[SensorEntry]:  # noqa: UP045
    sensor_format = sensor_format or classify_file(path)
    if sensor_format is None:
        return None
    location = guess_location(path.name, sensor_format)
    base = _SENSOR_ID_BY_FORMAT[sensor_format]
    if sensor_format in ("cwa", "gt3x", "geneactiv"):
        short = {
            "Muñeca no dominante": "wrist_nd",
            "Muñeca dominante": "wrist_d",
            "Muslo": "thigh",
            "Calzado": "shoe",
            "Cadera": "hip",
        }.get(location)
        base = f"{base}_{short}" if short else base
    return SensorEntry(
        sensor_id=base,
        path=str(path),
        sensor_format=sensor_format,
        location=location,
        serial=read_serial(path, sensor_format),
    )


def read_video_log(path: Path) -> dict[str, float]:
    """Map video file names to the host time (Unix s) when recording started."""
    starts = {}
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        for row in csv.DictReader(f):
            name = Path((row.get("file_path") or "").replace("\\", "/")).name
            start = row.get("recording_start_time_utc") or ""
            if not name or not start:
                continue
            try:
                starts[name] = datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
    return starts


def discover(folder: str | Path) -> Discovery:
    """Scan ``folder`` (and its direct subfolders) for session files."""
    folder = Path(folder)
    found = Discovery()
    video_starts: dict[str, float] = {}
    videos: list[Path] = []

    candidates = sorted(p for p in folder.iterdir() if p.is_file())
    for sub in sorted(p for p in folder.iterdir() if p.is_dir() and p.name not in SKIP_DIRS):
        candidates += sorted(p for p in sub.iterdir() if p.is_file())

    for path in candidates:
        name = path.name.lower()
        if name == "session_manifest.json":
            found.manifest_path = found.manifest_path or str(path)
        elif path.suffix.lower() in VIDEO_SUFFIXES:
            videos.append(path)
        elif name.startswith("video_sync_session") and name.endswith(".csv"):
            video_starts.update(read_video_log(path))
        elif name.startswith((*POLAR_COMPANION_PREFIXES, "redcap_records")):
            found.ignored.append(str(path))
        else:
            entry = sensor_entry_for(path)
            if entry is not None:
                found.sensors.append(entry)
            elif path.suffix.lower() in SENSOR_SUFFIXES:
                found.ignored.append(str(path))

    if found.manifest_path:
        apply_manifest_sensors(found.sensors, found.manifest_path)

    found.videos = [VideoEntry(path=str(v), recording_start_unix_s=video_starts.get(v.name)) for v in videos]
    return found


_PLACEMENT_LABELS = {
    "wrist": "Muñeca",
    "wrist_non_dominant": "Muñeca no dominante",
    "wrist_dominant": "Muñeca dominante",
    "thigh": "Muslo",
    "hip": "Cadera",
    "chest": "Pecho",
    "upper_arm": "Brazo",
    "ankle": "Tobillo",
    "shoe": "Calzado",
    "shoe_left": "Calzado izquierdo",
    "shoe_right": "Calzado derecho",
    "room": "Sala",
}
_DOMINANCE_LABELS = {"dominant": "dominante", "non_dominant": "no dominante"}
_SIDE_LABELS = {"left": "izquierda", "right": "derecha"}


def manifest_location(sensor: dict) -> Optional[str]:  # noqa: UP045
    """Location label of a manifest sensor (placement, dominance and side)."""
    label = sensor.get("placement_label")
    if label:
        return str(label)
    placement = sensor.get("placement")
    if not placement:
        return None
    text = _PLACEMENT_LABELS.get(str(placement), str(placement))
    if sensor.get("dominance") in _DOMINANCE_LABELS:
        text += f" {_DOMINANCE_LABELS[sensor['dominance']]}"
    if sensor.get("side") in _SIDE_LABELS:
        text += f" ({_SIDE_LABELS[sensor['side']]})"
    return text


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def apply_manifest_sensors(entries: list[SensorEntry], manifest_path: str | Path) -> None:
    """
    Take sensor IDs and locations from the manifest. A file matches a
    manifest sensor by serial number (same digits) or by the sensor ID
    appearing in its file name.
    """
    try:
        sensors = json.loads(Path(manifest_path).read_text(encoding="utf-8")).get("sensors") or {}
    except (OSError, ValueError, AttributeError):
        return
    if not isinstance(sensors, dict):
        return
    used: set[str] = set()

    def assign(entry: SensorEntry, sensor_id: str) -> None:
        used.add(sensor_id)
        entry.sensor_id = sensor_id
        entry.location = manifest_location(sensors[sensor_id]) or entry.location

    # Serial numbers first, so a file name cannot take another file's sensor.
    pending = []
    for entry in entries:
        serial = _digits(entry.serial)
        match = next(
            (
                sensor_id
                for sensor_id, sensor in sensors.items()
                if sensor_id not in used
                and isinstance(sensor, dict)
                and serial
                and (manifest_serial := _digits(str(sensor.get("serial") or "")))
                and (serial.endswith(manifest_serial) or manifest_serial.endswith(serial))
            ),
            None,
        )
        if match:
            assign(entry, match)
        else:
            pending.append(entry)

    for entry in pending:
        stem = _normalize(Path(entry.path).stem)
        match = next(
            (
                sensor_id
                for sensor_id, sensor in sensors.items()
                if sensor_id not in used
                and isinstance(sensor, dict)
                and re.search(rf"(^|[^a-z0-9]){re.escape(sensor_id.lower())}($|[^a-z0-9])", stem)
            ),
            None,
        )
        if match:
            assign(entry, match)
