"""
Find the files of a session and guess what each one is.

lab_protocol_timer writes the manifest, the Polar CSVs, the video and a
``video_sync_session_*.csv`` with the host time at which each video started.
Offline sensor files (.cwa, .bin, SENS .csv) are copied next to them, or
dropped on the window from anywhere.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from readers import infer_format
from readers.cwa import _parse_header
from studio.project import SensorEntry, VideoEntry

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
SENSOR_SUFFIXES = (".cwa", ".bin", ".csv")
SKIP_DIRS = {"saves_sync", "saves_export", ".saves_cache", "__pycache__"}

FORMAT_LABELS = {
    "cwa": "Axivity",
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

_SENSOR_ID_BY_FORMAT = {"cwa": "ax", "matrix": "matrix", "sens": "sens", "polar_acc": "h10"}


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
    if sensor_format == "polar_acc":
        return "Pecho"
    stem = re.sub(r"non?[_\- ]?domin\w*", "nondom", _normalize(Path(file_name).stem))
    tokens = set(re.split(r"[^a-z0-9]+", stem))
    for words, location in _LOCATION_HINTS:
        if tokens.intersection(words):
            return location
    if sensor_format == "sens":
        return "Muslo"
    if sensor_format == "matrix":
        return "Muñeca no dominante"
    return "Otra"


def read_serial(path: Path, sensor_format: str) -> str:
    """Read the device serial from the file header when the format has one."""
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
    if sensor_format == "cwa":
        short = {
            "Muñeca no dominante": "wrist_nd",
            "Muñeca dominante": "wrist_d",
            "Muslo": "thigh",
            "Calzado": "shoe",
            "Cadera": "hip",
        }.get(location)
        base = f"ax_{short}" if short else "ax"
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
        elif name.startswith(("polar_hr_", "polar_ecg_", "redcap_records")):
            found.ignored.append(str(path))
        else:
            entry = sensor_entry_for(path)
            if entry is not None:
                found.sensors.append(entry)
            elif path.suffix.lower() in SENSOR_SUFFIXES:
                found.ignored.append(str(path))

    found.videos = [VideoEntry(path=str(v), recording_start_unix_s=video_starts.get(v.name)) for v in videos]
    return found
