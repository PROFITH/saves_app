"""
Synchronization state behind the window, with no Qt code.

It wraps the headless pipeline (clap detection and affine fit) and adds
what the interactive work needs: claps corrected or excluded by hand, video
claps marked by hand, the choice of reference and the mapping of every
clock (each sensor and the video) to one display timeline, the protocol
app's host clock in Unix seconds.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from algorithms.offset_search import OffsetSearch, estimate_offset, search_range_for
from models import Session, SyncAnchor
from pipeline.sync_report import (
    SensorSyncResult,
    anchors_table,
    coverage_metrics,
    detect_anchors,
    fit_against_reference,
    report_table,
)
from studio.project import SyncProject
from studio.signals import SignalData

VIDEO = "video"
# Seconds of signal kept around the session when looking for claps, so an
# offline clock that is minutes off still has its claps inside the crop.
CROP_MARGIN_S = 600.0

# Residual RMSE thresholds (ms) for the traffic light of each sensor.
GOOD_RMSE_MS = 25.0
FAIR_RMSE_MS = 80.0


def analyze_signal(
    session: Session,
    signal: SignalData,
    pad_seconds: float = 2.0,
    min_prominence: float = 1.0,
) -> tuple[dict[str, SyncAnchor], dict, OffsetSearch]:
    """
    Sync events per window, session coverage and coarse clock offset of one
    signal (thread-safe).

    The offset is searched first over the session plus a couple of hours on
    each side, so a clock that is far off still finds its claps and jumps.
    """
    session_start = session.start_time.timestamp() * 1000.0
    session_end = session.end_time.timestamp() * 1000.0
    search_s = search_range_for(signal.sensor_type)
    i0 = int(np.searchsorted(signal.t_ms, session_start - (search_s + 60.0) * 1000.0))
    i1 = int(np.searchsorted(signal.t_ms, session_end + (search_s + 60.0) * 1000.0, side="right"))
    search = estimate_offset(session, signal.t_ms[i0:i1], signal.channels["vm"][i0:i1], search_s)
    offset_ms = search.offset_ms if search.accepted else 0.0

    start = session_start + offset_ms - CROP_MARGIN_S * 1000.0
    end = session_end + offset_ms + CROP_MARGIN_S * 1000.0
    recording = signal.to_recording(start, end)
    if len(recording.data) < 3:
        empty = {"session_seconds": session.duration_seconds, "covered_seconds": 0.0,
                 "coverage_pct": 0.0, "n_gaps": 0, "longest_gap_seconds": 0.0}
        return {}, empty, search
    anchors = detect_anchors(session, recording, pad_seconds, min_prominence, offset_ms)
    return {a.event_id: a for a in anchors}, coverage_metrics(session, recording, offset_ms), search


@dataclass
class Affine:
    """target = a + b * (source - source_origin)."""

    a: float
    b: float
    source_origin: float = 0.0

    def __call__(self, source):
        return self.a + self.b * (np.asarray(source, dtype=float) - self.source_origin)

    def inverse(self, target):
        return self.source_origin + (np.asarray(target, dtype=float) - self.a) / self.b


@dataclass
class SensorStatus:
    sensor_id: str
    n_windows: int
    n_anchors: int
    n_pairs: int
    offset_ms: Optional[float]  # noqa: UP045
    drift_ppm: Optional[float]  # noqa: UP045
    rmse_ms: Optional[float]  # noqa: UP045
    max_abs_residual_ms: Optional[float]  # noqa: UP045
    coverage_pct: Optional[float]  # noqa: UP045
    quality: str  # "reference", "good", "fair", "poor", "none"
    note: str = ""


class SyncEngine:
    """Holds detected claps and fits for one project."""

    def __init__(self, project: SyncProject, session: Session):
        self.project = project
        self.session = session
        self.signals: dict[str, SignalData] = {}
        self.auto_anchors: dict[str, dict[str, SyncAnchor]] = {}
        self.coverage: dict[str, dict] = {}
        self.offset_searches: dict[str, OffsetSearch] = {}
        self.fits: dict[str, Optional[dict]] = {}  # noqa: UP045
        self.video_fit: Optional[Affine] = None  # noqa: UP045
        # Offline reference sensor -> host clock (from the cues), when no
        # Polar caught the events.
        self.reference_host_fit: Optional[Affine] = None  # noqa: UP045
        self.video_fit_source = "none"
        self.min_prominence = 1.0
        self.pad_seconds = 2.0

    # ------------------------------------------------------------------
    # Signals and detection
    # ------------------------------------------------------------------

    def set_signal(self, sensor_id: str, signal: SignalData) -> None:
        self.signals[sensor_id] = signal
        self.detect(sensor_id)

    def drop_signal(self, sensor_id: str) -> None:
        self.signals.pop(sensor_id, None)
        self.auto_anchors.pop(sensor_id, None)
        self.coverage.pop(sensor_id, None)
        self.fits.pop(sensor_id, None)
        self.offset_searches.pop(sensor_id, None)

    def detect(self, sensor_id: str) -> None:
        """Detect the claps and jumps of one sensor in every sync window."""
        anchors, coverage, search = analyze_signal(
            self.session, self.signals[sensor_id], self.pad_seconds, self.min_prominence
        )
        self.ingest(sensor_id, self.signals[sensor_id], anchors, coverage, search)

    def ingest(
        self,
        sensor_id: str,
        signal: SignalData,
        anchors: dict[str, SyncAnchor],
        coverage: dict,
        search: Optional[OffsetSearch] = None,  # noqa: UP045
    ) -> None:
        """Store a signal analysed elsewhere (in a background thread)."""
        self.signals[sensor_id] = signal
        self.auto_anchors[sensor_id] = anchors
        self.coverage[sensor_id] = coverage
        if search is not None:
            self.offset_searches[sensor_id] = search

    def detect_all(self) -> None:
        for sensor_id in self.signals:
            self.detect(sensor_id)

    # ------------------------------------------------------------------
    # Effective anchors (automatic + manual - excluded)
    # ------------------------------------------------------------------

    def anchor(self, sensor_id: str, event_id: str) -> Optional[SyncAnchor]:  # noqa: UP045
        manual = self.project.manual_anchors.get(sensor_id, {}).get(event_id)
        if manual is not None:
            window = self._window(event_id)
            return SyncAnchor(
                event_id=event_id,
                event_type=window.event_type if window else "manual_sync",
                phase=window.phase if window else "",
                sensor_timestamp_unix_ms=int(round(manual)),
                selection_rule="manual",
                prominence=float("nan"),
                peak_value=float("nan"),
                candidate_count=0,
            )
        return self.auto_anchors.get(sensor_id, {}).get(event_id)

    def anchor_state(self, sensor_id: str, event_id: str) -> str:
        """'manual', 'auto', 'excluded' or 'missing'."""
        if self.project.is_excluded(sensor_id, event_id):
            return "excluded"
        if event_id in self.project.manual_anchors.get(sensor_id, {}):
            return "manual"
        if event_id in self.auto_anchors.get(sensor_id, {}):
            return "auto"
        return "missing"

    def effective_anchors(self, sensor_id: str) -> list[SyncAnchor]:
        anchors = []
        for window in self.session.sync_windows:
            if self.project.is_excluded(sensor_id, window.event_id):
                continue
            anchor = self.anchor(sensor_id, window.event_id)
            if anchor is not None:
                anchors.append(anchor)
        return anchors

    def _window(self, event_id: str):
        for window in self.session.sync_windows:
            if window.event_id == event_id:
                return window
        return None

    # ------------------------------------------------------------------
    # Reference and fits
    # ------------------------------------------------------------------

    def active_sensors(self) -> list[str]:
        enabled = {s.sensor_id for s in self.project.sensors if s.enabled}
        return [sid for sid in self.signals if sid in enabled]

    def resolved_reference(self) -> Optional[str]:  # noqa: UP045
        """Sensor ID or ``"video"`` used as the reference timeline."""
        ref = self.project.reference
        active = self.active_sensors()
        if ref == VIDEO:
            return VIDEO if len(self.project.video_events) >= 2 else None
        if ref != "auto":
            return ref if ref in active else None
        # Prefer a live sensor already on the host clock (Polar): the one
        # that caught the most events, the chest strap on a tie.
        host = [sid for sid in self.host_clock_sensors() if len(self.effective_anchors(sid)) >= 2]
        if host:
            return max(host, key=lambda sid: (len(self.effective_anchors(sid)), sid == "polar_h10"))
        if len(self.project.video_events) >= 2:
            return VIDEO
        if not active:
            return None
        # No Polar caught the events (claps barely move the chest strap): the
        # sensor with the most events, put on the host clock by the cues.
        return max(active, key=lambda sid: len(self.effective_anchors(sid)))

    def host_clock_sensors(self) -> list[str]:
        """Active sensors recorded by the protocol app (on the host clock)."""
        return [sid for sid in self.active_sensors() if self.signals[sid].sensor_type == "polar_acc"]

    def reference_seconds(self) -> dict[str, float]:
        """
        Host time of each event in the reference timeline.

        With an automatic reference on the host clock, events the reference
        missed are taken from the other host-clock sensors (a clap barely
        moves the chest strap, but the arm band catches it).
        """
        ref = self.resolved_reference()
        if ref is None:
            return {}
        if ref == VIDEO:
            return dict(self.project.video_events)
        seconds = {a.event_id: a.sensor_timestamp_unix_ms / 1000.0 for a in self.effective_anchors(ref)}
        if self.project.reference == "auto" and ref in self.host_clock_sensors():
            for sid in self.host_clock_sensors():
                if sid == ref:
                    continue
                for anchor in self.effective_anchors(sid):
                    seconds.setdefault(anchor.event_id, anchor.sensor_timestamp_unix_ms / 1000.0)
        return seconds

    def reference_label(self) -> str:
        """Name of the reference, with the host-clock sensors that fill it."""
        ref = self.resolved_reference()
        if ref is None:
            return "ninguna"
        if ref == VIDEO:
            return "vídeo"
        if ref not in self.host_clock_sensors():
            return ref + (" (a la hora de los avisos)" if self.reference_host_fit is not None else "")
        extra = [sid for sid in self.host_clock_sensors() if sid != ref] if self.project.reference == "auto" else []
        return ref + (f" (+ {', '.join(extra)})" if extra else "")

    def refit(self) -> None:
        """Fit every sensor against the reference and the video to the host clock."""
        ref = self.resolved_reference()
        reference_seconds = self.reference_seconds()
        self.fits = {}
        for sid in self.active_sensors():
            if sid == ref:
                self.fits[sid] = None
                continue
            try:
                self.fits[sid] = fit_against_reference(self.effective_anchors(sid), reference_seconds)
            except (np.linalg.LinAlgError, ValueError):
                # Degenerate claps (e.g. the same impact found in two windows).
                self.fits[sid] = None
        self.reference_host_fit = self._fit_reference_to_host(ref)
        self._fit_video(ref)

    def _fit_reference_to_host(self, ref: Optional[str]) -> Optional[Affine]:  # noqa: UP045
        """
        Host time of an offline reference sensor, from the cues of the timer.

        The participant claps or lands a fixed delay after the go of each cue
        (give or take a reaction time), so fitting the reference's events to
        the cue times places its clock on the host clock within about a tenth
        of a second. The other sensors are fitted to the reference itself, so
        they stay aligned with each other to the millisecond.
        """
        if ref is None or ref == VIDEO or ref in self.host_clock_sensors():
            return None
        pairs = []
        for anchor in self.effective_anchors(ref):
            window = self._window(anchor.event_id)
            if window is not None and window.expected_time is not None:
                pairs.append((anchor.sensor_timestamp_unix_ms / 1000.0, window.expected_time.timestamp()))
        if len(pairs) >= 2:
            sensor = np.array([p[0] for p in pairs])
            host = np.array([p[1] for p in pairs])
            origin = float(sensor[0])
            # Slope 1: the cues are too noisy to tell a drift of a few ppm.
            return Affine(a=float(np.mean(host - (sensor - origin))), b=1.0, source_origin=origin)
        search = self.offset_searches.get(ref)
        if search is not None and search.accepted:
            return Affine(a=-search.offset_ms / 1000.0, b=1.0)
        return None

    def _video_start(self) -> Optional[float]:  # noqa: UP045
        if self.project.videos and self.project.videos[0].recording_start_unix_s is not None:
            return self.project.videos[0].recording_start_unix_s
        return None

    def _fit_video(self, ref: Optional[str]) -> None:  # noqa: UP045
        """Video seconds -> host Unix seconds."""
        events = self.project.video_events
        start = self._video_start()
        self.video_fit, self.video_fit_source = None, "none"

        if ref == VIDEO:
            # Sensors were fitted to video time; place video time on the host
            # clock with the recorded start (or the session start).
            origin = start if start is not None else self.session.start_time.timestamp()
            self.video_fit = Affine(a=origin, b=1.0)
            self.video_fit_source = "recording_start" if start is not None else "session_start"
            return

        host = {}
        if ref is not None:
            for event_id in events:
                t = self.display_time_of_anchor(event_id)
                if t is not None:
                    host[event_id] = t
        shared = [e for e in events if e in host]
        if len(shared) >= 2:
            v = np.array([events[e] for e in shared])
            h = np.array([host[e] for e in shared])
            b, a = np.polyfit(v, h, 1)
            self.video_fit, self.video_fit_source = Affine(a=float(a), b=float(b)), "claps"
        elif len(shared) == 1:
            e = shared[0]
            self.video_fit, self.video_fit_source = Affine(a=host[e] - events[e], b=1.0), "one_clap"
        elif start is not None:
            self.video_fit, self.video_fit_source = Affine(a=start, b=1.0), "recording_start"
        else:
            self.video_fit = Affine(a=self.session.start_time.timestamp(), b=1.0)
            self.video_fit_source = "session_start"

    def video_residuals_ms(self) -> dict[str, float]:
        if self.video_fit_source != "claps":
            return {}
        out = {}
        for event_id, v in self.project.video_events.items():
            t = self.display_time_of_anchor(event_id)
            if t is not None:
                out[event_id] = (t - float(self.video_fit(v))) * 1000.0
        return out

    # ------------------------------------------------------------------
    # Clock mapping
    # ------------------------------------------------------------------

    def _reference_to_display(self, seconds):
        """Reference timeline (s) -> display time (host Unix s)."""
        ref = self.resolved_reference()
        if ref == VIDEO and self.video_fit is not None:
            return self.video_fit(seconds)
        if self.reference_host_fit is not None and ref is not None and ref != VIDEO:
            return self.reference_host_fit(seconds)
        return seconds

    def _display_to_reference(self, display_s):
        ref = self.resolved_reference()
        if ref == VIDEO and self.video_fit is not None:
            return self.video_fit.inverse(display_s)
        if self.reference_host_fit is not None and ref is not None and ref != VIDEO:
            return self.reference_host_fit.inverse(display_s)
        return display_s

    def _coarse_offset_s(self, sensor_id: str) -> float:
        search = self.offset_searches.get(sensor_id)
        return search.offset_ms / 1000.0 if search is not None and search.accepted else 0.0

    def sensor_to_display(self, sensor_id: str, t_ms):
        """Native sensor time (Unix ms) -> display time (host Unix s)."""
        seconds = np.asarray(t_ms, dtype=float) / 1000.0
        fit = self.fits.get(sensor_id)
        if fit is None:
            if sensor_id == self.resolved_reference():
                return self._reference_to_display(seconds)
            # Not synchronized yet: at least without the coarse offset.
            return seconds - self._coarse_offset_s(sensor_id)
        mapped = fit["reference_origin_seconds"] + fit["intercept_seconds"] + fit["slope"] * (
            seconds - fit["sensor_origin_unix_seconds"]
        )
        return self._reference_to_display(mapped)

    def display_to_sensor(self, sensor_id: str, display_s):
        """Display time (host Unix s) -> native sensor time (Unix ms)."""
        display_s = np.asarray(display_s, dtype=float)
        fit = self.fits.get(sensor_id)
        if fit is None:
            if sensor_id == self.resolved_reference():
                return np.asarray(self._display_to_reference(display_s), dtype=float) * 1000.0
            return (display_s + self._coarse_offset_s(sensor_id)) * 1000.0
        reference = self._display_to_reference(display_s)
        seconds = fit["sensor_origin_unix_seconds"] + (
            reference - fit["reference_origin_seconds"] - fit["intercept_seconds"]
        ) / fit["slope"]
        return seconds * 1000.0

    def video_to_display(self, video_s):
        return self.video_fit(video_s) if self.video_fit is not None else np.asarray(video_s, dtype=float)

    def display_to_video(self, display_s):
        return self.video_fit.inverse(display_s) if self.video_fit is not None else np.asarray(display_s, dtype=float)

    def display_time_of_anchor(self, event_id: str) -> Optional[float]:  # noqa: UP045
        """Display time of a clap or jump, taken from the reference timeline."""
        ref = self.resolved_reference()
        if ref is None or ref == VIDEO:
            return None
        anchor = None if self.project.is_excluded(ref, event_id) else self.anchor(ref, event_id)
        if anchor is not None:
            return float(self.sensor_to_display(ref, anchor.sensor_timestamp_unix_ms))
        if ref in self.host_clock_sensors():
            # Filled in by another sensor on the host clock.
            return self.reference_seconds().get(event_id)
        return None

    def window_display_range(self, event_id: str) -> tuple[float, float]:
        window = self._window(event_id)
        return window.window_start.timestamp(), window.window_end.timestamp()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def residual_ms(self, sensor_id: str, event_id: str) -> Optional[float]:  # noqa: UP045
        fit = self.fits.get(sensor_id)
        if not fit:
            return None
        return fit["residuals_ms"].get(event_id)

    def status(self, sensor_id: str) -> SensorStatus:
        ref = self.resolved_reference()
        fit = self.fits.get(sensor_id)
        anchors = self.effective_anchors(sensor_id)
        coverage = self.coverage.get(sensor_id, {})
        if sensor_id == ref:
            quality, note = "reference", "Referencia"
            if self.reference_host_fit is not None:
                note = ("Referencia llevada a la hora del ordenador con los avisos del cronómetro "
                        "(± 0,1 s); los demás sensores quedan alineados con ella al milisegundo")
        elif fit is None:
            quality = "none"
            note = "Menos de 2 eventos (palmada o salto) compartidos con la referencia" if ref else "Sin referencia"
        else:
            rmse = fit["rmse_ms"]
            if fit["n_pairs"] < 3:
                quality, note = "fair", "Solo 2 eventos: no se puede comprobar el ajuste"
            elif rmse <= GOOD_RMSE_MS:
                quality, note = "good", ""
            elif rmse <= FAIR_RMSE_MS:
                quality, note = "fair", "Revisa los eventos con más residuo"
            else:
                quality, note = "poor", "Error alto: revisa los eventos"
        search = self.offset_searches.get(sensor_id)
        if search is not None and search.accepted and abs(search.offset_ms) >= 5000:
            note = (note + ". " if note else "") + f"Reloj desfasado {search.describe()}"
        return SensorStatus(
            sensor_id=sensor_id,
            n_windows=len(self.session.sync_windows),
            n_anchors=len(anchors),
            n_pairs=(fit or {}).get("n_pairs", 0) or 0,
            offset_ms=(fit or {}).get("offset_ms"),
            drift_ppm=(fit or {}).get("drift_ppm"),
            rmse_ms=(fit or {}).get("rmse_ms"),
            max_abs_residual_ms=(fit or {}).get("max_abs_residual_ms"),
            coverage_pct=coverage.get("coverage_pct"),
            quality=quality,
            note=note,
        )

    def worst_event(self, sensor_id: str) -> Optional[str]:  # noqa: UP045
        fit = self.fits.get(sensor_id)
        if not fit or not fit["residuals_ms"]:
            return None
        return max(fit["residuals_ms"], key=lambda e: abs(fit["residuals_ms"][e]))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _results(self) -> dict[str, SensorSyncResult]:
        ref = self.resolved_reference()
        results = {}
        for sid in self.active_sensors():
            signal = self.signals[sid]
            result = SensorSyncResult(
                sensor_id=sid,
                sensor_type=signal.sensor_type,
                sampling_rate_hz=signal.sampling_rate_hz,
                anchors=self.effective_anchors(sid),
                fit=self.fits.get(sid),
                coverage=self.coverage.get(sid, {}),
            )
            if sid == ref:
                result.note = "reference"
            elif result.fit is None:
                result.note = "fewer than two claps shared with the reference"
            results[sid] = result
        return results

    def activity_labels(self, sensor_id: str) -> pd.DataFrame:
        """Protocol activities expressed in the sensor's own clock."""
        rows = []
        for activity in self.session.activities:
            start, end = activity.start_time.timestamp(), activity.end_time.timestamp()
            rows.append({
                "activity_index": activity.activity_index,
                "activity_name": activity.activity_name,
                "compendium_code": activity.compendium_code,
                "start_host_utc": activity.start_time.isoformat(),
                "end_host_utc": activity.end_time.isoformat(),
                "start_sensor_unix_ms": float(self.display_to_sensor(sensor_id, start)),
                "end_sensor_unix_ms": float(self.display_to_sensor(sensor_id, end)),
            })
        return pd.DataFrame(rows)

    def export(self, out_dir: str | Path, export_aligned: bool = False) -> list[Path]:
        """Write the synchronization outputs; returns the files written."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        results = self._results()
        written = []

        report = report_table(self.session, results)
        report.insert(2, "location", [self._location(sid) for sid in report["sensor_id"]])
        report.insert(3, "serial", [self._serial(sid) for sid in report["sensor_id"]])
        report["reference"] = self.resolved_reference()
        written.append(out / "sync_report.csv")
        report.to_csv(written[-1], index=False)

        anchors = anchors_table(self.session, results)
        anchors["manual"] = [
            self.anchor_state(r.sensor_id, r.event_id) == "manual" for r in anchors.itertuples()
        ]
        written.append(out / "anchors.csv")
        anchors.to_csv(written[-1], index=False)

        video_rows = [
            {"event_id": e, "video_seconds": v, "residual_ms": self.video_residuals_ms().get(e)}
            for e, v in sorted(self.project.video_events.items(), key=lambda kv: kv[1])
        ]
        written.append(out / "video_events.csv")
        pd.DataFrame(video_rows, columns=["event_id", "video_seconds", "residual_ms"]).to_csv(written[-1], index=False)

        for sid in results:
            written.append(out / f"activities_{sid}.csv")
            self.activity_labels(sid).to_csv(written[-1], index=False)

        qc = {
            "session_id": self.session.session_id,
            "reference": self.resolved_reference(),
            "video": {
                "files": [v.path for v in self.project.videos],
                "mapping_source": self.video_fit_source,
                "host_seconds": (
                    {"a": self.video_fit.a, "b": self.video_fit.b} if self.video_fit else None
                ),
            },
            "sensors": {
                sid: {
                    "file": self.project.sensor(sid).path,
                    "location": self._location(sid),
                    "serial": self._serial(sid),
                    "sensor_type": r.sensor_type,
                    "sampling_rate_hz": r.sampling_rate_hz,
                    "fit": _clean(r.fit),
                    "coverage": r.coverage,
                    "manual_anchors": self.project.manual_anchors.get(sid, {}),
                    "excluded_anchors": self.project.excluded_anchors.get(sid, []),
                }
                for sid, r in results.items()
            },
        }
        written.append(out / "qc.json")
        written[-1].write_text(json.dumps(qc, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

        if export_aligned:
            for sid in results:
                signal = self.signals[sid]
                frame = pd.DataFrame({
                    "timestamp_native_unix_ms": signal.t_ms,
                    "timestamp_aligned_unix_ms": self.sensor_to_display(sid, signal.t_ms) * 1000.0,
                    **{c: signal.channels[c] for c in ("x", "y", "z")},
                })
                written.append(out / f"{sid}_aligned.csv.gz")
                frame.to_csv(written[-1], index=False)
        return written

    def _location(self, sid: str) -> str:
        try:
            return self.project.sensor(sid).location
        except KeyError:
            return ""

    def _serial(self, sid: str) -> str:
        try:
            return self.project.sensor(sid).serial
        except KeyError:
            return ""


def _clean(fit: Optional[dict]) -> Optional[dict]:  # noqa: UP045
    if fit is None:
        return None
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in fit.items()}
