"""
Headless synchronization of offline sensors against a reference timeline.

For each sensor the synchronization claps are detected inside the windows
logged by the protocol app, and an affine model (offset + drift) is fitted
against the same claps in a reference timeline: another sensor (usually the
Polar H10, whose timestamps are already mapped to the host clock) or manual
video events. The residuals of that fit are the synchronization error
reported for the session.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from algorithms.clap_detector import ClapDetector
from algorithms.sync_event_detector import SyncEventDetector
from algorithms.sync_model import SyncModel
from models import Session, SyncAnchor, SyncAnchorPair
from readers.recording import TIMESTAMP_COLUMN, SensorRecording
from utils.sync_window_extractor import SyncWindowExtractor

GAP_FACTOR = 3.0


@dataclass
class SensorSyncResult:
    """Anchors, fit and data-quality metrics of one sensor."""

    sensor_id: str
    sensor_type: str
    sampling_rate_hz: float
    anchors: list[SyncAnchor] = field(default_factory=list)
    fit: Optional[dict] = None  # noqa: UP045
    coverage: dict = field(default_factory=dict)
    note: str = ""


def detect_anchors(
    session: Session,
    recording: SensorRecording,
    pad_seconds: float = 2.0,
    min_prominence: float = 1.0,
) -> list[SyncAnchor]:
    """Detect one synchronization anchor per window, when present."""
    windows = SyncWindowExtractor().extract(
        session,
        recording.data,
        timestamp_column=TIMESTAMP_COLUMN,
        pad_seconds=pad_seconds,
    )

    clap_detector = ClapDetector()
    event_detector = SyncEventDetector(min_prominence=min_prominence)

    anchors = []
    for window in session.sync_windows:
        samples = windows.get(window.event_id)
        if samples is None or len(samples) < 3:
            continue

        anchor = event_detector.detect(window, clap_detector.detect(samples))
        if anchor is not None:
            anchors.append(anchor)

    return anchors


def fit_against_reference(
    anchors: list[SyncAnchor],
    reference_seconds: dict[str, float],
) -> Optional[dict]:  # noqa: UP045
    """
    Fit reference_time = intercept + slope * sensor_relative_time.

    ``reference_seconds`` maps event IDs to the reference time of the same
    clap, in seconds (Unix seconds for a sensor reference, seconds from the
    start of the video for a video reference). Returns None when fewer than
    two claps are shared.
    """
    shared = [anchor for anchor in anchors if anchor.event_id in reference_seconds]
    if len(shared) < 2:
        return None

    reference_origin = min(reference_seconds[a.event_id] for a in shared)

    pairs = [
        SyncAnchorPair(
            event_id=anchor.event_id,
            sensor_timestamp_unix_ms=anchor.sensor_timestamp_unix_ms,
            video_timestamp_seconds=reference_seconds[anchor.event_id] - reference_origin,
            sensor_selection_rule=anchor.selection_rule,
            video_selection_method="reference",
        )
        for anchor in shared
    ]

    fit = SyncModel().fit(pairs)

    residuals_ms = np.asarray(fit["residuals_seconds"]) * 1000.0
    reference_at_first = reference_origin + fit["intercept_seconds"]

    return {
        "n_pairs": fit["n_anchors"],
        "reference_origin_seconds": reference_origin,
        "sensor_origin_unix_seconds": fit["sensor_origin_unix_seconds"],
        "intercept_seconds": fit["intercept_seconds"],
        "slope": fit["slope"],
        "drift_ppm": fit["drift_ppm"],
        # Reference minus sensor time at the first shared clap.
        "offset_ms": (reference_at_first - fit["sensor_origin_unix_seconds"]) * 1000.0,
        "rmse_ms": fit["rmse_seconds"] * 1000.0,
        "max_abs_residual_ms": float(np.max(np.abs(residuals_ms))),
        "residuals_ms": {
            pair.event_id: float(residual)
            for pair, residual in zip(
                sorted(pairs, key=lambda p: p.sensor_timestamp_unix_ms),
                residuals_ms,
            )
        },
    }


def coverage_metrics(session: Session, recording: SensorRecording) -> dict:
    """
    Share of the session covered by samples and gaps in the recording.

    A gap is an interval between consecutive samples longer than
    ``GAP_FACTOR`` times the nominal sampling interval.
    """
    timestamps = recording.data[TIMESTAMP_COLUMN].to_numpy(dtype=float)
    start_ms = session.start_time.timestamp() * 1000.0
    end_ms = session.end_time.timestamp() * 1000.0
    session_ms = max(end_ms - start_ms, 0.0)

    inside = timestamps[(timestamps >= start_ms) & (timestamps <= end_ms)]
    nominal_ms = 1000.0 / recording.sampling_rate_hz

    if inside.size < 2 or session_ms == 0:
        return {
            "session_seconds": session_ms / 1000.0,
            "covered_seconds": 0.0,
            "coverage_pct": 0.0,
            "n_gaps": 0,
            "longest_gap_seconds": 0.0,
        }

    diffs = np.diff(inside)
    gaps = diffs[diffs > GAP_FACTOR * nominal_ms]

    # Time with samples: span of the samples inside the session minus the
    # gaps (an uncovered start or end of the session is not counted).
    covered_ms = (inside[-1] - inside[0]) - float(gaps.sum())

    return {
        "session_seconds": session_ms / 1000.0,
        "covered_seconds": covered_ms / 1000.0,
        "coverage_pct": 100.0 * covered_ms / session_ms,
        "n_gaps": int(gaps.size),
        "longest_gap_seconds": float(gaps.max() / 1000.0) if gaps.size else 0.0,
    }


def synchronize_session(
    session: Session,
    recordings: dict[str, SensorRecording],
    reference: str,
    video_events: Optional[dict[str, float]] = None,  # noqa: UP045
    pad_seconds: float = 2.0,
    min_prominence: float = 1.0,
) -> dict[str, SensorSyncResult]:
    """
    Detect anchors in every sensor and fit each one against the reference.

    Parameters
    ----------
    session
        Parsed session manifest.
    recordings
        Sensor recordings keyed by sensor ID.
    reference
        Sensor ID used as the reference timeline, or ``"video"`` to use
        ``video_events`` (event ID -> seconds from the start of the video).
    """
    results = {}
    for sensor_id, recording in recordings.items():
        results[sensor_id] = SensorSyncResult(
            sensor_id=sensor_id,
            sensor_type=recording.sensor_type,
            sampling_rate_hz=recording.sampling_rate_hz,
            anchors=detect_anchors(session, recording, pad_seconds, min_prominence),
            coverage=coverage_metrics(session, recording),
        )

    if reference == "video":
        if not video_events:
            raise ValueError("A video reference requires video events.")
        reference_seconds = dict(video_events)
    elif reference in results:
        reference_seconds = {
            anchor.event_id: anchor.sensor_timestamp_unix_ms / 1000.0
            for anchor in results[reference].anchors
        }
    else:
        raise ValueError(f"Unknown reference '{reference}'.")

    for sensor_id, result in results.items():
        if sensor_id == reference:
            result.note = "reference"
            continue

        result.fit = fit_against_reference(result.anchors, reference_seconds)
        if result.fit is None:
            result.note = "fewer than two claps shared with the reference"

    return results


def anchors_table(session: Session, results: dict[str, SensorSyncResult]) -> pd.DataFrame:
    """One row per detected anchor."""
    activity_by_event = {w.event_id: w.activity_index for w in session.sync_windows}

    rows = [
        {
            "sensor_id": result.sensor_id,
            "event_id": anchor.event_id,
            "activity_index": activity_by_event.get(anchor.event_id),
            "phase": anchor.phase,
            "sensor_timestamp_unix_ms": anchor.sensor_timestamp_unix_ms,
            "selection_rule": anchor.selection_rule,
            "prominence": anchor.prominence,
            "peak_value": anchor.peak_value,
            "candidate_count": anchor.candidate_count,
            "residual_ms": (
                result.fit["residuals_ms"].get(anchor.event_id)
                if result.fit
                else None
            ),
        }
        for result in results.values()
        for anchor in result.anchors
    ]

    return pd.DataFrame(rows, columns=[
        "sensor_id",
        "event_id",
        "activity_index",
        "phase",
        "sensor_timestamp_unix_ms",
        "selection_rule",
        "prominence",
        "peak_value",
        "candidate_count",
        "residual_ms",
    ])


def report_table(session: Session, results: dict[str, SensorSyncResult]) -> pd.DataFrame:
    """One row per sensor with the synchronization and coverage summary."""
    n_windows = len(session.sync_windows)

    rows = []
    for result in results.values():
        fit = result.fit or {}
        rows.append({
            "session_id": session.session_id,
            "sensor_id": result.sensor_id,
            "sensor_type": result.sensor_type,
            "sampling_rate_hz": result.sampling_rate_hz,
            "n_windows": n_windows,
            "n_anchors": len(result.anchors),
            "n_pairs": fit.get("n_pairs"),
            "offset_ms": fit.get("offset_ms"),
            "drift_ppm": fit.get("drift_ppm"),
            "rmse_ms": fit.get("rmse_ms"),
            "max_abs_residual_ms": fit.get("max_abs_residual_ms"),
            "coverage_pct": result.coverage.get("coverage_pct"),
            "n_gaps": result.coverage.get("n_gaps"),
            "longest_gap_seconds": result.coverage.get("longest_gap_seconds"),
            "note": result.note,
        })

    return pd.DataFrame(rows)


def apply_alignment(recording: SensorRecording, fit: dict) -> pd.DataFrame:
    """
    Return the recording with ``timestamp_aligned_unix_ms``: sensor time
    mapped to the reference timeline (only meaningful for a sensor
    reference, whose timeline is Unix time).
    """
    data = recording.data.copy()
    sensor_seconds = data[TIMESTAMP_COLUMN].to_numpy(dtype=float) / 1000.0
    relative = sensor_seconds - fit["sensor_origin_unix_seconds"]

    aligned = (
        fit["reference_origin_seconds"]
        + fit["intercept_seconds"]
        + fit["slope"] * relative
    )
    data["timestamp_aligned_unix_ms"] = aligned * 1000.0

    return data
