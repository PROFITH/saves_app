"""
Coarse clock offset of a sensor, found before looking for the claps.

The claps are looked for in short windows around the times logged by the
protocol app, so a sensor whose clock is more than a couple of seconds off
(set by hand, configured in another time zone, or never synchronized) would
miss them all. Here every isolated impact of the recording votes for the
offsets that would put it on one of the expected events; the offset that
lines up the most events wins. It searches up to a couple of hours, which
covers a device left in UTC or in winter time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from algorithms.impulse import impulse_candidates
from models import Session

# Search range of offline sensors and of sensors already on the host clock.
OFFLINE_SEARCH_S = 2 * 3600 + 600
HOST_SEARCH_S = 10.0
# How far from the expected time an event may be once the offset is applied.
CUE_TOLERANCE_S = 1.0


@dataclass
class OffsetSearch:
    """Result of the coarse offset search of one sensor."""

    offset_ms: float = 0.0
    accepted: bool = False
    votes: int = 0
    n_events: int = 0
    runner_up_votes: int = 0
    matched: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def describe(self) -> str:
        if not self.accepted:
            return self.note or "Sin desfase estimado"
        return f"{format_offset(self.offset_ms)} ({self.votes}/{self.n_events} eventos)"


def format_offset(offset_ms: float) -> str:
    """+1 h 02 min 03,4 s, +3 min 12,0 s or +0,85 s."""
    sign = "-" if offset_ms < 0 else "+"
    seconds = abs(offset_ms) / 1000.0
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours >= 1:
        return f"{sign}{int(hours)} h {int(minutes):02d} min {secs:04.1f} s".replace(".", ",")
    if minutes >= 1:
        return f"{sign}{int(minutes)} min {secs:04.1f} s".replace(".", ",")
    return f"{sign}{secs:.2f} s".replace(".", ",")


def expected_events(session: Session) -> list[tuple[str, float, float]]:
    """(event_id, expected host time in ms, tolerance in s) of every window."""
    events = []
    for window in session.sync_windows:
        expected = window.expected_time
        if expected is not None:
            events.append((window.event_id, expected.timestamp() * 1000.0, CUE_TOLERANCE_S))
        else:
            # No cue logged: the clap is somewhere in the window.
            start = window.window_start.timestamp() * 1000.0
            end = window.window_end.timestamp() * 1000.0
            events.append((window.event_id, 0.5 * (start + end), 0.5 * (end - start) / 1000.0 + 0.5))
    return events


def estimate_offset(
    session: Session,
    t_ms: np.ndarray,
    vm: np.ndarray,
    max_offset_s: float = OFFLINE_SEARCH_S,
) -> OffsetSearch:
    """
    Sensor clock minus host clock (ms) that lines up the most sync events.

    ``t_ms`` and ``vm`` are the sensor's timestamps (its own clock, Unix ms)
    and acceleration magnitude (g), covering the session plus the search
    range on both sides.
    """
    events = expected_events(session)
    result = OffsetSearch(n_events=len(events))
    if len(events) < 2:
        result.note = "Hacen falta 2 eventos de sincronización en el manifiesto"
        return result

    times, prominences = impulse_candidates(np.asarray(t_ms, dtype=float), np.asarray(vm, dtype=float))
    if times.size == 0:
        result.note = "No hay impactos claros en la señal"
        return result

    # Every (impact, event) pair votes for one offset.
    offsets, owners, widths = [], [], []
    limit = max_offset_s * 1000.0
    for index, (_event_id, expected_ms, tolerance_s) in enumerate(events):
        d = times - expected_ms
        keep = np.abs(d) <= limit
        offsets.append(d[keep])
        owners.append(np.full(int(keep.sum()), index))
        widths.append(np.full(int(keep.sum()), tolerance_s * 1000.0))
    d = np.concatenate(offsets)
    owner = np.concatenate(owners)
    if d.size == 0:
        result.note = "Ningún impacto cae cerca de los eventos esperados"
        return result
    order = np.argsort(d)
    d, owner = d[order], owner[order]
    width = 2.0 * min(float(np.concatenate(widths).max()), 6000.0)

    best_votes, best_start = _sliding_best(d, owner, width)
    best_offset = _bin_offset(d, owner, best_start, width)

    # Best vote count away from the winner, to tell a clear winner from noise.
    away = np.abs(d - best_offset) > max(width, 5000.0)
    runner_up = _sliding_best(d[away], owner[away], width)[0] if away.any() else 0

    result.votes = best_votes
    result.runner_up_votes = runner_up
    needed = max(2, int(np.ceil(0.25 * len(events))))
    clear = best_votes >= runner_up + 2 or best_votes >= 2 * max(runner_up, 1)
    if best_votes < needed or not clear:
        result.note = (
            f"Desfase dudoso: {best_votes} eventos coinciden (hacen falta {needed}); "
            "se buscan los eventos a la hora del manifiesto"
        )
        return result

    # Refine with the impact closest to each expected event.
    matched = {}
    residuals = []
    for event_id, expected_ms, tolerance_s in events:
        target = expected_ms + best_offset
        i = int(np.argmin(np.abs(times - target)))
        if abs(times[i] - target) <= tolerance_s * 1000.0:
            matched[event_id] = float(times[i])
            residuals.append(times[i] - expected_ms)
    result.offset_ms = float(np.median(residuals)) if residuals else best_offset
    result.matched = matched
    result.accepted = True
    return result


def _sliding_best(d: np.ndarray, owner: np.ndarray, width: float) -> tuple[int, int]:
    """Most distinct events inside any interval of ``width`` (sorted ``d``)."""
    counts: dict[int, int] = {}
    best, best_start, right = 0, 0, 0
    n = d.size
    for left in range(n):
        while right < n and d[right] - d[left] <= width:
            counts[owner[right]] = counts.get(owner[right], 0) + 1
            right += 1
        if len(counts) > best:
            best, best_start = len(counts), left
        counts[owner[left]] -= 1
        if counts[owner[left]] == 0:
            del counts[owner[left]]
    return best, best_start


def _bin_offset(d: np.ndarray, owner: np.ndarray, start: int, width: float) -> float:
    """Median offset of the winning interval, one value per event."""
    stop = int(np.searchsorted(d, d[start] + width, side="right"))
    values: dict[int, float] = {}
    centre = float(np.median(d[start:stop]))
    for value, event in zip(d[start:stop], owner[start:stop]):
        if event not in values or abs(value - centre) < abs(values[event] - centre):
            values[event] = float(value)
    return float(np.median(list(values.values())))


def search_range_for(sensor_type: str) -> float:
    """Sensors written by the protocol app are already on the host clock."""
    return HOST_SEARCH_S if sensor_type == "polar_acc" else OFFLINE_SEARCH_S


def describe_optional(search: Optional[OffsetSearch]) -> str:  # noqa: UP045
    return search.describe() if search is not None else ""
