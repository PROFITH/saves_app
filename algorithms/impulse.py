"""
Impacts in the acceleration magnitude: claps and jump landings.

Both detectors return candidates with the same keys as ClapDetector, so the
protocol rules (SyncEventDetector) treat them alike. The time used as the
anchor is the onset of the impact, where the magnitude crosses half of its
rise to the peak, interpolated between samples: it is what the video shows
as "the first frame of the impact" and depends less on the sampling rate
than the peak sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Below this magnitude (g) the body is in the air.
FLIGHT_G = 0.6
# Longest flight looked for before a landing (s); a 30-cm jump flies 0.5 s.
MAX_FLIGHT_S = 0.7
# Peaks closer than this (s) belong to the same jump (takeoff and landing).
JUMP_CLUSTER_S = 0.8
# A landing is an impact of at least this magnitude (g).
MIN_LANDING_G = 1.3


def onset_ms(t_ms: np.ndarray, vm: np.ndarray, peak: int, base: float) -> float:
    """
    Time at which the rise from ``base`` to the peak crosses half its height.

    Scans back from the peak to the last sample below the half height and
    interpolates linearly with the next one.
    """
    top = float(vm[peak])
    half = base + 0.5 * (top - base)
    j = peak
    while j > 0 and vm[j - 1] >= half:
        j -= 1
    if j == 0 or vm[j] == vm[j - 1]:
        return float(t_ms[j])
    fraction = (half - vm[j - 1]) / (vm[j] - vm[j - 1])
    fraction = min(max(float(fraction), 0.0), 1.0)
    return float(t_ms[j - 1] + fraction * (t_ms[j] - t_ms[j - 1]))


def valid_arrays(
    sensor_data: pd.DataFrame,
    timestamp_column: str,
    signal_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    signal = pd.to_numeric(sensor_data[signal_column], errors="coerce").to_numpy(dtype=float)
    timestamps = pd.to_numeric(sensor_data[timestamp_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(signal) & np.isfinite(timestamps)
    return timestamps[valid], signal[valid]


def robust_sd(values: np.ndarray) -> float:
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


class JumpDetector:
    """
    Detect jump landings in a synchronization window.

    A landing is a large impact preceded by a flight, a stretch in which the
    magnitude falls towards 0 g. The takeoff before it (the push after the
    countermovement) is a smaller peak without that flight. Each jump gives
    one candidate: its landing.
    """

    def detect(
        self,
        sensor_data: pd.DataFrame,
        timestamp_column: str = "timestamp_unix_ms",
        signal_column: str = "Vector Magnitude",
    ) -> dict:
        if sensor_data.empty:
            raise ValueError("Cannot detect events in an empty sensor window.")
        t, vm = valid_arrays(sensor_data, timestamp_column, signal_column)
        if len(vm) < 3:
            return {"detected": False, "candidates": []}

        dt = float(np.median(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Unable to determine the sensor sampling interval.")

        scale = robust_sd(vm)
        min_prominence = max(3.0 * scale, 0.3)
        peaks, properties = find_peaks(vm, prominence=min_prominence, height=MIN_LANDING_G)

        raw = []
        for i, peak in enumerate(peaks):
            lo = int(np.searchsorted(t, t[peak] - MAX_FLIGHT_S * 1000.0))
            before = vm[lo:peak + 1]
            dip = int(lo + np.argmin(before))
            flight_s = float(np.count_nonzero(before < FLIGHT_G)) * dt / 1000.0
            prominence = float(properties["prominences"][i])
            raw.append({
                "peak": int(peak),
                "dip": dip,
                "flight_s": flight_s,
                "prominence": prominence,
                # Landings: big impacts right after a flight.
                "score": prominence * (1.0 + flight_s / 0.1),
            })

        # One candidate per jump: the best scored peak of each cluster.
        clusters: list[list[dict]] = []
        for item in raw:
            if clusters and t[item["peak"]] - t[clusters[-1][-1]["peak"]] <= JUMP_CLUSTER_S * 1000.0:
                clusters[-1].append(item)
            else:
                clusters.append([item])

        candidates = []
        for cluster in clusters:
            best = max(cluster, key=lambda item: item["score"])
            peak = best["peak"]
            flight = best["flight_s"] >= 0.04
            base = float(vm[best["dip"]]) if flight else float(np.median(vm[max(peak - int(500 / dt), 0):peak + 1]))
            candidates.append({
                "timestamp_unix_ms": int(t[peak]),
                "onset_unix_ms": onset_ms(t, vm, peak, base),
                "peak_value": float(vm[peak]),
                "prominence": best["prominence"],
                "sample_index": peak,
                "flight_seconds": best["flight_s"],
                "flight": flight,
            })

        # When some candidate is a real jump, the others are not.
        if any(c["flight"] for c in candidates):
            candidates = [c for c in candidates if c["flight"]]

        candidates.sort(key=lambda c: c["prominence"], reverse=True)
        return {
            "detected": bool(candidates),
            "candidates": candidates,
            "median_sample_interval_ms": dt,
            "robust_sd": scale,
            "min_prominence": min_prominence,
        }


def add_onsets(clap_result: dict, sensor_data: pd.DataFrame, timestamp_column: str, signal_column: str) -> dict:
    """Add the onset time to the candidates of ClapDetector."""
    candidates = clap_result.get("candidates", [])
    if not candidates:
        return clap_result
    t, vm = valid_arrays(sensor_data, timestamp_column, signal_column)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 10.0
    span = max(int(round(500.0 / dt)), 3)
    for candidate in candidates:
        peak = int(candidate["sample_index"])
        if peak >= len(vm):
            continue
        base = float(np.median(vm[max(peak - span, 0):max(peak - 1, 1)]))
        candidate["onset_unix_ms"] = onset_ms(t, vm, peak, min(base, float(vm[peak])))
    return clap_result


def impulse_candidates(
    t_ms: np.ndarray,
    vm: np.ndarray,
    min_prominence_g: float = 0.5,
    quiet_factor: float = 8.0,
    exclusion_s: float = 1.2,
    context_s: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Isolated impacts in a long recording: clear peaks with quiet signal on
    both sides, like a clap or a jump during a still phase of the protocol.

    Steps, sitting down or handling the sensor are not isolated and are
    left out. Returns the times (ms) and prominences of the impacts.
    """
    if t_ms.size < 10:
        return np.zeros(0), np.zeros(0)
    dt = float(np.median(np.diff(t_ms)))
    if not np.isfinite(dt) or dt <= 0:
        return np.zeros(0), np.zeros(0)
    wlen = max(int(round(2 * exclusion_s * 1000.0 / dt)) | 1, 3)
    peaks, properties = find_peaks(
        vm,
        height=1.0 + min_prominence_g * 0.5,
        prominence=min_prominence_g,
        wlen=wlen,
        distance=max(int(round(500.0 / dt)), 1),
    )
    if peaks.size == 0:
        return np.zeros(0), np.zeros(0)

    values = vm.astype(float)
    s1 = np.concatenate([[0.0], np.cumsum(values)])
    s2 = np.concatenate([[0.0], np.cumsum(values * values)])

    def side_sd(a_s: float, b_s: float) -> np.ndarray:
        i0 = np.searchsorted(t_ms, t_ms[peaks] + a_s * 1000.0)
        i1 = np.searchsorted(t_ms, t_ms[peaks] + b_s * 1000.0)
        n = np.maximum(i1 - i0, 1)
        mean = (s1[i1] - s1[i0]) / n
        var = np.maximum((s2[i1] - s2[i0]) / n - mean * mean, 0.0)
        sd = np.sqrt(var)
        # Too few samples on a side (edge of the recording): not quiet.
        return np.where(i1 - i0 >= 3, sd, np.inf)

    left = side_sd(-context_s, -exclusion_s)
    right = side_sd(exclusion_s, context_s)
    context = np.maximum(left, right)
    prominences = properties["prominences"]
    keep = prominences >= quiet_factor * (context + 0.01)
    return t_ms[peaks][keep], prominences[keep]
