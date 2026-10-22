"""
Fast loading and plotting support for sensor signals.

Parsing a week of 100 Hz data takes tens of seconds, so each file is parsed
once and cached as ``.npz`` in ``<session>/.saves_cache``; later openings
load in about a second. For plotting, a min/max pyramid returns at most a
few thousand points for any zoom level, so panning a 7-day recording is as
fluid as a 10-minute one.

A signal keeps every channel of its file: the acceleration (``x``, ``y``,
``z`` and their magnitude ``vm``, the only channels the plots and the clap
detection use) and whatever else the reader found (gyroscope, light,
temperature, heart rate...), on the same sample grid, for the export.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from readers import read_sensor
from readers.recording import MAGNITUDE_COLUMN, TIMESTAMP_COLUMN, SensorRecording

CACHE_DIR = ".saves_cache"
CACHE_VERSION = 2
PYRAMID_FACTOR = 8
# Channels every signal has; the rest depend on the file.
CHANNELS = ("x", "y", "z", "vm")
# Values of this size or more lose precision as float32 (integer counters, timestamps).
_FLOAT32_EXACT = 2**24


@dataclass
class Envelope:
    """Min/max of one channel over bins of increasing size."""

    # levels[k] = (bin_start_ms, minimum, maximum); level 0 is the raw signal.
    levels: list[tuple[np.ndarray, np.ndarray, np.ndarray]]

    @classmethod
    def build(cls, t_ms: np.ndarray, values: np.ndarray, min_bins: int = 2000) -> Envelope:
        levels = [(t_ms, values, values)]
        t, lo, hi = t_ms, values, values
        while t.size > min_bins * PYRAMID_FACTOR:
            n = (t.size // PYRAMID_FACTOR) * PYRAMID_FACTOR
            t = t[:n:PYRAMID_FACTOR]
            lo = lo[:n].reshape(-1, PYRAMID_FACTOR).min(axis=1)
            hi = hi[:n].reshape(-1, PYRAMID_FACTOR).max(axis=1)
            levels.append((t, lo, hi))
        return cls(levels)

    def query(self, t0_ms: float, t1_ms: float, max_points: int = 4000) -> tuple[np.ndarray, np.ndarray]:
        """
        Points to draw between ``t0_ms`` and ``t1_ms``.

        Uses the finest level that fits in ``max_points``; binned levels
        return the minimum and maximum of each bin so impacts stay visible.
        """
        for level, (t, lo, hi) in enumerate(self.levels):
            i0 = max(int(np.searchsorted(t, t0_ms)) - 1, 0)
            i1 = min(int(np.searchsorted(t, t1_ms)) + 1, t.size)
            count = i1 - i0 if level == 0 else 2 * (i1 - i0)
            if count <= max_points or level == len(self.levels) - 1:
                break
        if level == 0 and i1 - i0 <= max_points:
            return t[i0:i1], lo[i0:i1]
        t, lo, hi = t[i0:i1], lo[i0:i1], hi[i0:i1]
        # The coarsest level can still hold too many bins for a long span:
        # merge neighbouring bins on the fly.
        step = int(np.ceil(2 * t.size / max_points))
        if step > 1:
            n = (t.size // step) * step
            t = t[:n:step]
            lo = lo[:n].reshape(-1, step).min(axis=1)
            hi = hi[:n].reshape(-1, step).max(axis=1)
        x = np.repeat(t, 2)
        y = np.empty(x.size, dtype=lo.dtype)
        y[0::2] = lo
        y[1::2] = hi
        return x, y


@dataclass
class SignalData:
    """
    Samples of one sensor kept as plain arrays.

    ``channels`` always holds ``x``, ``y``, ``z`` and ``vm`` first, then
    the other channels of the file in the order the reader gave them.
    """

    sensor_type: str
    t_ms: np.ndarray
    channels: dict[str, np.ndarray]
    sampling_rate_hz: float
    metadata: dict = field(default_factory=dict)
    _envelopes: dict[str, Envelope] = field(default_factory=dict, repr=False)

    @property
    def start_ms(self) -> float:
        return float(self.t_ms[0])

    @property
    def end_ms(self) -> float:
        return float(self.t_ms[-1])

    @property
    def duration_s(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0

    @property
    def extra_channels(self) -> list[str]:
        """Channels other than x, y, z and vm, in reader order."""
        return [c for c in self.channels if c not in CHANNELS]

    def envelope(self, channel: str) -> Envelope:
        if channel not in self._envelopes:
            self._envelopes[channel] = Envelope.build(self.t_ms, self.channels[channel])
        return self._envelopes[channel]

    def to_recording(self, t0_ms: Optional[float] = None, t1_ms: Optional[float] = None) -> SensorRecording:  # noqa: UP045
        """SensorRecording for the headless pipeline (every channel), optionally cropped."""
        i0 = 0 if t0_ms is None else int(np.searchsorted(self.t_ms, t0_ms))
        i1 = self.t_ms.size if t1_ms is None else int(np.searchsorted(self.t_ms, t1_ms, side="right"))
        frame = pd.DataFrame({TIMESTAMP_COLUMN: self.t_ms[i0:i1]})
        for name in ("x", "y", "z", *self.extra_channels):
            frame[name] = self.channels[name][i0:i1].astype(float)
        frame[MAGNITUDE_COLUMN] = self.channels["vm"][i0:i1].astype(float)
        return SensorRecording(
            sensor_type=self.sensor_type,
            data=frame,
            sampling_rate_hz=self.sampling_rate_hz,
            metadata=self.metadata,
        )

    @classmethod
    def from_recording(cls, recording: SensorRecording) -> SignalData:
        data = recording.data
        channels = {c: data[c].to_numpy(dtype=np.float32) for c in ("x", "y", "z")}
        channels["vm"] = data[MAGNITUDE_COLUMN].to_numpy(dtype=np.float32)
        for column in data.columns:
            if column not in (TIMESTAMP_COLUMN, MAGNITUDE_COLUMN, "x", "y", "z"):
                channels[str(column)] = _compact(data[column].to_numpy(dtype=np.float64))
        return cls(
            sensor_type=recording.sensor_type,
            t_ms=data[TIMESTAMP_COLUMN].to_numpy(dtype=np.float64),
            channels=channels,
            sampling_rate_hz=float(recording.sampling_rate_hz),
            metadata=_jsonable(recording.metadata),
        )


def _compact(values: np.ndarray) -> np.ndarray:
    """float32 unless the values are too large for it to keep them exactly."""
    finite = values[np.isfinite(values)]
    if finite.size and np.abs(finite).max() >= _FLOAT32_EXACT:
        return values
    return values.astype(np.float32)


def _jsonable(value):
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return {}


def cache_path(file_path: str | Path, cache_root: str | Path, timezone: str) -> Path:
    """Cache file name tied to the file's path, size, date and time zone."""
    p = Path(file_path)
    stat = p.stat()
    key = f"{p.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{timezone}|{CACHE_VERSION}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return Path(cache_root) / CACHE_DIR / f"{p.stem}_{digest}.npz"


def load_signal(
    file_path: str | Path,
    sensor_format: str,
    cache_root: Optional[str | Path] = None,  # noqa: UP045
    timezone: str = "Europe/Madrid",
    progress: Optional[Callable[[str], None]] = None,  # noqa: UP045
) -> SignalData:
    """Read a sensor file, using (and filling) the cache when possible."""
    cached = cache_path(file_path, cache_root, timezone) if cache_root else None

    if cached is not None and cached.exists():
        if progress:
            progress("Cargando desde la caché")
        try:
            with np.load(cached, allow_pickle=False) as npz:
                meta = json.loads(str(npz["meta"]))
                return SignalData(
                    sensor_type=meta["sensor_type"],
                    t_ms=npz["t_ms"],
                    channels={c: npz[f"ch_{c}"] for c in meta["channels"]},
                    sampling_rate_hz=meta["sampling_rate_hz"],
                    metadata=meta.get("metadata", {}),
                )
        except (OSError, KeyError, ValueError):
            pass  # Corrupt cache: parse the file again.

    if progress:
        progress("Leyendo el archivo")
    signal = SignalData.from_recording(read_sensor(file_path, sensor_format, timezone=timezone))

    if cached is not None:
        if progress:
            progress("Guardando en la caché")
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            meta = {
                "sensor_type": signal.sensor_type,
                "sampling_rate_hz": signal.sampling_rate_hz,
                "channels": list(signal.channels),
                "metadata": signal.metadata,
            }
            tmp = cached.with_name(cached.stem + ".tmp.npz")
            np.savez(tmp, t_ms=signal.t_ms, meta=json.dumps(meta), **{f"ch_{c}": v for c, v in signal.channels.items()})
            tmp.replace(cached)
        except OSError:
            pass  # A read-only folder only loses the speed-up.
    return signal
