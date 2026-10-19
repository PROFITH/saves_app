"""
Frame-accurate video access with PyAV (FFmpeg), without Qt.

Each frame carries its presentation timestamp, so a clap marked on a frame
gets the time FFmpeg reports for it, never one rebuilt from the nominal
frame rate. When the video has sound, claps can also be found in the audio
track, which is sharper than any frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import av
except ImportError:  # pragma: no cover - reported in the window
    av = None

MAX_WIDTH = 960


@dataclass
class VideoFrame:
    time_s: float
    index_hint: int
    rgb: np.ndarray  # (H, W, 3) uint8


class VideoReader:
    """Random and sequential access to the frames of one video file."""

    def __init__(self, path: str | Path, max_width: int = MAX_WIDTH):
        if av is None:
            raise RuntimeError("Falta PyAV: instala el paquete 'av' para ver vídeo.")
        self.path = str(path)
        self.container = av.open(self.path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.time_base = float(self.stream.time_base)
        rate = self.stream.average_rate or self.stream.guessed_rate or 30
        self.fps = float(rate)
        self.frame_duration = 1.0 / self.fps if self.fps > 0 else 1 / 30
        if self.stream.duration is not None:
            self.duration_s = float(self.stream.duration * self.stream.time_base)
        elif self.container.duration is not None:
            self.duration_s = self.container.duration / 1_000_000.0
        else:
            self.duration_s = 0.0
        self.start_s = float(self.stream.start_time * self.stream.time_base) if self.stream.start_time else 0.0
        width, height = self.stream.codec_context.width, self.stream.codec_context.height
        scale = min(1.0, max_width / width) if width else 1.0
        self.out_size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
        self._decoder = None
        self._pending = None
        self.current: Optional[VideoFrame] = None  # noqa: UP045

    def close(self) -> None:
        self.container.close()

    def _to_frame(self, frame) -> VideoFrame:
        t = float(frame.pts * self.time_base) - self.start_s if frame.pts is not None else 0.0
        rgb = frame.to_ndarray(format="rgb24", width=self.out_size[0], height=self.out_size[1])
        return VideoFrame(time_s=t, index_hint=int(round(t * self.fps)), rgb=rgb)

    def _seek(self, t_s: float) -> None:
        target = int(max(t_s + self.start_s, 0.0) / self.time_base)
        self.container.seek(target, stream=self.stream, backward=True, any_frame=False)
        self._decoder = self.container.decode(self.stream)
        self._pending = None

    def frame_at(self, t_s: float) -> Optional[VideoFrame]:  # noqa: UP045
        """The frame on screen at time ``t_s`` (last frame starting at or before it)."""
        t_s = min(max(t_s, 0.0), max(self.duration_s - self.frame_duration, 0.0))
        self._seek(t_s)
        previous = None
        self._pending = None
        for frame in self._decoder:
            if frame.pts is None:
                continue
            ft = float(frame.pts * self.time_base) - self.start_s
            if ft > t_s + self.frame_duration * 0.5:
                # Already decoded: next_frame() returns it without decoding again.
                self._pending = frame
                break
            previous = frame
        if previous is None:
            return self.next_frame()
        self.current = self._to_frame(previous)
        return self.current

    def next_frame(self) -> Optional[VideoFrame]:  # noqa: UP045
        if self._decoder is None:
            self._seek(0.0)
        if self._pending is not None:
            frame, self._pending = self._pending, None
            self.current = self._to_frame(frame)
            return self.current
        for frame in self._decoder:
            if frame.pts is None:
                continue
            self.current = self._to_frame(frame)
            return self.current
        return None

    def previous_frame(self) -> Optional[VideoFrame]:  # noqa: UP045
        if self.current is None:
            return self.frame_at(0.0)
        return self.frame_at(self.current.time_s - self.frame_duration)


def has_audio(path: str | Path) -> bool:
    if av is None:
        return False
    with av.open(str(path)) as container:
        return len(container.streams.audio) > 0


def load_audio_envelope(path: str | Path, rate: int = 4000) -> tuple[np.ndarray, float]:
    """Mono audio at ``rate`` Hz and the time (s) of its first sample."""
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        chunks, start = [], None
        for frame in container.decode(stream):
            if start is None and frame.pts is not None:
                start = float(frame.pts * stream.time_base)
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return audio, start or 0.0


def detect_audio_claps(
    path: str | Path,
    windows_video_s: dict[str, tuple[float, float]],
    rate: int = 4000,
) -> dict[str, float]:
    """
    Find the loudest sharp sound inside each window (video seconds).

    Returns event_id -> onset time in video seconds: the first sample that
    reaches half of the peak, which marks the impact better than the peak.
    """
    audio, start = load_audio_envelope(path, rate)
    if audio.size == 0:
        return {}
    transient = np.abs(np.diff(audio, prepend=audio[:1]))
    found = {}
    for event_id, (v0, v1) in windows_video_s.items():
        i0 = max(int((v0 - start) * rate), 0)
        i1 = min(int((v1 - start) * rate), transient.size)
        if i1 - i0 < rate // 10:
            continue
        segment = transient[i0:i1]
        peak = int(np.argmax(segment))
        noise = float(np.median(segment)) + 1e-9
        if segment[peak] < 8 * noise:
            continue
        onset = peak
        threshold = 0.5 * segment[peak]
        while onset > 0 and segment[onset - 1] >= threshold:
            onset -= 1
        found[event_id] = start + (i0 + onset) / rate
    return found
