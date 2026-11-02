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


def probe_duration(path: str | Path) -> Optional[float]:  # noqa: UP045
    """Duration of a video in seconds, or None if it cannot be read."""
    if av is None:
        return None
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if stream.duration is not None:
                return float(stream.duration * stream.time_base)
            if container.duration is not None:
                return container.duration / 1_000_000.0
    except (OSError, IndexError, ValueError):
        return None
    except Exception:  # noqa: BLE001 - any FFmpeg error
        return None
    return None


def has_audio(path: str | Path) -> bool:
    if av is None:
        return False
    with av.open(str(path)) as container:
        return len(container.streams.audio) > 0


def load_audio_envelope(path: str | Path, rate: int = 4000) -> tuple[np.ndarray, float]:
    """
    Mono audio at ``rate`` Hz and the time (s) of its first sample, in the
    video's own time (seconds from its first frame, as VideoReader counts).
    """
    with av.open(str(path)) as container:
        video_start = 0.0
        if container.streams.video:
            vs = container.streams.video[0]
            video_start = float(vs.start_time * vs.time_base) if vs.start_time else 0.0
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
    return audio, (start or 0.0) - video_start


# A clap or a landing rises out of quiet and fades within a few tens of
# milliseconds; the timer's beeps and the voice last much longer. The sound
# 20-60 ms before a peak and 60-150 ms after it, relative to the peak, tells
# them apart.
FRAME_S = 0.005
ATTACK_FROM_S = 0.06
ATTACK_TO_S = 0.02
MAX_ATTACK = 0.5
SUSTAIN_FROM_S = 0.06
SUSTAIN_TO_S = 0.15
MAX_SUSTAIN = 0.45
# A sharp sound stands this many times above the background around it.
MIN_RATIO = 8.0
# Two events never share a sound closer than this (s).
SAME_SOUND_S = 0.3
# Isolated sounds, the only ones that vote for the video's offset: nothing as
# loud between 1.2 and 4 s on either side (steps come every half second).
EXCLUSION_S = 1.2
CONTEXT_S = 4.0


@dataclass
class SharpSounds:
    """Short, sharp sounds of an audio track (claps, landings, steps, knocks)."""

    onsets_s: np.ndarray  # first sample at half the peak, video seconds
    strength: np.ndarray  # height of the peak in the envelope
    isolated: np.ndarray  # bool: no sound as loud 1.2-4 s around it


@dataclass
class AudioMatch:
    """Claps and landings found in the sound of a video."""

    found: dict[str, float]  # event_id -> onset, video seconds
    shift_s: float = 0.0  # how far the video was from its expected time
    searched: bool = False  # whether the offset search found a clear shift


def sharp_sounds(audio: np.ndarray, start: float, rate: int) -> SharpSounds:
    """Every short, sharp sound of ``audio`` (mono, ``rate`` Hz, first sample at ``start``)."""
    empty = SharpSounds(np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool))
    transient = np.abs(np.diff(audio, prepend=audio[:1])) if audio.size else audio
    frame = max(int(rate * FRAME_S), 1)
    n = transient.size // frame
    if n < 40:
        return empty
    env = transient[: n * frame].reshape(n, frame).max(axis=1).astype(float)

    # Background: median of each second, smoothed over the five around it.
    block = int(round(1.0 / FRAME_S))
    n_blocks = max(n // block, 1)
    medians = np.array([np.median(env[i * block:(i + 1) * block]) for i in range(n_blocks)])
    padded = np.pad(medians, 2, mode="edge")
    smooth = np.median(np.lib.stride_tricks.sliding_window_view(padded, 5), axis=1)
    noise = smooth[np.minimum(np.arange(n) // block, n_blocks - 1)] + 1e-9

    # Tops of their 100-ms neighbourhood, well above the background.
    half = int(round(0.05 / FRAME_S))
    local_max = np.lib.stride_tricks.sliding_window_view(np.pad(env, half), 2 * half + 1).max(axis=1)
    peaks = np.flatnonzero((env >= local_max) & (env >= MIN_RATIO * noise))

    sustain = (int(round(SUSTAIN_FROM_S / FRAME_S)), int(round(SUSTAIN_TO_S / FRAME_S)))
    attack = (int(round(ATTACK_FROM_S / FRAME_S)), int(round(ATTACK_TO_S / FRAME_S)))
    kept, last = [], -10 * half
    for k in peaks:
        if k - last <= 2 * half:
            continue  # same sound (a flat top)
        value = env[k]
        after = env[k + sustain[0]:k + sustain[1]]
        if after.size and float(np.median(after)) > MAX_SUSTAIN * value:
            continue  # a beep or a word, not an impact
        before = env[max(k - attack[0], 0):max(k - attack[1], 0)]
        if before.size and float(np.median(before)) > MAX_ATTACK * value:
            continue  # inside a longer sound
        kept.append(k)
        last = k
    if not kept:
        return empty

    onsets, strength = [], []
    for k in kept:
        # Sample-level onset: first sample at half of the peak.
        i0 = max((k - 4) * frame, 0)
        i1 = min((k + 1) * frame, transient.size)
        piece = transient[i0:i1]
        peak = int(np.argmax(piece))
        threshold = 0.5 * piece[peak]
        onset = peak
        while onset > 0 and piece[onset - 1] >= threshold:
            onset -= 1
        onsets.append(start + (i0 + onset) / rate)
        strength.append(float(env[k]))
    onsets_s, strength_a = np.array(onsets), np.array(strength)

    isolated = np.ones(onsets_s.size, dtype=bool)
    for i, t in enumerate(onsets_s):
        d = np.abs(onsets_s - t)
        near = (d >= EXCLUSION_S) & (d <= CONTEXT_S)
        if near.any() and strength_a[near].max() >= 0.5 * strength_a[i]:
            isolated[i] = False
    return SharpSounds(onsets_s, strength_a, isolated)


def match_sounds(
    sounds: SharpSounds,
    windows_video_s: dict[str, tuple[float, float]],
    expected_video_s: Optional[dict[str, float]] = None,  # noqa: UP045
    search_s: float = 0.0,
) -> AudioMatch:
    """
    Pick the clap or landing of each window among the sharp sounds.

    With ``search_s`` the isolated sounds first vote for a shift of the
    whole video (up to ``search_s`` seconds), which finds a video whose
    start time was not logged. Then, inside each window, the sharp sound
    closest to the expected time is taken (the loudest without one), never
    a sound already given to another event.
    """
    from algorithms.offset_search import CUE_TOLERANCE_S, vote_offset

    expected_video_s = expected_video_s or {}
    match = AudioMatch(found={})
    if sounds.onsets_s.size == 0:
        return match

    if search_s > 0 and len(windows_video_s) >= 2:
        events = []
        for event_id, (v0, v1) in windows_video_s.items():
            if event_id in expected_video_s:
                events.append((event_id, expected_video_s[event_id] * 1000.0, CUE_TOLERANCE_S))
            else:
                events.append((event_id, 500.0 * (v0 + v1), 0.5 * (v1 - v0) + 0.5))
        vote = vote_offset(sounds.onsets_s[sounds.isolated] * 1000.0, events, search_s)
        if vote.accepted:
            match.shift_s, match.searched = vote.offset_ms / 1000.0, True

    shift = match.shift_s
    used: list[float] = []
    order = sorted(windows_video_s, key=lambda e: expected_video_s.get(e, windows_video_s[e][0]))
    for event_id in order:
        v0, v1 = windows_video_s[event_id]
        inside = np.flatnonzero((sounds.onsets_s >= v0 + shift) & (sounds.onsets_s <= v1 + shift))
        inside = [i for i in inside if all(abs(sounds.onsets_s[i] - u) > SAME_SOUND_S for u in used)]
        if not inside:
            continue
        strongest = max(sounds.strength[i] for i in inside)
        inside = [i for i in inside if sounds.strength[i] >= 0.3 * strongest]
        target = expected_video_s.get(event_id)
        if target is not None:
            pick = min(inside, key=lambda i: abs(sounds.onsets_s[i] - (target + shift)))
        else:
            pick = max(inside, key=lambda i: sounds.strength[i])
        match.found[event_id] = float(sounds.onsets_s[pick])
        used.append(float(sounds.onsets_s[pick]))
    return match


def find_audio_events(
    path: str | Path,
    windows_video_s: dict[str, tuple[float, float]],
    rate: int = 4000,
    expected_video_s: Optional[dict[str, float]] = None,  # noqa: UP045
    search_s: float = 0.0,
) -> AudioMatch:
    """
    Find the clap or the landing of a jump of each window (video seconds) in
    the sound of a video. Only short, sharp sounds count: the beeps of the
    countdown and the voice are left out. Times are onsets (the first sample
    that reaches half of the peak), which mark the impact better than the peak.
    """
    audio, start = load_audio_envelope(path, rate)
    return match_sounds(sharp_sounds(audio, start, rate), windows_video_s, expected_video_s, search_s)


def detect_audio_claps(
    path: str | Path,
    windows_video_s: dict[str, tuple[float, float]],
    rate: int = 4000,
    expected_video_s: Optional[dict[str, float]] = None,  # noqa: UP045
) -> dict[str, float]:
    """event_id -> onset (video seconds) of the clap or landing of each window."""
    return find_audio_events(path, windows_video_s, rate, expected_video_s).found
