"""
Build a synthetic session to try SAVES Studio without real data.

Six minutes with four activities and six synchronization claps, recorded by
a Polar H10 (host clock), an AX3 on the wrist and a Parmay Matrix and a
SENS on other sites, each offline clock with its own offset and drift, and
a video (with a flash and a click at every clap) whose logged start time is
a quarter of a second off.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tests.synthetic import iso, write_cwa, write_matrix_bin, write_polar_acc_csv, write_sens_csv

START = 1_793_089_800.0  # 2026-10-27T08:30:00Z
DURATION = 360.0
ACTIVITIES = [
    (1, "Sentado en reposo", "07021", 20.0, 80.0, 0.00, 0.0),
    (2, "Caminar 4,8 km/h", "17190", 100.0, 170.0, 0.35, 1.9),
    (3, "Correr 8 km/h", "12030", 190.0, 260.0, 1.10, 2.7),
    (4, "Subir escaleras", "17133", 280.0, 340.0, 0.55, 1.6),
]
# (event_id, phase, window start offset); claps fall ~4 s after the tap.
WINDOWS = [
    ("sync_1", "static_prep", 8.0),
    ("sync_2", "static_post", 84.0),
    ("sync_3", "static_prep", 176.0),
    ("sync_4", "static_post", 263.0),
    ("sync_5", "transition_lap", 270.0),
    ("sync_6", "static_post", 344.0),
]
CLAPS = [START + w[2] + 4.0 + 0.11 * i for i, w in enumerate(WINDOWS)]
VIDEO_START = START + 3.0
VIDEO_LOG_ERROR = 0.25


def body_signal(t: np.ndarray, seed: int, gain: float = 1.0, clap_ms: float = 12.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xyz = np.zeros((t.size, 3))
    xyz[:, 2] = 1.0
    xyz += rng.normal(0, 0.01, xyz.shape)
    for _, _, _, a0, a1, amp, freq in ACTIVITIES:
        mask = (t >= START + a0) & (t <= START + a1)
        if amp:
            phase = 2 * np.pi * freq * t[mask]
            xyz[mask, 0] += gain * amp * np.sin(phase)
            xyz[mask, 1] += gain * 0.5 * amp * np.sin(2 * phase + 0.7)
            xyz[mask, 2] += gain * 0.8 * amp * np.abs(np.sin(phase))
    for clap in CLAPS:
        xyz[:, 0] += 4.5 * impact(t - clap, clap_ms / 1000.0)
    return xyz


def impact(dt: np.ndarray, duration_s: float) -> np.ndarray:
    """An impact that starts at dt = 0: a fast rise and a decay."""
    rise = 0.15 * duration_s
    after = np.maximum(dt - rise, 0.0)
    return np.where(dt < 0, 0.0, np.where(dt < rise, dt / rise, np.exp(-after / duration_s)))


def write_demo_manifest(path: Path) -> None:
    manifest = {
        "schema_version": "1.1",
        "session_id": "demo_session",
        "created_by": "saves_studio_demo",
        "session_start_time_utc": iso(START),
        "session_end_time_utc": iso(START + DURATION),
        "protocol": {"protocol_id": "demo", "protocol_version": "1"},
        "participants": [{"participant_id": "DEMO01", "redcap_event_name": "da_0__visita_inici_arm_1"}],
        "activities": [
            {
                "activity_index": idx,
                "activity_name": name,
                "compendium_code": code,
                "activity_start_time_utc": iso(START + a0),
                "activity_end_time_utc": iso(START + a1),
                "duration_seconds": a1 - a0,
                "redcap_instance": idx,
            }
            for idx, name, code, a0, a1, _, _ in ACTIVITIES
        ],
        "sync_windows": [
            {
                "event_id": eid,
                "event_type": "manual_sync",
                "activity_index": min(i // 2 + 1, 4),
                "phase": phase,
                "window_start_time_utc": iso(START + w0),
                "window_end_time_utc": iso(START + w0 + 10.0),
            }
            for i, (eid, phase, w0) in enumerate(WINDOWS)
        ],
        "video": {"files": ["camara_1.mp4"]},
        "sensors": {"polar_h10": {"present": True, "timestamp_domain": "native_plus_host_utc"}},
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_demo_video(path: Path, fps: int = 15, size: tuple[int, int] = (320, 180)) -> None:
    import av

    n_frames = int((DURATION - 6.0) * fps)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = 400_000
        audio = container.add_stream("aac", rate=16000)
        audio.layout = "mono"

        for i in range(n_frames):
            t = VIDEO_START + i / fps
            level = 40
            for _, _, _, a0, a1, amp, _ in ACTIVITIES:
                if START + a0 <= t <= START + a1:
                    level = 60 + int(80 * amp)
            img = np.full((size[1], size[0], 3), level, dtype=np.uint8)
            # Moving bar so playback is visible.
            x = int((i * 4) % size[0])
            img[:, x:x + 6] = 200
            if any(0 <= t - c < 1.0 / fps for c in CLAPS):
                img[:] = 255
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

        rate = 16000
        n = int((DURATION - 6.0) * rate)
        t = VIDEO_START + np.arange(n) / rate
        rng = np.random.default_rng(5)
        samples = rng.normal(0, 0.005, n).astype(np.float32)
        for clap in CLAPS:
            k = (t >= clap) & (t < clap + 0.03)
            samples[k] += (0.8 * np.exp(-(t[k] - clap) * 150) * rng.choice([-1, 1], k.sum())).astype(np.float32)
        block = 1024
        for start in range(0, n, block):
            chunk = samples[start:start + block]
            frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="flt", layout="mono")
            frame.sample_rate = rate
            frame.pts = start
            for packet in audio.encode(frame):
                container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)


def build_demo_session(folder: str | Path, with_video: bool = True) -> Path:
    """Write the demo session into ``folder`` and return it."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_demo_manifest(folder / "session_manifest.json")

    t_ref = np.arange(START - 5, START + DURATION + 5, 1 / 200)
    write_polar_acc_csv(folder / "polar_acc_raw_200hz_demo.csv", t_ref * 1000, body_signal(t_ref, 1, 0.8))

    # AX3 on the non-dominant wrist: 0.9 s ahead, 120 ppm fast (Madrid local clock).
    t_true = np.arange(START - 5, START + DURATION + 5, 1 / 100)
    drift = 120e-6
    t_dev = t_true + 0.9 + drift * (t_true - t_true[0])
    start_local = datetime.fromtimestamp(t_dev[0] + 3600, tz=timezone.utc).replace(tzinfo=None)
    write_cwa(folder / "R4H_DEMO_wrist_nd.cwa", start_local, body_signal(t_true, 2), true_frequency=100 / (1 + drift), fractional=True)

    # Matrix (UTC clock, whole-second packages) 2 s behind.
    t_mx = np.arange(START - 4, START + DURATION + 5, 1 / 100)
    write_matrix_bin(folder / "matrix_muneca.bin", int(START - 4 - 1.6), body_signal(t_mx, 3))

    # SENS on the thigh at 12.5 Hz, 0.4 s ahead and 60 ppm slow; the sensor
    # is tapped, a longer impact than a clap, so 12.5 Hz still catches it.
    t_s = np.arange(START - 5, START + DURATION + 5, 1 / 12.5)
    t_s_dev = t_s + 0.4 - 60e-6 * (t_s - t_s[0])
    write_sens_csv(folder / "sens_muslo.csv", t_s_dev * 1000, body_signal(t_s, 4, 0.6, clap_ms=160.0))

    if with_video:
        write_demo_video(folder / "camara_1.mp4")
        with (folder / "video_sync_session_demo.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file_path", "participant_id", "activity_index", "recording_start_time_utc", "recording_end_time_utc", "duration_seconds"])
            logged = VIDEO_START + VIDEO_LOG_ERROR
            writer.writerow([
                str(folder / "camara_1.mp4"), "DEMO01", 1,
                datetime.fromtimestamp(logged, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                datetime.fromtimestamp(logged + DURATION - 6, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                f"{DURATION - 6:.3f}",
            ])
    return folder
