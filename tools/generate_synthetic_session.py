"""
Generate a synthetic lab session for testing the SAVES pipeline.

The output mimics what lab_protocol_timer exports today plus the files
recorded by the offline accelerometers:

    session_manifest.json          (timer, same keys the timer writes)
    sync_windows.csv               (timer)
    activity_markers.csv           (timer)
    video_sync_session_<ts>.csv    (timer, video recording log)
    session_video.mp4              (timer camera)
    <participant>_sens.csv         (SENS accelerometer, native clock)
    <participant>_matrix.bin       (Parmay Matrix binary, native clock)
    ground_truth.json              (true clock offsets, drifts, clap
                                    times and activity names; never
                                    produced by real hardware)

Every sensor clock has a known offset and drift relative to true time,
and the first video frame starts a known latency after the time the
timer logs as the recording start. A correct synchronization must
recover those values from the data alone.

Usage:
    python tools/generate_synthetic_session.py [output_dir] [--no-video]
"""

import argparse
import binascii
import json
import shutil
import struct
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

SEED = 20261013

SESSION_START = datetime(2026, 10, 13, 9, 0, 0, tzinfo=timezone.utc)

# Protocol timing, matching the timer defaults (seconds).
STATIC_PREP_S = 30.0
STATIC_POST_S = 30.0
TRANSITION_S = 20.0

# Activities performed, in order. The timer only exports the index;
# the names are kept in ground_truth.json.
ACTIVITIES = [
    {"name": "Caminar", "duration_s": 60.0, "freq_hz": 1.8, "amp_g": 0.35},
    {"name": "Correr", "duration_s": 60.0, "freq_hz": 2.7, "amp_g": 1.10},
    {"name": "Sedestacion", "duration_s": 45.0, "freq_hz": 0.0, "amp_g": 0.00},
]

# Clap time inside each static phase, relative to the phase start.
# The timer prompts the prep clap after a 5 s countdown at the end of the
# initial static phase, and the post clap in its first 5 s.
PREP_CLAP_OFFSET_S = STATIC_PREP_S - 3.0
POST_CLAP_OFFSET_S = 2.5

# The researcher presses the sync button this long before the clap, and
# the timer opens a fixed 10 s window.
SYNC_WINDOW_LEAD_S = 4.0
SYNC_WINDOW_LENGTH_S = 10.0

PARTICIPANTS = [
    {"participant_id": "P01", "redcap_event_name": "day_0_arm_1"},
    {"participant_id": "P02", "redcap_event_name": "day_0_arm_1"},
]

# Offline devices: native clock = true * (1 + drift) + offset.
DEVICES = [
    {
        "device_id": "sens_P01",
        "type": "SENS",
        "participant_id": "P01",
        "fs_hz": 50.0,
        "offset_s": 2.370,
        "drift_ppm": 40.0,
    },
    {
        "device_id": "matrix_P01",
        "type": "Matrix",
        "participant_id": "P01",
        "fs_hz": 100.0,
        "offset_s": -1.120,
        "drift_ppm": -25.0,
        "acc_range_g": 16,
    },
    {
        "device_id": "matrix_P02",
        "type": "Matrix",
        "participant_id": "P02",
        "fs_hz": 100.0,
        "offset_s": 0.640,
        "drift_ppm": 15.0,
        "acc_range_g": 16,
    },
]

# Video: the timer logs the recording start once the camera command
# returns; frame 0 actually starts this much later.
VIDEO_FPS = 10
VIDEO_SIZE = (640, 360)
VIDEO_RECORDING_LOGGED_AT_S = -5.0  # relative to session start
VIDEO_FIRST_FRAME_LATENCY_S = 0.420
VIDEO_TAIL_S = 5.0

CLAP_PEAK_G = 5.0
CLAP_FLASH_S = 0.3


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def at(seconds: float) -> datetime:
    return SESSION_START + timedelta(seconds=seconds)


# ---------------------------------------------------------------------------
# Protocol timeline (true time, seconds from session start)
# ---------------------------------------------------------------------------


def build_timeline() -> dict:
    t = 0.0
    activities = []
    claps = []
    for index, activity in enumerate(ACTIVITIES, start=1):
        prep_start = t
        act_start = prep_start + STATIC_PREP_S
        act_end = act_start + activity["duration_s"]
        post_end = act_end + STATIC_POST_S
        transition_end = post_end + TRANSITION_S

        activities.append({
            **activity,
            "activity_index": index,
            "prep_start": prep_start,
            "start": act_start,
            "end": act_end,
            "post_end": post_end,
        })
        claps.append({
            "event_id": f"sync_{len(claps) + 1}",
            "activity_index": index,
            "phase": "static_prep",
            "t": prep_start + PREP_CLAP_OFFSET_S,
        })
        claps.append({
            "event_id": f"sync_{len(claps) + 1}",
            "activity_index": index,
            "phase": "static_post",
            "t": act_end + POST_CLAP_OFFSET_S,
        })
        t = transition_end

    return {"activities": activities, "claps": claps, "end": t}


def phase_at(timeline: dict, t: float) -> tuple[str, dict | None]:
    for activity in timeline["activities"]:
        if activity["prep_start"] <= t < activity["start"]:
            return "static_prep", activity
        if activity["start"] <= t < activity["end"]:
            return "activity_main", activity
        if activity["end"] <= t < activity["post_end"]:
            return "static_post", activity
    return "transition_lap", None


# ---------------------------------------------------------------------------
# Acceleration model (true time -> x, y, z in g)
# ---------------------------------------------------------------------------


def acceleration(timeline: dict, t: np.ndarray, rng: np.random.Generator):
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    z = np.ones_like(t)

    for activity in timeline["activities"]:
        mask = (t >= activity["start"]) & (t < activity["end"])
        if activity["amp_g"] > 0:
            phase = 2 * np.pi * activity["freq_hz"] * (t[mask] - activity["start"])
            z[mask] += activity["amp_g"] * np.sin(phase)
            x[mask] += 0.4 * activity["amp_g"] * np.sin(phase + 0.8)
            y[mask] += 0.2 * activity["amp_g"] * np.sin(2 * phase)
        else:
            # Sitting: posture change, gravity moves to the x axis.
            x[mask] += 0.7
            z[mask] -= 0.3

    # Transitions: slow walking between stations.
    for start, end in zip(
        [a["post_end"] for a in timeline["activities"]],
        [a["prep_start"] for a in timeline["activities"][1:]] + [timeline["end"]],
    ):
        mask = (t >= start) & (t < end)
        z[mask] += 0.25 * np.sin(2 * np.pi * 1.6 * (t[mask] - start))

    # Claps: short impulse, a few samples wide.
    for clap in timeline["claps"]:
        pulse = CLAP_PEAK_G * np.exp(-0.5 * ((t - clap["t"]) / 0.012) ** 2)
        x += 0.6 * pulse
        z += pulse

    noise = 0.01
    x += rng.normal(0, noise, t.size)
    y += rng.normal(0, noise, t.size)
    z += rng.normal(0, noise, t.size)
    return x, y, z


def native_to_true(device: dict, native_s: np.ndarray) -> np.ndarray:
    return (native_s - device["offset_s"]) / (1 + device["drift_ppm"] * 1e-6)


def true_to_native(device: dict, true_s: float) -> float:
    return true_s * (1 + device["drift_ppm"] * 1e-6) + device["offset_s"]


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_sens(path: Path, device: dict, timeline: dict, rng) -> None:
    start_native = true_to_native(device, -10.0)
    end_native = true_to_native(device, timeline["end"] + 10.0)
    native = np.arange(start_native, end_native, 1.0 / device["fs_hz"])
    x, y, z = acceleration(timeline, native_to_true(device, native), rng)

    epoch = SESSION_START.timestamp()
    unixts = np.round((epoch + native) * 1000).astype(np.int64)
    with path.open("w", encoding="utf-8") as f:
        f.write("unixts,x,y,z\n")
        for row in zip(unixts, x, y, z):
            f.write(f"{row[0]},{row[1]:.4f},{row[2]:.4f},{row[3]:.4f}\n")


def write_matrix(path: Path, device: dict, timeline: dict, rng) -> None:
    fs = int(device["fs_hz"])
    acc_range = device["acc_range_g"]
    epoch = SESSION_START.timestamp()

    # Packets cover whole seconds of the device clock.
    first = int(np.floor(epoch + true_to_native(device, -10.0)))
    last = int(np.ceil(epoch + true_to_native(device, timeline["end"] + 10.0)))

    def to_raw(values: np.ndarray) -> np.ndarray:
        scaled = np.where(
            values > 0, values * 0x7FFF / acc_range, values * 0x8000 / acc_range
        )
        return np.clip(np.round(scaled), -0x8000, 0x7FFF).astype(np.int16)

    packets = []
    for t_start in range(first, last):
        native = (t_start - epoch) + np.arange(fs) / fs
        x, y, z = acceleration(timeline, native_to_true(device, native), rng)
        payload = np.column_stack([to_raw(x), to_raw(y), to_raw(z)]).astype("<i2")
        payload_bytes = payload.tobytes()

        body = struct.pack("<IIIIII", t_start, t_start + 1, fs, 0, 0, 0)
        crc = binascii.crc32(body + payload_bytes)
        packets.append(b"MDTCPACK" + struct.pack("<I", crc) + body + payload_bytes)

    remarks = f"SYNTHETIC {device['device_id']}".encode().ljust(512, b"\0")
    header = struct.pack("<4sIHH", b"MDTC", len(packets), acc_range, 2000)
    path.write_bytes(remarks + header + b"".join(packets))


def write_video(path: Path, timeline: dict, frame0_true_s: float) -> float:
    import cv2

    width, height = VIDEO_SIZE
    duration = timeline["end"] + VIDEO_TAIL_S - frame0_true_s
    n_frames = int(duration * VIDEO_FPS)

    raw_path = path.with_suffix(".raw.avi")
    writer = cv2.VideoWriter(
        str(raw_path), cv2.VideoWriter_fourcc(*"MJPG"), VIDEO_FPS, (width, height)
    )
    clap_times = [c["t"] for c in timeline["claps"]]

    for i in range(n_frames):
        t = frame0_true_s + i / VIDEO_FPS
        phase, activity = phase_at(timeline, t)
        flash = any(0 <= t - c < CLAP_FLASH_S for c in clap_times)

        frame = np.full((height, width, 3), 255 if flash else 40, np.uint8)
        color = (0, 0, 0) if flash else (230, 230, 230)

        label = activity["name"] if activity else "Transicion"
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(frame, phase, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(
            frame, f"frame {i}", (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1,
        )
        if flash:
            cv2.putText(
                frame, "CLAP", (width // 2 - 90, height // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6,
            )

        # A moving dot so motion is visible during activities.
        if phase == "activity_main" and activity["amp_g"] > 0:
            offset = np.sin(2 * np.pi * activity["freq_hz"] * t)
            cy = int(height * 0.65 + 40 * offset)
        else:
            cy = int(height * 0.65)
        cv2.circle(frame, (width - 80, cy), 18, (80, 200, 80), -1)

        writer.write(frame)
    writer.release()

    if shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", str(path),
            ],
            check=True,
        )
        raw_path.unlink()
    else:
        raw_path.rename(path.with_suffix(".avi"))
        path = path.with_suffix(".avi")

    return n_frames / VIDEO_FPS


def write_timer_exports(out: Path, timeline: dict, video_name: str | None) -> None:
    session_id = f"session_{int(SESSION_START.timestamp() * 1_000_000)}"

    activities = [
        {
            "activity_index": a["activity_index"],
            "activity_start_time_utc": iso(at(a["start"])),
            "activity_end_time_utc": iso(at(a["end"])),
            "duration_seconds": a["end"] - a["start"],
        }
        for a in timeline["activities"]
    ]

    # Same keys as SyncWindow.toJson() in lab_protocol_timer.
    sync_windows = [
        {
            "event_id": c["event_id"],
            "event_type": "manual_sync",
            "activity_index": c["activity_index"],
            "phase": c["phase"],
            "window_start": iso(at(c["t"] - SYNC_WINDOW_LEAD_S)),
            "window_end": iso(
                at(c["t"] - SYNC_WINDOW_LEAD_S + SYNC_WINDOW_LENGTH_S)
            ),
        }
        for c in timeline["claps"]
    ]

    manifest = {
        "schema_version": "1.0",
        "session_id": session_id,
        "created_by": "lab_protocol_timer",
        "session_start_time_utc": iso(SESSION_START),
        "session_end_time_utc": iso(at(timeline["end"])),
        "participants": PARTICIPANTS,
        "activities": activities,
        "sync_windows": sync_windows,
        "video": {"files": [video_name] if video_name else []},
        "sensors": {
            "polar_h10": {
                "present": True,
                "timestamp_domain": "native_plus_host_utc",
            },
            "offline_sensor_a": {"present": False, "timestamp_domain": "unknown"},
            "offline_sensor_b": {"present": False, "timestamp_domain": "unknown"},
        },
    }
    (out / "session_manifest.json").write_text(json.dumps(manifest, indent=2))

    with (out / "sync_windows.csv").open("w", encoding="utf-8") as f:
        f.write("event_id,event_type,activity_index,phase,window_start,window_end\n")
        for w in sync_windows:
            f.write(
                f"{w['event_id']},{w['event_type']},{w['activity_index']},"
                f"{w['phase']},{w['window_start']},{w['window_end']}\n"
            )

    with (out / "activity_markers.csv").open("w", encoding="utf-8") as f:
        f.write(
            "activity_index,activity_start_time_utc,"
            "activity_end_time_utc,duration_seconds\n"
        )
        for a in activities:
            f.write(
                f"{a['activity_index']},{a['activity_start_time_utc']},"
                f"{a['activity_end_time_utc']},{a['duration_seconds']:.6f}\n"
            )


def write_video_log(out: Path, video_name: str, video_duration_s: float) -> None:
    logged_start = at(VIDEO_RECORDING_LOGGED_AT_S)
    logged_end = at(
        VIDEO_RECORDING_LOGGED_AT_S + VIDEO_FIRST_FRAME_LATENCY_S + video_duration_s
    )
    stamp = SESSION_START.strftime("%Y%m%d_%H%M%S")
    with (out / f"video_sync_session_{stamp}.csv").open("w", encoding="utf-8") as f:
        f.write(
            "file_path,participant_id,activity_index,"
            "recording_start_time_utc,recording_end_time_utc,duration_seconds\n"
        )
        f.write(
            f"{video_name},ALL,0,{iso(logged_start)},{iso(logged_end)},"
            f"{(logged_end - logged_start).total_seconds():.6f}\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "output_dir", nargs="?", default="tmp/synthetic_session_01",
        help="Folder to write the session into (default: %(default)s)",
    )
    parser.add_argument("--no-video", action="store_true", help="Skip the video")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    timeline = build_timeline()

    sensor_files = {}
    for device in DEVICES:
        if device["type"] == "SENS":
            name = f"{device['participant_id']}_sens.csv"
            write_sens(out / name, device, timeline, rng)
        else:
            name = f"{device['participant_id']}_matrix.bin"
            write_matrix(out / name, device, timeline, rng)
        sensor_files[device["device_id"]] = name

    frame0_true_s = VIDEO_RECORDING_LOGGED_AT_S + VIDEO_FIRST_FRAME_LATENCY_S
    video_name = None
    if not args.no_video:
        video_duration = write_video(out / "session_video.mp4", timeline, frame0_true_s)
        video_name = (out / "session_video.mp4").name
        if not (out / video_name).exists():
            video_name = "session_video.avi"
        write_video_log(out, video_name, video_duration)

    write_timer_exports(out, timeline, video_name)

    ground_truth = {
        "note": "Synthetic session. Not produced by real hardware.",
        "seed": SEED,
        "session_start_time_utc": iso(SESSION_START),
        "video": None if video_name is None else {
            "file": video_name,
            "fps": VIDEO_FPS,
            "recording_start_logged_utc": iso(at(VIDEO_RECORDING_LOGGED_AT_S)),
            "first_frame_true_utc": iso(at(frame0_true_s)),
            "first_frame_latency_seconds": VIDEO_FIRST_FRAME_LATENCY_S,
        },
        "activities": [
            {
                "activity_index": a["activity_index"],
                "name": a["name"],
                "start_true_utc": iso(at(a["start"])),
                "end_true_utc": iso(at(a["end"])),
            }
            for a in timeline["activities"]
        ],
        "claps": [
            {
                "event_id": c["event_id"],
                "activity_index": c["activity_index"],
                "phase": c["phase"],
                "true_utc": iso(at(c["t"])),
                "video_seconds": round(c["t"] - frame0_true_s, 6),
                "sensor_unix_ms": {
                    d["device_id"]: round(
                        (SESSION_START.timestamp() + true_to_native(d, c["t"])) * 1000
                    )
                    for d in DEVICES
                },
            }
            for c in timeline["claps"]
        ],
        "devices": [
            {**d, "file": sensor_files[d["device_id"]]} for d in DEVICES
        ],
    }
    (out / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))

    print(f"Synthetic session written to {out.resolve()}")
    for path in sorted(out.iterdir()):
        print(f"  {path.name:40s} {path.stat().st_size / 1024:9.1f} KB")


if __name__ == "__main__":
    main()
