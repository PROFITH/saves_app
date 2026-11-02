"""
A RUN4HEALTH session as lab_protocol_timer writes it, with simulated sensors.

The folder files (manifest, sync windows, activity markers and event log)
come from a real run of the timer (tests/fixtures/lpt_run4health). The
sensor files are simulated: each body site moves with the activities of the
protocol and feels the claps and jumps of the cues, and each offline clock
has its own error, one of them a device left in UTC (two hours off). The
live Polar devices also get the other files the timer writes for them (heart
rate and ECG for the H10; heart rate, PPG, PPI, gyroscope and magnetometer
for the Verity Sense), on the same device clock as their acceleration.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tests.synthetic import (
    write_cwa,
    write_matrix_bin,
    write_polar_acc_csv,
    write_polar_ecg_csv,
    write_polar_gyro_csv,
    write_polar_hr_csv,
    write_polar_mag_csv,
    write_polar_ppg_csv,
    write_polar_ppi_csv,
    write_sens_csv,
)

FIXTURE = Path(__file__).parent / "fixtures" / "lpt_run4health"
MADRID_SUMMER = 2 * 3600  # 21 October 2026 is still CEST.
MARGIN_S = 300.0

# Movement of each activity: (frequency Hz, gain per site).
ACTIVITY_MOTION = {
    1: (0.0, {}),
    2: (0.0, {}),
    3: (4.0, {"wrist_nd": 0.05, "wrist_d": 0.06}),
    4: (1.8, {"thigh": 0.6, "chest": 0.3, "wrist_nd": 0.4, "wrist_d": 0.4, "arm": 0.3}),
    5: (1.0, {"thigh": 0.2, "chest": 0.15, "wrist_nd": 0.6, "wrist_d": 0.7, "arm": 0.4}),
    6: (1.6, {"thigh": 0.8, "chest": 0.4, "wrist_nd": 0.5, "wrist_d": 0.5, "arm": 0.4}),
    7: (2.7, {"thigh": 1.5, "chest": 1.0, "wrist_nd": 1.0, "wrist_d": 1.0, "arm": 0.9}),
    8: (0.5, {"thigh": 0.7, "chest": 0.3, "wrist_nd": 0.2, "wrist_d": 0.2, "arm": 0.2}),
    9: (2.0, {"thigh": 0.8, "chest": 0.4, "wrist_nd": 0.7, "wrist_d": 0.7, "arm": 0.5}),
}
# Peak (g) of a clap at each site: the lower wrist rests on the thigh; the
# chest strap barely moves.
CLAP_G = {"wrist_nd": 4.5, "wrist_d": 4.5, "thigh": 2.5, "arm": 2.2, "chest": 0.12}
# The impact lasts longer where it arrives through soft tissue (s).
CLAP_S = {"wrist_nd": 0.012, "wrist_d": 0.012, "thigh": 0.02, "arm": 0.035, "chest": 0.04}
LANDING_G = {"wrist_nd": 3.0, "wrist_d": 3.0, "thigh": 4.0, "arm": 3.2, "chest": 3.0}

# The Verity Sense delivers the PPI pulses in batches, the first one about
# 25 s after the start.
PPI_BATCH_S = 5.0
PPI_DELAY_S = 25.0

# sensor_id: (site, format, rate Hz, clock offset s, drift ppm)
SENSORS = {
    "polar_h10": ("chest", "polar_acc", 200.0, 0.004, 0.0),
    "verity_arm": ("arm", "polar_acc", 52.0, -0.006, 0.0),
    "ax3_wnd": ("wrist_nd", "cwa_utc", 100.0, -MADRID_SUMMER + 0.6, 80.0),
    "ax3_wd": ("wrist_d", "cwa", 100.0, 1.3, -40.0),
    "ax3_thigh": ("thigh", "cwa", 100.0, 45.0, 30.0),
    "sens_thigh": ("thigh", "sens", 12.5, 0.4, -60.0),
    "matrix_wnd": ("wrist_nd", "matrix", 100.0, -2.0, 0.0),
}


def _iso_seconds(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def load_manifest() -> dict:
    return json.loads((FIXTURE / "session_manifest.json").read_text(encoding="utf-8"))


def sync_events(manifest: dict, seed: int = 7) -> list[tuple[str, str, float]]:
    """(event_id, kind, true host time) of each clap or jump landing."""
    rng = np.random.default_rng(seed)
    events = []
    for window in manifest["sync_windows"]:
        cue = _iso_seconds(window["cue_time_utc"])
        kind = window.get("sync_kind", "clap")
        delay = rng.uniform(0.25, 0.5) if kind == "clap" else rng.uniform(0.75, 1.15)
        events.append((window["event_id"], kind, cue + delay))
    return events


def body(t: np.ndarray, site: str, manifest: dict, events, seed: int) -> np.ndarray:
    """Acceleration (g) of one body site at the true host times ``t``."""
    rng = np.random.default_rng(seed)
    xyz = np.zeros((t.size, 3))
    xyz[:, 2] = 1.0
    xyz += rng.normal(0.0, 0.01, xyz.shape)

    for activity in manifest["activities"]:
        start = _iso_seconds(activity["activity_start_time_utc"])
        end = _iso_seconds(activity["activity_end_time_utc"])
        freq, gains = ACTIVITY_MOTION.get(activity["activity_index"], (0.0, {}))
        gain = gains.get(site, 0.0)
        if not gain:
            continue
        mask = (t >= start) & (t <= end)
        phase = 2 * np.pi * freq * t[mask]
        xyz[mask, 0] += gain * np.sin(phase)
        xyz[mask, 1] += 0.5 * gain * np.sin(2 * phase + 0.7)
        # Steps: a brief impact each cycle.
        xyz[mask, 2] += gain * np.maximum(np.sin(phase), 0.0) ** 6

    for _event_id, kind, when in events:
        dt = t - when
        if kind == "clap":
            xyz[:, 0] += CLAP_G[site] * _impact(dt, CLAP_S[site])
            continue
        # Countermovement, push, flight and landing.
        xyz[(dt >= -0.9) & (dt < -0.65), 2] = 0.6
        xyz[(dt >= -0.65) & (dt < -0.35), 2] = 1.8
        flight = (dt >= -0.33) & (dt < 0.0)
        xyz[flight] = rng.normal(0.0, 0.03, (int(flight.sum()), 3))
        if site.startswith("wrist"):
            xyz[flight, 0] += 0.25  # The arms still move in the air.
        xyz[:, 2] += LANDING_G[site] * _impact(dt, 0.04)
    return xyz


def heart_rate_bpm(t: np.ndarray, manifest: dict) -> np.ndarray:
    """Heart rate (bpm) at the true host times: 62 at rest, higher with the MET of each activity."""
    hr = np.full(t.size, 62.0)
    for activity in manifest["activities"]:
        start = _iso_seconds(activity["activity_start_time_utc"])
        end = _iso_seconds(activity["activity_end_time_utc"])
        hr[(t >= start) & (t <= end)] = 62.0 + 14.0 * (float(activity.get("met", 1.0)) - 1.0)
    return hr


def write_polar_companions(folder: Path, sensor_id: str, stamp: str, site: str, manifest: dict, events,
                           device_clock, seed: int) -> None:
    """
    The other files of a live Polar device, on its device clock
    (``device_clock`` maps true host seconds to the device's seconds).

    The chest strap (H10) writes heart rate with RR intervals and ECG; the
    arm band (Verity Sense) writes heart rate without RR, PPG, PPI in late
    batches, gyroscope and magnetometer. They cover the session plus 5 s.
    """
    rng = np.random.default_rng(seed)
    t0 = _iso_seconds(manifest["session_start_time_utc"]) - 5.0
    t1 = _iso_seconds(manifest["session_end_time_utc"]) + 5.0

    def name(kind: str) -> Path:
        return folder / f"polar_{kind}_{sensor_id}_{stamp}.csv"

    def sensor_ns(t: np.ndarray) -> np.ndarray:
        return np.round((t - t0) * 1e9).astype(np.int64)

    def beats(t: np.ndarray, rate: float) -> np.ndarray:
        return np.cumsum(heart_rate_bpm(t, manifest) / 60.0 / rate)

    t_hr = np.arange(t0, t1, 1.0)
    hr = np.round(heart_rate_bpm(t_hr, manifest) + rng.normal(0.0, 1.0, t_hr.size))
    if site == "chest":
        rr = [[60000.0 / h + rng.normal(0.0, 8.0)] for h in hr]
        write_polar_hr_csv(name("hr"), device_clock(t_hr) * 1000.0, hr, rr, hr_source="h10_ecg_firmware")
        t_ecg = np.arange(t0, t1, 1 / 130.0)
        cycle = beats(t_ecg, 130.0) % 1.0
        ecg = 1200.0 * np.exp(-((cycle - 0.5) ** 2) / 0.0008) - 150.0 * np.sin(2 * np.pi * cycle)
        write_polar_ecg_csv(name("ecg"), device_clock(t_ecg) * 1000.0, ecg + rng.normal(0.0, 20.0, t_ecg.size),
                            sensor_ns(t_ecg))
        return

    write_polar_hr_csv(name("hr"), device_clock(t_hr) * 1000.0, hr, None, hr_source="verity_ppg_firmware")
    t_ppg = np.arange(t0, t1, 1 / 55.0)
    pulse = np.sin(2 * np.pi * beats(t_ppg, 55.0))
    ppg = 300000.0 + 4000.0 * pulse[:, None] * np.array([1.0, 0.9, 1.1]) + rng.normal(0.0, 150.0, (t_ppg.size, 3))
    write_polar_ppg_csv(name("ppg"), device_clock(t_ppg) * 1000.0, ppg, 20000.0 + rng.normal(0.0, 50.0, t_ppg.size),
                        sensor_ns(t_ppg))
    # PPI: one row per pulse; the pulses of each 5-s batch share the arrival
    # time of the batch (25 s late) and index_in_frame orders them.
    pulses = [t0]
    while pulses[-1] < t1:
        pulses.append(pulses[-1] + 60.0 / float(heart_rate_bpm(np.array([pulses[-1]]), manifest)[0]))
    pulses = np.array(pulses[1:])
    ppi = np.diff(np.concatenate(([t0], pulses))) * 1000.0 + rng.normal(0.0, 5.0, pulses.size)
    batch = np.floor((pulses - t0) / PPI_BATCH_S)
    arrival = t0 + (batch + 1) * PPI_BATCH_S + PPI_DELAY_S
    index_in_frame = np.concatenate([np.arange(np.count_nonzero(batch == b)) for b in np.unique(batch)])
    write_polar_ppi_csv(
        name("ppi"), device_clock(arrival) * 1000.0, ppi, 60000.0 / ppi, rng.integers(5, 40, pulses.size),
        blocker=(rng.random(pulses.size) < 0.05).astype(int), index_in_frame=index_in_frame,
    )
    t_gyro = np.arange(t0, t1, 1 / 52.0)
    gyro = 40.0 * (body(t_gyro, site, manifest, events, seed) - np.array([0.0, 0.0, 1.0]))
    write_polar_gyro_csv(name("gyro"), device_clock(t_gyro) * 1000.0, gyro + rng.normal(0.0, 1.0, gyro.shape),
                         sensor_ns(t_gyro))
    t_mag = np.arange(t0, t1, 1 / 50.0)
    mag = np.array([0.25, -0.10, 0.40]) + rng.normal(0.0, 0.005, (t_mag.size, 3))
    write_polar_mag_csv(name("mag"), device_clock(t_mag) * 1000.0, mag, sensor_ns(t_mag))


def _impact(dt: np.ndarray, duration_s: float) -> np.ndarray:
    rise = 0.15 * duration_s
    after = np.maximum(dt - rise, 0.0)
    return np.where(dt < 0, 0.0, np.where(dt < rise, dt / rise, np.exp(-after / duration_s)))


def build_lpt_session(folder: str | Path, sensors: dict | None = None) -> Path:
    """Copy the timer's folder files and write the simulated sensor files."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("session_manifest.json", "sync_windows.csv", "activity_markers.csv", "session_events.csv"):
        shutil.copy(FIXTURE / name, folder / name)

    manifest = load_manifest()
    events = sync_events(manifest)
    start = _iso_seconds(manifest["session_start_time_utc"]) - MARGIN_S
    end = _iso_seconds(manifest["session_end_time_utc"]) + MARGIN_S
    stamp = "20261021_103000"

    for index, (sensor_id, (site, fmt, rate, offset, drift_ppm)) in enumerate((sensors or SENSORS).items()):
        t_true = np.arange(start, end, 1.0 / rate)
        values = body(t_true, site, manifest, events, seed=index + 1)
        drift = drift_ppm * 1e-6
        t_device = t_true + offset + drift * (t_true - t_true[0])
        if fmt == "polar_acc":
            write_polar_acc_csv(folder / f"polar_acc_{sensor_id}_{stamp}.csv", t_device * 1000.0, values)
            write_polar_companions(
                folder, sensor_id, stamp, site, manifest, events,
                lambda t, offset=offset, drift=drift: t + offset + drift * (t - start), seed=index + 1,
            )
        elif fmt in ("cwa", "cwa_utc"):
            # The device writes local time and SAVES reads it as Madrid time,
            # so a device left in UTC ends up two hours early (its offset).
            local = t_device[0] + MADRID_SUMMER
            start_local = datetime.fromtimestamp(local, tz=timezone.utc).replace(tzinfo=None)
            write_cwa(
                folder / f"R4H_001_D0_{sensor_id}.cwa",
                start_local,
                values,
                true_frequency=rate / (1 + drift),
                fractional=True,
            )
        elif fmt == "sens":
            write_sens_csv(folder / f"R4H_001_D0_{sensor_id}.csv", t_device * 1000.0, values)
        elif fmt == "matrix":
            # Whole-second packages starting at the device clock.
            first = int(np.floor(t_device[0]))
            skip = int(round((first + 1 - t_device[0]) * rate))
            write_matrix_bin(folder / f"R4H_001_D0_{sensor_id}.bin", first + 1, values[skip:])
    return folder


def true_offsets_ms(at_unix_s: float | None = None) -> dict[str, float]:
    """
    Sensor clock minus host clock (ms) at ``at_unix_s`` (true host time),
    or at the start of the recordings.
    """
    manifest = load_manifest()
    t0 = _iso_seconds(manifest["session_start_time_utc"]) - MARGIN_S
    elapsed = 0.0 if at_unix_s is None else at_unix_s - t0
    return {sid: (spec[3] + spec[4] * 1e-6 * elapsed) * 1000.0 for sid, spec in SENSORS.items()}
