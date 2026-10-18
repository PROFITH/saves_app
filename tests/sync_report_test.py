import json

import numpy as np
import pandas as pd
import pytest

import cli
from importers.manifest_importer import ManifestImporter
from pipeline.sync_report import synchronize_session
from readers import read_sensor
from tests.synthetic import clap_signal, write_cwa, write_manifest, write_polar_acc_csv

START = 1_793_089_800.0  # 2026-10-27T08:30:00Z
END = START + 240.0

# The operator taps "sync" and the claps follow about 4 s later.
WINDOWS = [
    ("sync_1", "static_prep", START + 20.0),
    ("sync_2", "static_post", START + 110.0),
    ("sync_3", "transition_lap", START + 140.0),
    ("sync_4", "static_prep", START + 200.0),
]
CLAPS = [w[2] + 4.0 + 0.137 * i for i, w in enumerate(WINDOWS)]

OFFSET_S = 0.8      # AX3 clock ahead of UTC
DRIFT = 150e-6      # AX3 clock runs 150 ppm fast


@pytest.fixture
def session_files(tmp_path):
    manifest = tmp_path / "session_manifest.json"
    write_manifest(manifest, START, END, WINDOWS)

    # Reference: Polar ACC at 200 Hz on the host (UTC) clock.
    t_ref = np.arange(START - 5, END + 5, 1 / 200)
    polar = tmp_path / "polar_acc.csv"
    write_polar_acc_csv(polar, t_ref * 1000, clap_signal(t_ref, CLAPS, seed=1))

    # AX3 at 100 Hz whose clock is offset and drifting.
    t_true = np.arange(START - 5, END + 5, 1 / 100)
    t_device = t_true + OFFSET_S + DRIFT * (t_true - t_true[0])
    values = clap_signal(t_true, CLAPS, seed=2)
    ax3 = tmp_path / "ax3.cwa"
    # Madrid local time (UTC+1); the true rate seen by the device clock.
    start_local = pd.Timestamp(t_device[0] + 3600, unit="s").to_pydatetime()
    write_cwa(ax3, start_local, values, true_frequency=100 / (1 + DRIFT), fractional=True)

    return manifest, polar, ax3


def test_offset_and_drift_are_recovered(session_files):
    manifest, polar, ax3 = session_files
    session = ManifestImporter().load(manifest)

    results = synchronize_session(
        session,
        {"h10": read_sensor(polar), "wrist": read_sensor(ax3, timezone="Europe/Madrid")},
        reference="h10",
    )

    assert len(results["h10"].anchors) == 4
    assert results["h10"].note == "reference"

    fit = results["wrist"].fit
    assert fit["n_pairs"] == 4
    assert fit["offset_ms"] == pytest.approx(-OFFSET_S * 1000, abs=10)
    assert fit["drift_ppm"] == pytest.approx(-DRIFT * 1e6, abs=60)
    assert fit["rmse_ms"] < 10
    assert results["wrist"].coverage["coverage_pct"] > 99


def test_claps_outside_the_window_need_padding(session_files):
    manifest, polar, ax3 = session_files
    session = ManifestImporter().load(manifest)
    recordings = {"h10": read_sensor(polar), "wrist": read_sensor(ax3)}

    # A device clock 7 s ahead pushes the last clap of each window out.
    recordings["wrist"].data["timestamp_unix_ms"] += 7000

    narrow = synchronize_session(session, recordings, reference="h10", pad_seconds=0.0)
    wide = synchronize_session(session, recordings, reference="h10", pad_seconds=8.0)

    assert len(narrow["wrist"].anchors) < 4
    assert wide["wrist"].fit["offset_ms"] == pytest.approx(-7800, abs=10)


def test_cli_writes_report_anchors_and_qc(session_files, tmp_path):
    manifest, polar, ax3 = session_files
    out = tmp_path / "out"

    status = cli.main([
        "sync",
        "--manifest", str(manifest),
        "--sensor", f"h10={polar}",
        "--sensor", f"wrist={ax3}:cwa",
        "--reference", "h10",
        "--out", str(out),
        "--export-aligned",
    ])

    assert status == 0
    report = pd.read_csv(out / "sync_report.csv").set_index("sensor_id")
    assert report.loc["wrist", "n_anchors"] == 4
    assert report.loc["wrist", "rmse_ms"] < 10

    anchors = pd.read_csv(out / "anchors.csv")
    assert set(anchors["phase"]) == {"static_prep", "static_post", "transition_lap"}

    qc = json.loads((out / "qc.json").read_text(encoding="utf-8"))
    assert qc["sensors"]["wrist"]["metadata"]["device_type"] == "AX3"

    aligned = pd.read_csv(out / "wrist_aligned.csv.gz")
    clap_rows = aligned.loc[aligned["Vector Magnitude"].idxmax()]
    assert min(abs(clap_rows["timestamp_aligned_unix_ms"] / 1000 - c) for c in CLAPS) < 0.02


def test_video_reference(session_files, tmp_path):
    manifest, polar, ax3 = session_files
    events = tmp_path / "video_events.csv"
    video_start = START - 3.0
    pd.DataFrame({
        "event_id": [w[0] for w in WINDOWS],
        "timestamp_seconds": [c - video_start for c in CLAPS],
    }).to_csv(events, index=False)

    session = ManifestImporter().load(manifest)
    results = synchronize_session(
        session,
        {"wrist": read_sensor(ax3)},
        reference="video",
        video_events=pd.read_csv(events).set_index("event_id")["timestamp_seconds"].to_dict(),
    )

    assert results["wrist"].fit["drift_ppm"] == pytest.approx(-DRIFT * 1e6, abs=60)
    assert results["wrist"].fit["rmse_ms"] < 10
