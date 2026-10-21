"""Tests of SAVES Studio's headless core on the demo session (sensors with known clock errors)."""

import json

import numpy as np
import pandas as pd
import pytest

from importers.manifest_importer import ManifestImporter
from studio.demo import CLAPS, VIDEO_START, build_demo_session
from studio.discovery import discover, guess_location
from studio.engine import VIDEO, SyncEngine
from studio.project import SyncProject
from studio.signals import Envelope, load_signal


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    folder = tmp_path_factory.mktemp("demo")
    build_demo_session(folder)
    return folder


@pytest.fixture()
def engine(demo):
    found = discover(demo)
    project = SyncProject(folder=str(demo), manifest_path=found.manifest_path, videos=found.videos)
    for entry in found.sensors:
        project.add_sensor(entry)
    session = ManifestImporter().load(project.manifest_path)
    engine = SyncEngine(project, session)
    for entry in project.sensors:
        engine.set_signal(entry.sensor_id, load_signal(entry.path, entry.sensor_format, demo))
    engine.refit()
    return engine


def test_discover_finds_every_file_and_guesses_locations(demo):
    found = discover(demo)
    assert found.manifest_path.endswith("session_manifest.json")
    formats = sorted(s.sensor_format for s in found.sensors)
    assert formats == ["cwa", "matrix", "polar_acc", "sens"]
    by_format = {s.sensor_format: s for s in found.sensors}
    assert by_format["cwa"].location == "Muñeca no dominante"
    assert by_format["cwa"].serial.startswith("AX")
    assert by_format["sens"].location == "Muslo"
    assert len(found.videos) == 1
    # The video log is written 0.25 s late on purpose.
    assert found.videos[0].recording_start_unix_s == pytest.approx(VIDEO_START + 0.25, abs=1e-3)


def test_guess_location_reads_spanish_and_english_tokens():
    assert guess_location("P01_muneca_dominante.cwa", "cwa") == "Muñeca dominante"
    assert guess_location("P01_thigh.cwa", "cwa") == "Muslo"
    assert guess_location("P01_muneca_no_dominante.cwa", "cwa") == "Muñeca no dominante"
    assert guess_location("whatever.csv", "polar_acc") == "Pecho"


def test_envelope_keeps_extremes_when_decimating():
    t = np.arange(1_000_000, dtype=float)
    values = np.zeros(t.size, dtype=np.float32)
    values[123_456] = 9.0
    values[654_321] = -9.0
    env = Envelope.build(t, values)
    x, y = env.query(0, t[-1], max_points=2000)
    assert x.size <= 4000
    assert y.max() == 9.0 and y.min() == -9.0
    x, y = env.query(1000, 1100, max_points=2000)
    assert x.size == 102  # short spans return the raw samples (plus one before)


def test_signal_cache_is_reused(demo):
    path = next(p for p in demo.iterdir() if p.suffix == ".cwa")
    first = load_signal(path, "cwa", demo)
    cached = list((demo / ".saves_cache").glob(f"{path.stem}_*.npz"))
    assert len(cached) == 1
    second = load_signal(path, "cwa", demo)
    np.testing.assert_array_equal(first.t_ms, second.t_ms)
    assert second.sampling_rate_hz == pytest.approx(100, rel=0.01)


def test_engine_recovers_known_clock_errors(engine):
    assert engine.resolved_reference() == "h10"
    expected_offset_s = {"ax_wrist_nd": -0.9, "matrix": 2.0, "sens": -0.4}
    for sid, offset in expected_offset_s.items():
        status = engine.status(sid)
        assert status.quality == "good", sid
        assert status.n_pairs == 6
        assert status.offset_ms / 1000 == pytest.approx(offset, abs=0.05), sid
    assert engine.status("ax_wrist_nd").drift_ppm == pytest.approx(-120, abs=5)
    assert engine.status("h10").quality == "reference"


def test_manual_anchor_and_exclusion_change_the_fit(engine):
    sid, eid = "ax_wrist_nd", "sync_3"
    auto = engine.anchor(sid, eid).sensor_timestamp_unix_ms
    engine.project.set_manual_anchor(sid, eid, auto + 500)
    engine.refit()
    assert engine.anchor_state(sid, eid) == "manual"
    assert engine.worst_event(sid) == eid
    assert engine.status(sid).rmse_ms > 100
    engine.project.set_excluded(sid, eid, True)
    engine.refit()
    assert engine.anchor_state(sid, eid) == "excluded"
    assert engine.status(sid).n_pairs == 5
    assert engine.status(sid).rmse_ms < 25
    engine.project.set_excluded(sid, eid, False)
    engine.project.clear_manual_anchor(sid, eid)
    engine.refit()
    assert engine.anchor_state(sid, eid) == "auto"


def test_video_clock_from_marked_claps(engine):
    assert engine.video_fit_source == "recording_start"
    # Mark every clap at its true video time: the 0.25 s log error disappears.
    engine.project.video_events = {
        w.event_id: clap - VIDEO_START for w, clap in zip(engine.session.sync_windows, CLAPS)
    }
    engine.refit()
    assert engine.video_fit_source == "claps"
    assert float(engine.video_to_display(10.0)) == pytest.approx(VIDEO_START + 10.0, abs=0.01)
    assert max(abs(v) for v in engine.video_residuals_ms().values()) < 10
    engine.project.reference = VIDEO
    engine.refit()
    assert engine.resolved_reference() == VIDEO
    assert engine.status("ax_wrist_nd").quality == "good"


def test_sensor_and_display_clocks_round_trip(engine):
    t = engine.session.start_time.timestamp() + 100.0
    for sid in engine.active_sensors():
        back = engine.sensor_to_display(sid, engine.display_to_sensor(sid, t))
        assert float(back) == pytest.approx(t, abs=1e-6)


def test_export_writes_reports(engine, tmp_path):
    written = {p.name for p in engine.export(tmp_path, export_aligned=True)}
    assert {"sync_report.csv", "anchors.csv", "qc.json", "activities_ax_wrist_nd.csv"} <= written
    assert "ax_wrist_nd_aligned.csv.gz" in written
    report = pd.read_csv(tmp_path / "sync_report.csv")
    assert set(report["sensor_id"]) == {"h10", "ax_wrist_nd", "matrix", "sens"}
    assert "location" in report.columns
    qc = json.loads((tmp_path / "qc.json").read_text(encoding="utf-8"))
    assert qc["reference"] == "h10"


def test_project_round_trip_keeps_relative_paths(engine, demo):
    engine.project.set_manual_anchor("sens", "sync_2", 123.0)
    engine.project.video_events = {"sync_1": 4.5}
    engine.project.save()
    raw = json.loads((demo / "saves_project.json").read_text(encoding="utf-8"))
    assert not any(s["path"].startswith("/") for s in raw["sensors"])
    loaded = SyncProject.load(demo)
    assert [s.sensor_id for s in loaded.sensors] == [s.sensor_id for s in engine.project.sensors]
    assert loaded.manual_anchors["sens"]["sync_2"] == 123.0
    assert loaded.video_events == {"sync_1": 4.5}
    (demo / "saves_project.json").unlink()


def test_video_frames_and_audio_claps(engine):
    pytest.importorskip("av")
    from studio.video import VideoReader, detect_audio_claps

    video = engine.project.videos[0].path
    reader = VideoReader(video)
    assert reader.fps == pytest.approx(15)
    clap_video_s = CLAPS[0] - VIDEO_START
    frame = reader.frame_at(clap_video_s + 0.01)
    assert frame.rgb.mean() > 200  # the demo flashes a white frame at each clap
    following = reader.next_frame()
    assert following.time_s == pytest.approx(frame.time_s + 1 / 15, abs=1e-3)
    reader.close()

    windows = {w.event_id: (w.window_start.timestamp() - VIDEO_START, w.window_end.timestamp() - VIDEO_START)
               for w in engine.session.sync_windows}
    claps = detect_audio_claps(video, windows)
    assert len(claps) == 6
    for w, clap in zip(engine.session.sync_windows, CLAPS):
        assert claps[w.event_id] == pytest.approx(clap - VIDEO_START, abs=0.01)


def test_main_window_opens_the_demo(demo, monkeypatch):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QThreadPool
    from PySide6.QtWidgets import QApplication

    from studio.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.open_folder(demo)
    for _ in range(200):
        app.processEvents()
        if QThreadPool.globalInstance().waitForDone(50) and window.engine is not None \
                and len(window.engine.signals) == 4:
            break
    app.processEvents()
    assert len(window.engine.signals) == 4
    assert window.engine.resolved_reference() == "h10"
    window.close()


def test_audio_claps_ignore_the_beeps_of_the_countdown(tmp_path):
    av = pytest.importorskip("av")
    from studio.video import detect_audio_claps

    rate = 16000
    duration = 30.0
    t = np.arange(int(duration * rate)) / rate
    rng = np.random.default_rng(3)
    audio = rng.normal(0, 0.004, t.size)
    cues = [8.0, 20.0]
    claps = [cue + 0.42 for cue in cues]
    for cue in cues:
        # Loud beeps: 3-2-1 (880 Hz, 0.16 s) and the go (1320 Hz, 0.15 s).
        for k, (start, length, freq) in enumerate([(cue - 3, 0.16, 880), (cue - 2, 0.16, 880),
                                                    (cue - 1, 0.16, 880), (cue, 0.15, 1320)]):
            mask = (t >= start) & (t < start + length)
            audio[mask] += 0.9 * np.sin(2 * np.pi * freq * (t[mask] - start))
    for clap in claps:
        mask = (t >= clap) & (t < clap + 0.04)
        audio[mask] += 0.5 * np.exp(-(t[mask] - clap) * 120) * rng.choice([-1, 1], mask.sum())
    path = tmp_path / "beeps.m4a"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("aac", rate=rate)
        stream.layout = "mono"
        samples = audio.astype(np.float32)
        for start in range(0, samples.size, 1024):
            frame = av.AudioFrame.from_ndarray(samples[start:start + 1024].reshape(1, -1), format="flt", layout="mono")
            frame.sample_rate = rate
            frame.pts = start
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    windows = {f"sync_{i + 1}": (cue - 3.5, cue + 3.5) for i, cue in enumerate(cues)}
    expected = {f"sync_{i + 1}": cue + 0.35 for i, cue in enumerate(cues)}
    found = detect_audio_claps(path, windows, expected_video_s=expected)
    assert set(found) == set(windows)
    for i, clap in enumerate(claps):
        assert found[f"sync_{i + 1}"] == pytest.approx(clap, abs=0.02)


def test_audio_finds_a_video_whose_start_was_not_logged():
    from studio.video import match_sounds, sharp_sounds

    # A phone video started 95 s before the session, with steps between the
    # still phases and the countdown beeps before every clap or landing.
    rate = 4000
    lead = 95.0
    cues = [30.0, 95.0, 170.0, 240.0, 330.0, 400.0]
    duration = 450.0 + lead
    t = np.arange(int(duration * rate)) / rate
    rng = np.random.default_rng(11)
    audio = rng.normal(0, 0.003, t.size)

    def click(at, gain, decay=150.0):
        mask = (t >= at) & (t < at + 0.04)
        audio[mask] += gain * np.exp(-(t[mask] - at) * decay) * rng.choice([-1, 1], mask.sum())

    for cue in cues:
        v = cue + lead
        for start, length, freq in [(v - 3, 0.16, 880), (v - 2, 0.16, 880), (v - 1, 0.16, 880), (v, 0.15, 1320)]:
            mask = (t >= start) & (t < start + length)
            audio[mask] += 0.9 * np.sin(2 * np.pi * freq * (t[mask] - start))
    events = [cue + lead + rng.uniform(0.3, 0.5) for cue in cues]
    for at in events:
        click(at, 0.6)
    # Steps every half second between the cues (louder than the claps).
    for a, b in zip(cues[:-1], cues[1:]):
        for step in np.arange(a + lead + 8.0, b + lead - 8.0, 0.5):
            click(step + rng.uniform(-0.03, 0.03), 0.8)

    sounds = sharp_sounds(audio.astype(np.float32), 0.0, rate)
    windows = {f"sync_{i + 1}": (cue - 1.0, cue + 3.5) for i, cue in enumerate(cues)}
    expected = {f"sync_{i + 1}": cue + 0.35 for i, cue in enumerate(cues)}

    match = match_sounds(sounds, windows, expected, search_s=3600.0)
    assert match.searched
    assert match.shift_s == pytest.approx(lead, abs=0.3)
    assert set(match.found) == set(windows)
    for i, at in enumerate(events):
        assert match.found[f"sync_{i + 1}"] == pytest.approx(at, abs=0.005)


def test_two_events_never_share_one_sound():
    from studio.video import SharpSounds, match_sounds

    sounds = SharpSounds(np.array([10.0, 12.6]), np.array([1.0, 0.9]), np.array([True, True]))
    windows = {"sync_1": (8.0, 12.0), "sync_2": (9.5, 13.5)}
    found = match_sounds(sounds, windows, {"sync_1": 10.1, "sync_2": 10.4}).found
    assert found == {"sync_1": 10.0, "sync_2": 12.6}


def test_main_window_counts_steps_on_the_video(demo, monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QThreadPool
    from PySide6.QtWidgets import QApplication

    from studio.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.open_folder(demo)
    for _ in range(300):
        app.processEvents()
        if QThreadPool.globalInstance().waitForDone(50) and window.engine is not None \
                and len(window.engine.signals) == 4 and window.project.video_events:
            break
    app.processEvents()
    # The audio of the video placed the claps on its own.
    assert len(window.project.video_events) == 6
    assert window.engine.video_fit_source == "claps"

    window.project.steps = []
    activity = window.engine.session.activities[1]
    start_video = float(window.engine.display_to_video(activity.start_time.timestamp()))
    for k in range(10):
        window.video.seek(start_video + 2.0 + 0.5 * k)
        window.mark_step("L" if k % 2 == 0 else "R")
    assert len(window.project.steps) == 10
    assert window._step_counts[activity.activity_index] == (10, 10)
    window.video.seek(start_video + 2.0 + 0.5 * 9 + 0.1)
    window.remove_step()
    assert len(window.project.steps) == 9
    window.undo_step()
    assert len(window.project.steps) == 10
    assert window.steps_table.item(1, 1).text() == "10"
    assert "Pasos (10)" in window.tabs.tabText(1)

    written = {p.relative_to(tmp_path).as_posix() for p in window.engine.export(tmp_path)}
    assert {"steps.csv", "steps_per_activity.csv"} <= written
    window.project.steps = []
    window.project.video_events = {}
    window.close()
    (demo / "saves_project.json").unlink(missing_ok=True)
