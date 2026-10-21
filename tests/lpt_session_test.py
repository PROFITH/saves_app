"""SAVES on a session folder written by lab_protocol_timer, with simulated sensors."""

import pytest

from importers.manifest_importer import ManifestImporter
from studio.discovery import discover
from studio.engine import SyncEngine
from studio.project import SyncProject
from studio.signals import load_signal
from tests.lpt_session import SENSORS, build_lpt_session, true_offsets_ms


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return build_lpt_session(tmp_path_factory.mktemp("lpt") / "R4H_001" / "D0")


@pytest.fixture(scope="module")
def engine(folder):
    found = discover(folder)
    project = SyncProject(folder=str(folder), manifest_path=found.manifest_path, videos=found.videos)
    for entry in found.sensors:
        project.add_sensor(entry)
    session = ManifestImporter().load(project.manifest_path)
    engine = SyncEngine(project, session)
    for entry in project.sensors:
        engine.set_signal(entry.sensor_id, load_signal(entry.path, entry.sensor_format, folder))
    engine.refit()
    return engine


def test_manifest_carries_the_kind_and_cue_of_every_event(folder):
    session = ManifestImporter().load(folder / "session_manifest.json")
    kinds = [w.sync_kind for w in session.sync_windows]
    assert kinds.count("jump") == 12 and kinds.count("clap") == 6
    assert all(w.cue_time is not None for w in session.sync_windows)
    assert session.activities[0].met == 1.0


def test_files_match_the_sensors_of_the_manifest(folder):
    found = discover(folder)
    assert sorted(s.sensor_id for s in found.sensors) == sorted(SENSORS)
    by_id = {s.sensor_id: s for s in found.sensors}
    assert by_id["ax3_wnd"].location == "Muñeca no dominante"
    assert by_id["sens_thigh"].location == "Muslo"
    assert by_id["verity_arm"].location == "Brazo"


def test_clock_offsets_up_to_two_hours_are_found(engine):
    for sid, offset in true_offsets_ms().items():
        search = engine.offset_searches[sid]
        assert search.accepted, (sid, search.note)
        assert search.offset_ms == pytest.approx(offset, abs=600), sid


def test_every_offline_sensor_is_synchronized(engine):
    ref = engine.resolved_reference()
    assert ref in ("polar_h10", "verity_arm")
    for sid in true_offsets_ms():
        if sid in ("polar_h10", "verity_arm"):
            continue
        status = engine.status(sid)
        assert status.quality == "good", (sid, status)
        assert status.n_pairs >= 10, sid
        assert status.rmse_ms < 25, sid
        # offset_ms is reference minus sensor at the first shared event.
        first = engine.fits[sid]["reference_origin_seconds"]
        assert -status.offset_ms == pytest.approx(true_offsets_ms(first)[sid], abs=40), sid


def test_jumps_are_anchored_at_the_landing(engine):
    session = engine.session
    jump = next(w for w in session.sync_windows if w.sync_kind == "jump")
    times = {}
    for sid in ("ax3_thigh", "ax3_wd", "polar_h10"):
        anchor = engine.anchor(sid, jump.event_id)
        assert anchor is not None, sid
        assert anchor.selection_rule == "closest_to_cue"
        times[sid] = float(engine.sensor_to_display(sid, anchor.sensor_timestamp_unix_ms))
    assert max(times.values()) - min(times.values()) < 0.03


def test_chest_strap_misses_claps_but_the_arm_band_fills_the_reference(engine):
    windows = engine.session.sync_windows
    claps = [w for w in windows if w.sync_kind == "clap"]
    assert all(engine.anchor("polar_h10", w.event_id) is None for w in claps)
    assert any(engine.anchor("verity_arm", w.event_id) is not None for w in claps)
    # The reference holds every event caught by either Polar.
    caught = {
        w.event_id for w in windows
        if any(engine.anchor(sid, w.event_id) is not None for sid in ("polar_h10", "verity_arm"))
    }
    assert caught <= set(engine.reference_seconds())


def test_without_polar_an_offline_sensor_leads_on_the_cue_clock(folder):
    found = discover(folder)
    project = SyncProject(folder=str(folder), manifest_path=found.manifest_path)
    for entry in found.sensors:
        if entry.sensor_format != "polar_acc":
            project.add_sensor(entry)
    session = ManifestImporter().load(project.manifest_path)
    engine = SyncEngine(project, session)
    for entry in project.sensors:
        engine.set_signal(entry.sensor_id, load_signal(entry.path, entry.sensor_format, folder))
    engine.refit()

    ref = engine.resolved_reference()
    assert ref not in ("polar_h10", "verity_arm")
    assert engine.reference_host_fit is not None
    assert "avisos" in engine.reference_label()
    # Every sensor lands on the host clock within the reaction-time error,
    # and on each other within a few milliseconds.
    jump = next(w for w in session.sync_windows if w.sync_kind == "jump")
    times = []
    for sid in engine.active_sensors():
        anchor = engine.anchor(sid, jump.event_id)
        if anchor is None:
            continue
        host = float(engine.sensor_to_display(sid, anchor.sensor_timestamp_unix_ms))
        assert host == pytest.approx(jump.cue_time.timestamp() + 0.95, abs=0.35), sid
        times.append(host)
    assert len(times) >= 4
    assert max(times) - min(times) < 0.04


def test_export_labels_every_sample_on_the_host_clock(engine, tmp_path):
    import pandas as pd

    written = {p.relative_to(tmp_path).as_posix() for p in engine.export(tmp_path)}
    assert {"phases.csv", "activities.csv", "LEEME.txt", "sync_report.csv", "qc.json"} <= written
    assert "signals/ax3_wnd.csv.gz" in written

    phases = pd.read_csv(tmp_path / "phases.csv")
    # Four phases in each of the 9 activities; the protocol ends as the last
    # transition starts.
    assert len(phases) == 35
    assert set(phases["phase"]) == {"static_prep", "activity_main", "static_post", "transition_lap"}

    wrist = pd.read_csv(tmp_path / "signals" / "ax3_wnd.csv.gz")
    assert list(wrist.columns[:5]) == ["time_unix_ms", "time_native_unix_ms", "x", "y", "z"]
    # The device clock is two hours off; the export is on the host clock.
    assert (wrist["time_native_unix_ms"] - wrist["time_unix_ms"]).median() == pytest.approx(-7_199_400, abs=200)
    jog = wrist[(wrist["activity"] == "Trotar a ritmo propio") & (wrist["phase"] == "activity_main")]
    duration = (jog["time_unix_ms"].max() - jog["time_unix_ms"].min()) / 1000
    assert duration == pytest.approx(240, abs=0.5)
    session = engine.session
    first = wrist["time_unix_ms"].min() / 1000
    assert first == pytest.approx(session.start_time.timestamp() - 60, abs=1)

    activities = pd.read_csv(tmp_path / "activities.csv")
    assert len(activities) == 9
    assert {"met", "rpe", "completion", "end_reason"} <= set(activities.columns)


def test_steps_marked_on_the_video_are_counted_per_activity(engine, tmp_path):
    from studio.export import read_phases, step_table, steps_per_activity

    session = engine.session
    jog = next(a for a in session.activities if a.activity_name == "Trotar a ritmo propio")
    # A video that starts with the session, on the host clock.
    engine.video_fit = None
    origin = session.start_time.timestamp()
    from studio.engine import Affine

    engine.video_fit = Affine(a=origin, b=1.0)
    engine.video_duration_s = session.duration_seconds
    start = jog.start_time.timestamp() - origin
    engine.project.steps = []
    for k in range(60):
        engine.project.add_step(start + 1.0 + k * 0.37, "L" if k % 2 == 0 else "R")
    engine.project.add_step(start - 5.0)  # in the static phase before

    names = {a.activity_index: a.activity_name for a in session.activities}
    phases = read_phases(engine.project.folder, session)
    steps = step_table(engine, phases, names)
    assert len(steps) == 61
    per = steps_per_activity(engine, steps, phases, names).set_index("activity_index")
    assert per.loc[jog.activity_index, "steps_main"] == 60
    assert per.loc[jog.activity_index, "steps_all_phases"] == 61
    assert per.loc[jog.activity_index, "cadence_steps_min"] == pytest.approx(15.0, abs=0.1)
    assert per.loc[jog.activity_index, "video_coverage_pct"] == pytest.approx(100.0)
    engine.project.steps = []


def test_sit_to_stands_and_other_events_are_exported(engine, tmp_path):
    import pandas as pd

    from studio.engine import Affine

    session = engine.session
    chair = next(a for a in session.activities if a.activity_name == "Sentarse y levantarse 30 s")
    origin = session.start_time.timestamp()
    engine.video_fit = Affine(a=origin, b=1.0)
    engine.video_duration_s = session.duration_seconds
    start = chair.start_time.timestamp() - origin
    engine.project.technician = "AB"
    for k in range(14):
        engine.project.add_event("sit_to_stand", start + 1.0 + 2.0 * k)
    stairs = engine.project.add_event_type("Escalones", sides=True)
    engine.project.add_event(stairs.key, start + 0.5, "L")
    engine.project.set_task_state("sync", "finished")
    try:
        written = {p.relative_to(tmp_path).as_posix() for p in engine.export(tmp_path)}
        assert {"events_sit_to_stand.csv", "events_sit_to_stand_per_activity.csv",
                "events_escalones.csv", "tasks.csv"} <= written
        assert "steps.csv" not in written  # nothing counted

        stands = pd.read_csv(tmp_path / "events_sit_to_stand.csv")
        assert list(stands.columns[:4]) == ["event", "video_seconds", "time_unix_ms", "time_utc"]
        assert "side" not in stands.columns
        assert set(stands["activity"]) == {"Sentarse y levantarse 30 s"}
        per = pd.read_csv(tmp_path / "events_sit_to_stand_per_activity.csv").set_index("activity_index")
        assert per.loc[chair.activity_index, "sit_to_stand_main"] == 14
        assert "sit_to_stand_per_min" in per.columns
        assert list(pd.read_csv(tmp_path / "events_escalones.csv", keep_default_na=False)["side"]) == ["L"]

        activities = pd.read_csv(tmp_path / "activities.csv").set_index("activity_index")
        assert activities.loc[chair.activity_index, "sit_to_stand_video"] == 14
        tasks = pd.read_csv(tmp_path / "tasks.csv", keep_default_na=False)
        assert list(tasks["task"]) == ["sync", "steps", "sit_to_stand", "escalones"]
        assert list(tasks["state"]) == ["finished", "not_started", "not_started", "not_started"]
        assert tasks.loc[0, "technician"] == "AB" and tasks.loc[0, "finished_at"]
        readme = (tmp_path / "LEEME.txt").read_text(encoding="utf-8")
        assert readme.startswith("Exportado el ") and "por el técnico AB" in readme
        assert "Conteo de sentarse y levantarse (no iniciada)" in readme
    finally:
        engine.project.events = {}
        engine.project.remove_event_type(stairs.key)
        engine.project.tasks = {}
        engine.project.technician = ""
