"""What SAVES Studio sends to REDCap: the counts of the video in each lab_activity instance."""

import io
import json
import shutil
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

from importers.manifest_importer import ManifestImporter
from models import Participant
from studio import redcap
from studio.demo import VIDEO_LOG_ERROR, VIDEO_START, build_demo_session
from studio.discovery import discover
from studio.engine import SyncEngine
from studio.project import FINISHED, SIT_TO_STAND, STEPS, SYNC_TASK, SyncProject
from studio.signals import load_signal

FIXTURE = Path(__file__).parent / "fixtures" / "lpt_run4health"
PARTICIPANT = Participant("001", "da_0__visita_inici_arm_1")
DEMO = Participant("DEMO01", "da_0__visita_inici_arm_1")


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    folder = tmp_path_factory.mktemp("demo_redcap")
    build_demo_session(folder)
    return folder


@pytest.fixture()
def engine(demo):
    found = discover(demo)
    project = SyncProject(folder=str(demo), manifest_path=found.manifest_path, videos=found.videos, technician="T1")
    for entry in found.sensors:
        project.add_sensor(entry)
    engine = SyncEngine(project, ManifestImporter().load(project.manifest_path))
    for entry in project.sensors:
        engine.set_signal(entry.sensor_id, load_signal(entry.path, entry.sensor_format, demo))
    engine.refit()
    return engine


def _mark(engine, key, host_times, side=""):
    """Events at host times, on the video clock (the video is placed by its logged start)."""
    for t in host_times:
        engine.project.add_event(key, t - (VIDEO_START + VIDEO_LOG_ERROR), side)


def _session_folder(tmp_path, keep_instances=True):
    folder = tmp_path / "R4H_001" / "D0"
    shutil.copytree(FIXTURE, folder)
    if not keep_instances:
        manifest = json.loads((folder / "session_manifest.json").read_text(encoding="utf-8"))
        for activity in manifest["activities"]:
            activity.pop("redcap_instance", None)
        (folder / "session_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def _lpt_records(session, skipped_after=None):
    """lab_activity rows as lab_protocol_timer keeps them, with a skipped attempt after ``skipped_after``."""
    rows, instance = [], 0
    for activity in session.activities:
        instance += 1
        rows.append({
            "record_id": "001", "redcap_event_name": PARTICIPANT.redcap_event_name,
            "redcap_repeat_instrument": "lab_activity", "redcap_repeat_instance": str(instance),
            "la_name": activity.activity_name, "la_start": "2026-10-21 10:30:30",
        })
        if activity.activity_index == skipped_after:
            instance += 1
            rows.append({
                "record_id": "001", "redcap_event_name": PARTICIPANT.redcap_event_name,
                "redcap_repeat_instrument": "lab_activity", "redcap_repeat_instance": str(instance),
                "la_name": "Saltada", "la_completed": "3",
            })
    return [{"record_id": "001", "redcap_event_name": PARTICIPANT.redcap_event_name, "ws_session_id": "x"}, *rows]


def test_instances_come_from_the_manifest(tmp_path):
    folder = _session_folder(tmp_path)
    session = ManifestImporter().load(folder / "session_manifest.json")
    instances, source = redcap.attempt_instances(session, folder, PARTICIPANT)
    assert source == "session_manifest.json"
    assert instances == {(i, 1): i for i in range(1, 10)}


def test_older_sessions_take_the_instances_from_the_records_lpt_kept(tmp_path):
    folder = _session_folder(tmp_path, keep_instances=False)
    session = ManifestImporter().load(folder / "session_manifest.json")
    assert redcap.attempt_instances(session, folder, PARTICIPANT) == ({}, "")

    # A skipped attempt takes an instance but has no time span.
    (folder / "redcap_records.json").write_text(json.dumps(_lpt_records(session, skipped_after=4)), encoding="utf-8")
    instances, source = redcap.attempt_instances(session, folder, PARTICIPANT)
    assert source == "redcap_records.json"
    assert instances[(4, 1)] == 4 and instances[(5, 1)] == 6 and instances[(9, 1)] == 10

    # What was sent wins over what was pending, and names that do not
    # agree are never assumed.
    sent = _lpt_records(session)
    (folder / "redcap_records_enviados_20261021_120000.json").write_text(json.dumps(sent), encoding="utf-8")
    instances, source = redcap.attempt_instances(session, folder, PARTICIPANT)
    assert source == "redcap_records_enviados_20261021_120000.json" and instances[(5, 1)] == 5
    sent[3]["la_name"] = "Otra actividad"
    (folder / "redcap_records_enviados_20261021_120000.json").write_text(json.dumps(sent), encoding="utf-8")
    (folder / "redcap_records.json").unlink()
    assert redcap.attempt_instances(session, folder, PARTICIPANT) == ({}, "")


def test_only_finished_counts_are_prepared(engine):
    project = engine.project
    with pytest.raises(redcap.RedcapError, match="ningún conteo finalizado"):
        redcap.count_records(engine, DEMO)
    project.set_task_state(STEPS, FINISHED)
    with pytest.raises(redcap.RedcapError, match="sincronización"):
        redcap.count_records(engine, DEMO)
    project.set_task_state(SYNC_TASK, FINISHED)

    # 12 steps while walking (activity 2, 100-170 s) and 3 stands in the
    # stairs (activity 4, 280-340 s), none elsewhere.
    start = engine.session.start_time.timestamp()
    _mark(engine, STEPS, [start + 110 + 0.5 * k for k in range(12)], "L")
    _mark(engine, SIT_TO_STAND, [start + 290, start + 300, start + 310])
    prepared = redcap.count_records(engine, DEMO)
    assert prepared.kinds == [STEPS] and prepared.sent_counts == {STEPS: 4}
    assert prepared.session_id == "demo_session" and prepared.instances_from == "session_manifest.json"
    steps = {r["redcap_repeat_instance"]: r["la_steps_video"] for r in prepared.records}
    assert steps == {"1": "0", "2": "12", "3": "0", "4": "0"}
    assert all(r["la_steps_video_by"] == "T1" and "la_reps_video" not in r for r in prepared.records)
    assert prepared.records[0] == {
        "record_id": "DEMO01", "redcap_event_name": "da_0__visita_inici_arm_1",
        "redcap_repeat_instrument": "lab_activity", "redcap_repeat_instance": "1",
        "la_steps_video": "0", "la_steps_video_by": "T1",
    }

    # Sit-to-stands only where some were marked.
    project.set_task_state(SIT_TO_STAND, FINISHED)
    prepared = redcap.count_records(engine, DEMO)
    assert prepared.kinds == [STEPS, SIT_TO_STAND] and prepared.sent_counts[SIT_TO_STAND] == 1
    four = next(r for r in prepared.records if r["redcap_repeat_instance"] == "4")
    assert four["la_reps_video"] == "3" and four["la_reps_video_by"] == "T1"
    assert prepared.expected_names == {1: "Sentado en reposo", 2: "Caminar 4,8 km/h", 3: "Correr 8 km/h",
                                       4: "Subir escaleras"}

    # An activity the video does not cover gets no count.
    engine.video_duration_s = 200.0
    prepared = redcap.count_records(engine, DEMO)
    assert [r["redcap_repeat_instance"] for r in prepared.records] == ["1", "2"]
    assert any("«Correr 8 km/h»" in line and "el vídeo cubre" in line for line in prepared.left_out)


def test_the_import_file_has_every_column(tmp_path):
    records = [
        {"record_id": "001", "redcap_event_name": "e", "redcap_repeat_instrument": "lab_activity",
         "redcap_repeat_instance": "2", "la_steps_video": "12", "la_steps_video_by": "T1"},
    ]
    path = redcap.write_import_csv(records, tmp_path / "out" / redcap.CSV_NAME)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(frame.columns) == redcap.CSV_COLUMNS
    assert frame.loc[0, "la_steps_video"] == "12" and frame.loc[0, "la_reps_video"] == ""


def _prepared(**counts):
    prepared = redcap.CountRecords(participant=PARTICIPANT, technician="CD", kinds=[STEPS], session_id="s1",
                                   expected_names={1: "Barrer", 2: "Trotar"})
    prepared.records = [
        {"record_id": "001", "redcap_event_name": PARTICIPANT.redcap_event_name,
         "redcap_repeat_instrument": "lab_activity", "redcap_repeat_instance": str(i),
         "la_steps_video": str(n), "la_steps_video_by": "CD"}
        for i, n in counts.items()
    ]
    return prepared


def test_counts_go_only_to_this_sessions_attempts():
    prepared = _prepared(**{"1": 40, "2": 300})
    prepared.records = [{**r, "redcap_repeat_instance": r["redcap_repeat_instance"]} for r in prepared.records]
    rows = {1: {"la_session_id": "s1", "la_name": "Barrer"}, 2: {"la_session_id": "s1", "la_name": "Trotar"}}
    assert redcap.check_instances(prepared, rows) == []
    assert "todavía no tiene las actividades" in redcap.check_instances(prepared, {1: rows[1]})[0]
    other = {**rows, 2: {"la_session_id": "s0", "la_name": "Trotar"}}
    assert redcap.check_instances(prepared, other) == ["La instancia 2 es de otra sesión (s0), no de s1."]
    renamed = {**rows, 1: {"la_session_id": "s1", "la_name": "Correr"}}
    assert redcap.check_instances(prepared, renamed) == ["La instancia 1 es «Correr», no «Barrer»."]

    # Another technician's count is replaced only if the technician says so.
    rows[1].update(la_steps_video="38", la_steps_video_by="AB")
    rows[2].update(la_steps_video="301", la_steps_video_by="CD")
    assert redcap.counts_of_others(prepared, rows) == {STEPS: {"AB"}}


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_the_api_reads_and_sends_with_the_token(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        body = dict(pd.Series(dict(x.split("=", 1) for x in request.data.decode().split("&"))))
        calls.append((request.full_url, body))
        if body["content"] == "record" and "data" not in body:
            rows = [
                {"record_id": "001", "redcap_event_name": "e", "redcap_repeat_instrument": "", "redcap_repeat_instance": ""},
                {"record_id": "001", "redcap_event_name": "e", "redcap_repeat_instrument": "lab_activity",
                 "redcap_repeat_instance": "2", "la_session_id": "s1", "la_name": "Trotar",
                 "la_steps_video": "", "la_steps_video_by": ""},
            ]
            return FakeResponse(json.dumps(rows).encode())
        return FakeResponse(b'{"count": 1}')

    monkeypatch.setattr(redcap.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("REDCAP_API_URL", "https://redcap.example/api/")
    rows = redcap.current_rows("T" * 32, PARTICIPANT)
    assert list(rows) == [2] and rows[2]["la_name"] == "Trotar"
    url, body = calls[-1]
    assert url == "https://redcap.example/api/"
    assert body["records%5B0%5D"] == "001" and "la_steps_video_by" in body.values()
    assert redcap.import_records("T" * 32, _prepared(**{"2": 300}).records) == 1
    assert calls[-1][1]["overwriteBehavior"] == "normal"


def test_refusals_of_redcap_say_what_to_do(monkeypatch):
    def refuse(code, message):
        def fake_urlopen(request, timeout):
            raise urllib.error.HTTPError(request.full_url, code, "", {}, io.BytesIO(json.dumps({"error": message}).encode()))
        monkeypatch.setattr(redcap.urllib.request, "urlopen", fake_urlopen)

    refuse(403, "You do not have permissions to use the API")
    with pytest.raises(redcap.TokenRejected):
        redcap.current_rows("T" * 32, PARTICIPANT)
    refuse(400, 'The following values in the parameter "fields" are not valid: la_steps_video')
    with pytest.raises(redcap.RedcapError, match="diccionario v6"):
        redcap.current_rows("T" * 32, PARTICIPANT)


def test_a_finished_count_waits_to_be_sent_until_it_is(tmp_path):
    project = SyncProject(folder=str(tmp_path), technician="T1")
    assert not redcap.needs_sending(project, STEPS)
    project.add_event(STEPS, 1.0)
    project.set_task_state(STEPS, FINISHED)
    assert redcap.needs_sending(project, STEPS)
    redcap.mark_sent(project, [STEPS])
    assert not redcap.needs_sending(project, STEPS)
    assert project.tasks[STEPS]["redcap_sent_by"] == "T1"
    # Finished again after a change: REDCap has the old count.
    project.tasks[STEPS]["redcap_sent_at"] = "2026-10-21T10:00:00+02:00"
    project.tasks[STEPS]["finished_at"] = "2026-10-21T11:00:00+02:00"
    assert redcap.needs_sending(project, STEPS)
    # The synchronization never goes to REDCap.
    project.set_task_state(SYNC_TASK, FINISHED)
    assert not redcap.needs_sending(project, SYNC_TASK)
