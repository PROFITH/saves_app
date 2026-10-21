"""The project file of SAVES Studio: technicians, tasks and events counted on the video."""

import json

import pytest

from studio.project import (
    FINISHED,
    NOT_STARTED,
    STARTED,
    SyncProject,
    event_type_key,
    normalize_technician,
)


def test_technician_codes_and_event_keys():
    assert normalize_technician(" jhm ") == "JHM"
    assert normalize_technician("José") == "JOSE"
    assert normalize_technician("a b") == ""
    assert normalize_technician("x" * 13) == ""
    assert event_type_key("Escalones") == "escalones"
    assert event_type_key("Subir escalón", {"subir_escalon"}) == "subir_escalon_2"
    assert event_type_key("¿?") == "eventos"


def test_each_technician_has_a_file(tmp_path):
    first = SyncProject(folder=str(tmp_path), technician="AB")
    first.add_event("steps", 1.0, "L")
    assert first.save().name == "saves_project_AB.json"
    SyncProject(folder=str(tmp_path), technician="CD").save()
    (tmp_path / "saves_project.json").write_text("{}", encoding="utf-8")
    (tmp_path / "saves_project_ab.json.tmp").write_text("{}", encoding="utf-8")
    assert SyncProject.technicians_in(tmp_path) == ["AB", "CD"]
    assert SyncProject.exists_in(tmp_path, "AB") and not SyncProject.exists_in(tmp_path, "EF")
    loaded = SyncProject.load(tmp_path, "AB")
    assert loaded.technician == "AB"
    assert loaded.events["steps"] == [{"video_s": 1.0, "side": "L"}]
    assert not SyncProject.load(tmp_path, "CD").events.get("steps")


def test_files_of_the_first_version_keep_their_steps(tmp_path):
    (tmp_path / "saves_project.json").write_text(json.dumps({
        "folder": str(tmp_path),
        "steps": [{"video_s": 2.0, "foot": "R"}, {"video_s": 1.0, "foot": ""}],
        "event_types": [{"key": "steps", "name": "Pasos", "sides": True, "hint": "texto antiguo"}],
    }), encoding="utf-8")
    project = SyncProject.load(tmp_path)
    assert project.steps == [{"video_s": 1.0, "side": ""}, {"video_s": 2.0, "side": "R"}]
    # The texts of the built-in kinds follow the app.
    assert project.event_type("steps").hint == "cada apoyo del pie"
    assert [k.key for k in project.event_types] == ["steps"]


def test_events_of_any_kind(tmp_path):
    project = SyncProject(folder=str(tmp_path))
    assert [k.key for k in project.event_types] == ["steps", "sit_to_stand"]
    assert not project.event_type("sit_to_stand").sides
    project.add_event("sit_to_stand", 3.0)
    project.add_event("sit_to_stand", 1.0)
    project.add_event("sit_to_stand", 1.0004)  # same frame: replaces
    assert project.event_times("sit_to_stand") == [1.0004, 3.0]
    assert project.remove_event_near("sit_to_stand", 2.9)["video_s"] == 3.0
    assert project.remove_event_near("sit_to_stand", 2.0) is None
    kind = project.add_event_type("Escalones", sides=False)
    assert project.add_event_type("escalones") is kind
    assert project.task_keys() == ["sync", "steps", "sit_to_stand", "escalones"]
    assert project.task_name("escalones") == "Conteo de escalones"
    assert project.add_event_type("TUG").lower_name == "TUG"
    with pytest.raises(ValueError):
        project.remove_event_type("sit_to_stand")
    project.set_task_state("escalones", STARTED)
    project.remove_event_type("escalones")
    assert "escalones" not in project.task_keys() and "escalones" not in project.tasks


def test_only_the_technician_finishes_a_task(tmp_path):
    project = SyncProject(folder=str(tmp_path), technician="AB")
    assert project.pending_tasks() == ["sync", "steps", "sit_to_stand"]
    assert project.work_on("sync") == "started"
    assert project.work_on("sync") == ""
    assert project.task_state("sync") == STARTED
    project.set_task_state("sync", FINISHED)
    task = project.tasks["sync"]
    assert task["by"] == "AB" and task["started_at"] and task["finished_at"]
    assert project.pending_tasks() == ["steps", "sit_to_stand"]
    # A change after finishing sends the task back to started.
    assert project.work_on("sync") == "reopened"
    assert project.task_state("sync") == STARTED and "finished_at" not in project.tasks["sync"]
    project.set_task_state("sync", NOT_STARTED)
    assert "started_at" not in project.tasks["sync"]
    with pytest.raises(ValueError):
        project.set_task_state("sync", "done")
    project.add_event("steps", 1.0)
    project.work_on("steps")
    assert project.progress_summary() == (
        "Sincronización: no iniciada · Pasos: iniciada (1) · Sentarse y levantarse: no iniciada")


def test_a_second_technician_can_start_from_the_first_ones_synchronization(tmp_path):
    first = SyncProject(folder=str(tmp_path), technician="AB", reference="h10")
    first.video_events = {"sync_1": 4.5}
    first.set_manual_anchor("sens", "sync_2", 123.0)
    first.set_excluded("sens", "sync_3", True)
    first.add_event("steps", 1.0)
    first.set_task_state("sync", FINISHED)
    second = SyncProject(folder=str(tmp_path), technician="CD")
    second.copy_sync_from(first)
    assert second.video_events == {"sync_1": 4.5} and second.reference == "h10"
    assert second.manual_anchors == {"sens": {"sync_2": 123.0}}
    assert second.is_excluded("sens", "sync_3")
    assert not second.events.get("steps")
    assert second.task_state("sync") == FINISHED
    assert second.tasks["sync"]["by"] == "AB" and second.tasks["sync"]["copied_from"] == "AB"
    # Copies, not the same objects.
    second.video_events["sync_2"] = 9.0
    assert "sync_2" not in first.video_events


def test_hidden_plots_follow_the_sensors(tmp_path):
    project = SyncProject(folder=str(tmp_path))
    project.set_plot_hidden("sens", True)
    project.set_plot_hidden("sens", True)
    assert project.hidden_plots == ["sens"]
    project.remove_sensor("sens")
    assert project.hidden_plots == []
