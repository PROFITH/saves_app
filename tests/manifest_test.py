import json
from pathlib import Path

import pytest

from importers.manifest_importer import ManifestImporter
from tests.synthetic import write_manifest

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "docs" / "examples" / "session_manifest.v1.1.json"
SCHEMA = REPO / "schemas" / "session_manifest.v1.1.schema.json"

START = 1_793_089_800.0  # 2026-10-27T08:30:00Z


def test_example_manifest_matches_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(json.loads(EXAMPLE.read_text(encoding="utf-8")), schema)


def test_example_manifest_loads_with_activity_metadata():
    session = ManifestImporter().load(EXAMPLE)

    assert session.schema_version == "1.1"
    assert len(session.sync_windows) == 2
    activity = session.activities[0]
    assert activity.compendium_code == "17200"
    assert activity.activity_name.startswith("Caminar")
    assert activity.planned_duration_seconds == 120


@pytest.mark.parametrize("schema_version", ["1.0", "1.1"])
def test_sync_window_keys_of_both_versions_are_accepted(tmp_path, schema_version):
    # lab_protocol_timer <= Sept 2026 wrote window_start / window_end.
    path = tmp_path / "session_manifest.json"
    write_manifest(
        path,
        START,
        START + 240,
        [("sync_1", "static_prep", START + 20), ("sync_2", "transition_lap", START + 200)],
        schema_version=schema_version,
    )

    session = ManifestImporter().load(path)

    assert [w.event_id for w in session.sync_windows] == ["sync_1", "sync_2"]
    assert session.sync_windows[0].duration_seconds == pytest.approx(10.0)
    assert session.sync_windows[0].window_start.timestamp() == pytest.approx(START + 20)


def test_unsupported_schema_version_is_rejected(tmp_path):
    path = tmp_path / "session_manifest.json"
    write_manifest(path, START, START + 60, [], schema_version="1.1")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "2.0"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported manifest schema version"):
        ManifestImporter().load(path)
