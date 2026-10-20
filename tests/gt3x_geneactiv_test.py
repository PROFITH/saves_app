import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from readers import (
    FORMAT_BY_SENSOR_TYPE,
    infer_format,
    read_geneactiv,
    read_gt3x,
    read_sensor,
)
from tests.synthetic import write_geneactiv_bin, write_gt3x, write_matrix_bin

# 2026-10-27 09:30:00 local; the files declare UTC+1.
START_LOCAL = datetime(2026, 10, 27, 9, 30, 0)
START_UTC_S = 1_793_089_800.0

# Real files are not committed (licences of their source repositories). Point
# SAVES_SAMPLE_DIR at a folder with TAS1H30182785_2019-09-17.gt3x (read.gt3x
# inst/extdata) and GENEActiv_testfile.bin (GGIRread inst/testfiles).
SAMPLES = Path(os.environ.get("SAVES_SAMPLE_DIR", "/nonexistent"))


def _accel(n, seed=3):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.9, 1.9, size=(n, 3))


@pytest.mark.parametrize("record_type", [0x1A, 0x00])
def test_gt3x_log_bin_samples_idle_sleep_and_utc(tmp_path, record_type):
    rate = 30
    xyz = _accel(rate * 10)
    path = tmp_path / "wgt3x.gt3x"
    write_gt3x(path, START_LOCAL, xyz, rate=rate, idle_seconds=(4, 5, 6), record_type=record_type, trailing_idle_s=2)

    recording = read_gt3x(path)
    data = recording.data

    assert recording.sensor_type == "actigraph"
    assert recording.metadata["serial"] == "TAS1H00000001"
    assert recording.metadata["utc_offset_minutes"] == 60
    assert recording.sampling_rate_hz == rate
    # 10 recorded seconds + 3 idle + 2 idle at the end.
    assert len(data) == rate * 15
    np.testing.assert_allclose(data["timestamp_unix_ms"].iloc[0], START_UTC_S * 1000)
    np.testing.assert_allclose(np.diff(data["timestamp_unix_ms"]), 1000 / rate, atol=1e-3)

    imputed = data["imputed"].to_numpy().astype(bool)
    assert imputed[4 * rate:7 * rate].all() and imputed[-2 * rate:].all()
    assert not imputed[:4 * rate].any() and not imputed[7 * rate:13 * rate].any()

    recorded = data.loc[~imputed, ["x", "y", "z"]].to_numpy()
    step = 1 / 256
    np.testing.assert_allclose(recorded, xyz, atol=step / 2 + 1e-9)
    # Idle sleep repeats the last recorded sample.
    np.testing.assert_allclose(data.loc[4 * rate, ["x", "y", "z"]].to_numpy(), recorded[4 * rate - 1])
    assert [e["event"] for e in recording.metadata["idle_sleep_events"]] == ["idle_sleep_start", "idle_sleep_end"]


def test_gt3x_old_activity_bin_layout(tmp_path):
    xyz = _accel(40 * 7, seed=5)
    path = tmp_path / "old.gt3x"
    write_gt3x(path, START_LOCAL, xyz, rate=40, scale=341.0, layout="activity.bin", timezone_offset="-05:00:00")

    recording = read_gt3x(path, fill_to_last_sample=False)

    assert recording.metadata["layout"] == "activity.bin"
    assert recording.metadata["utc_offset_minutes"] == -300
    assert len(recording.data) == xyz.shape[0]
    np.testing.assert_allclose(recording.data[["x", "y", "z"]], xyz, atol=0.5 / 341 + 1e-9)
    assert recording.data["timestamp_unix_ms"].iloc[0] == (START_UTC_S + 3600 + 5 * 3600) * 1000


def test_geneactiv_calibration_light_button_and_page_times(tmp_path):
    rate = 100.0
    xyz = _accel(300 * 6, seed=7) * 0.9
    path = tmp_path / "geneactiv.bin"
    write_geneactiv_bin(path, START_LOCAL, xyz, rate=rate, truncate_last_page=True)

    recording = read_geneactiv(path)
    data = recording.data

    assert recording.sensor_type == "geneactiv"
    assert recording.metadata["serial"] == "045678"
    assert recording.metadata["utc_offset_minutes"] == 60
    assert recording.metadata["device_location"] == "left wrist"
    # The truncated last page is dropped.
    assert recording.metadata["n_pages"] == 5
    assert len(data) == 1500
    np.testing.assert_allclose(data[["x", "y", "z"]], xyz[:1500], atol=100 / 2 / 25538 + 1e-9)
    np.testing.assert_allclose(data["light"].iloc[:4], np.arange(4) * 800 / 300)
    assert data["button"].iloc[0] == 0 and data["button"].iloc[300] == 1
    assert data["temperature"].iloc[0] == 24.5
    np.testing.assert_allclose(data["timestamp_unix_ms"].iloc[0], START_UTC_S * 1000)
    np.testing.assert_allclose(np.diff(data["timestamp_unix_ms"]), 10.0, atol=1e-3)


def test_bin_inference_tells_geneactiv_from_matrix(tmp_path):
    gene = tmp_path / "a.bin"
    matrix = tmp_path / "b.bin"
    write_geneactiv_bin(gene, START_LOCAL, _accel(600))
    write_matrix_bin(matrix, int(START_UTC_S), _accel(200) / 2)

    assert infer_format(gene) == "geneactiv"
    assert infer_format(matrix) == "matrix"
    assert infer_format(tmp_path / "c.gt3x") == "gt3x"
    assert read_sensor(gene).sensor_type == "geneactiv"
    assert FORMAT_BY_SENSOR_TYPE["actigraph_gt3x"] == "gt3x"
    assert FORMAT_BY_SENSOR_TYPE["geneactiv"] == "geneactiv"


@pytest.mark.skipif(not (SAMPLES / "TAS1H30182785_2019-09-17.gt3x").exists(), reason="real sample not available")
def test_real_gt3x_sample():
    recording = read_gt3x(SAMPLES / "TAS1H30182785_2019-09-17.gt3x")
    data = recording.data
    assert recording.metadata["device_type"] == "Link"
    assert recording.metadata["accel_scale_counts_per_g"] == 256.0
    assert recording.metadata["utc_offset_minutes"] == -240
    # Start 2019-09-17 18:40:00 local (UTC-4) to the last sample time.
    assert data["timestamp_unix_ms"].iloc[0] == 1_568_760_000_000
    assert len(data) == 240_500
    assert len(recording.metadata["idle_sleep_events"]) == 10
    np.testing.assert_allclose(data.loc[0, ["x", "y", "z"]].to_numpy(), [0.0, 0.0078125, 0.99609375])
    assert data.loc[data["imputed"] == 0, "Vector Magnitude"].median() == pytest.approx(1.0, abs=0.05)


@pytest.mark.skipif(not (SAMPLES / "GENEActiv_testfile.bin").exists(), reason="real sample not available")
def test_real_geneactiv_sample():
    recording = read_geneactiv(SAMPLES / "GENEActiv_testfile.bin")
    data = recording.data
    assert recording.metadata["serial"] == "012967"
    assert recording.sampling_rate_hz == pytest.approx(85.7)
    assert recording.metadata["utc_offset_minutes"] == 60
    assert recording.metadata["n_pages"] == 16
    # 2013-05-30 10:12:54.500 local (UTC+1).
    assert data["timestamp_unix_ms"].iloc[0] == pytest.approx(1_369_905_174_500)
    np.testing.assert_allclose(
        data.loc[0, ["x", "y", "z"]].to_numpy(),
        [(196 * 100 - 439) / 25875, (-3 * 100 + 662) / 25734, (-195 * 100 + 3056) / 25538],
    )
    assert data["Vector Magnitude"].median() == pytest.approx(1.0, abs=0.05)


def test_studio_discovery_uses_manifest_placement_and_serials(tmp_path):
    import json

    from studio.discovery import discover

    write_gt3x(tmp_path / "P01_actigraph.gt3x", START_LOCAL, _accel(300), rate=30)
    write_geneactiv_bin(tmp_path / "P01_gene.bin", START_LOCAL, _accel(600))
    write_matrix_bin(tmp_path / "A01_geneactiv_wnd_extra.bin", int(START_UTC_S), _accel(200) / 2)
    (tmp_path / "session_manifest.json").write_text(json.dumps({
        "schema_version": "1.1",
        "sensors": {
            "actigraph_gt3x_hip_r": {
                "present": True, "timestamp_domain": "device_clock_local", "type": "actigraph_gt3x",
                "placement": "hip", "side": "right", "serial": "TAS1H00000001",
            },
            "geneactiv_wnd": {
                "present": True, "timestamp_domain": "device_clock_local", "type": "geneactiv",
                "placement": "wrist", "dominance": "non_dominant", "side": "left",
                "placement_label": "Muñeca no dominante (izquierda)", "serial": "045678",
            },
            "polar_h10": {"present": True, "timestamp_domain": "native_plus_host_utc", "placement": "chest"},
        },
    }), encoding="utf-8")

    found = discover(tmp_path)
    by_format = {s.sensor_format: s for s in found.sensors}

    assert set(by_format) == {"gt3x", "geneactiv", "matrix"}
    assert by_format["gt3x"].sensor_id == "actigraph_gt3x_hip_r"
    assert by_format["gt3x"].location == "Cadera (derecha)"
    assert by_format["gt3x"].serial == "TAS1H00000001"
    assert by_format["geneactiv"].sensor_id == "geneactiv_wnd"
    assert by_format["geneactiv"].location == "Muñeca no dominante (izquierda)"
    # The Matrix file names a sensor ID that the GENEActiv already took by serial.
    assert by_format["matrix"].sensor_id == "matrix"
