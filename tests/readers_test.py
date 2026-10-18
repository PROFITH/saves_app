from datetime import datetime

import numpy as np
import pytest

from readers import infer_format, read_cwa, read_matrix, read_polar_acc, read_sens, read_sensor
from tests.synthetic import (
    write_cwa,
    write_matrix_bin,
    write_polar_acc_csv,
    write_sens_csv,
)

# 2026-10-27 09:30:00 in Madrid (CET, UTC+1 after the October DST change).
START_LOCAL = datetime(2026, 10, 27, 9, 30, 0)
START_UTC_S = 1_793_089_800.0


def _accel(n, seed=1):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-1.9, 1.9, size=(n, 3))
    xyz[::50] *= 3.5  # large values exercise the packed exponent
    return xyz


@pytest.mark.parametrize("fractional", [False, True])
def test_cwa_packed_values_and_utc_timestamps(tmp_path, fractional):
    path = tmp_path / "ax3.cwa"
    values = _accel(120 * 30 + 37)  # partial last sector
    local = write_cwa(path, START_LOCAL, values, fractional=fractional)

    recording = read_cwa(path, timezone="Europe/Madrid")

    assert recording.metadata["device_type"] == "AX3"
    assert recording.metadata["utc_offset_minutes"] == 60
    assert recording.metadata["annotation"]["_sc"] == "R4H001"
    assert recording.sampling_rate_hz == 100
    assert len(recording.data) == values.shape[0]

    # 10-bit packing: error below one quantization step of the exponent.
    step = (2 ** np.ceil(np.log2(np.maximum(np.abs(values).max(axis=1) * 256 / 511, 1)))) / 256
    error = np.abs(recording.data[["x", "y", "z"]].to_numpy() - values)
    assert np.all(error <= step[:, None])

    expected_ms = (local - 3600) * 1000.0
    tolerance_ms = 1.0 if fractional else 5.0  # integer offsets round to half a sample
    assert np.max(np.abs(recording.data["timestamp_unix_ms"].to_numpy() - expected_ms)) <= tolerance_ms
    assert recording.data["timestamp_unix_ms"].iloc[0] == pytest.approx(START_UTC_S * 1000, abs=tolerance_ms)


def test_cwa_follows_the_real_sampling_rate(tmp_path):
    # Device nominally at 100 Hz but actually sampling 0.1% faster.
    path = tmp_path / "ax3.cwa"
    values = _accel(120 * 300)
    local = write_cwa(path, START_LOCAL, values, true_frequency=100.1, fractional=True)

    recording = read_cwa(path, utc_offset_minutes=60)

    expected_ms = (local - 3600) * 1000.0
    assert np.max(np.abs(recording.data["timestamp_unix_ms"].to_numpy() - expected_ms)) <= 1.0


def test_cwa_skips_corrupt_sectors(tmp_path):
    path = tmp_path / "ax3.cwa"
    values = _accel(120 * 10)
    local = write_cwa(path, START_LOCAL, values, fractional=True, corrupt_sectors=(4,))

    recording = read_cwa(path, utc_offset_minutes=60)

    assert recording.metadata["n_invalid_sectors"] == 1
    assert len(recording.data) == 120 * 9
    kept = np.r_[0:480, 600:1200]
    expected_ms = (local[kept] - 3600) * 1000.0
    assert np.max(np.abs(recording.data["timestamp_unix_ms"].to_numpy() - expected_ms)) <= 1.0


def test_cwa_ax6_with_gyroscope(tmp_path):
    path = tmp_path / "ax6.cwa"
    n = 40 * 20
    rng = np.random.default_rng(3)
    gyro = rng.uniform(-500, 500, size=(n, 3))
    accel = _accel(n)
    write_cwa(path, START_LOCAL, np.hstack([gyro, accel]), packed=False, channels=6, fractional=True)

    recording = read_cwa(path, utc_offset_minutes=60)

    assert recording.metadata["device_type"] == "AX6"
    np.testing.assert_allclose(recording.data[["x", "y", "z"]], accel, atol=1 / 2048)
    np.testing.assert_allclose(recording.data[["gyro_x", "gyro_y", "gyro_z"]], gyro, atol=2000 / 32768)


def test_cwa_matches_open_movement_reader(tmp_path):
    om = pytest.importorskip("openmovement.load.cwa_load")
    path = tmp_path / "ax3.cwa"
    values = _accel(120 * 20)
    # Non-fractional timestamps: the fractional path of openmovement 0.0.1
    # overflows with numpy 2.
    write_cwa(path, START_LOCAL, values, fractional=False)

    with om.CwaData(str(path), include_gyro=False) as reference:
        expected = reference.get_sample_values()

    recording = read_cwa(path, utc_offset_minutes=0)

    np.testing.assert_allclose(recording.data[["x", "y", "z"]], expected[:, 1:4])
    # openmovement extrapolates nothing at the edges; compare the interior.
    np.testing.assert_allclose(
        recording.data["timestamp_unix_ms"].to_numpy()[120:-120] / 1000.0,
        expected[120:-120, 0],
        atol=1e-3,
    )


def test_matrix_reader_keeps_timestamps_and_skips_bad_crc(tmp_path):
    path = tmp_path / "matrix.bin"
    xyz = _accel(1000) / 2
    write_matrix_bin(path, int(START_UTC_S), xyz, corrupt_packages=(3,))

    recording = read_matrix(path)

    assert recording.metadata["n_crc_errors"] == 1
    assert len(recording.data) == 900
    assert recording.sampling_rate_hz == pytest.approx(100, rel=0.02)
    assert recording.data["timestamp_unix_ms"].iloc[0] == START_UTC_S * 1000
    assert recording.data["timestamp_unix_ms"].iloc[300] == (START_UTC_S + 4) * 1000
    np.testing.assert_allclose(recording.data["x"].to_numpy()[:100], xyz[:100, 0], atol=8 / 0x7FFF)


def test_csv_readers_and_format_inference(tmp_path):
    ts = START_UTC_S * 1000 + np.arange(500) * 10.0
    xyz = _accel(500)

    sens = tmp_path / "sens.csv"
    polar = tmp_path / "polar_acc.csv"
    write_sens_csv(sens, ts, xyz)
    write_polar_acc_csv(polar, ts, xyz)

    assert infer_format(sens) == "sens"
    assert infer_format(polar) == "polar_acc"
    assert infer_format(tmp_path / "a.cwa") == "cwa"
    assert infer_format(tmp_path / "a.bin") == "matrix"

    for recording in (read_sens(sens), read_polar_acc(polar), read_sensor(polar)):
        assert recording.sampling_rate_hz == pytest.approx(100)
        np.testing.assert_allclose(recording.data["timestamp_unix_ms"], ts, atol=0.001)
        np.testing.assert_allclose(recording.data[["x", "y", "z"]], xyz, atol=1e-6)
