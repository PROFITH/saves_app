from datetime import datetime
from itertools import pairwise

import numpy as np
import pytest

from readers import (
    infer_format,
    polar_file_parts,
    read_cwa,
    read_matrix,
    read_polar_acc,
    read_polar_companion,
    read_polar_ecg,
    read_polar_gyro,
    read_polar_hr,
    read_polar_mag,
    read_polar_ppg,
    read_polar_ppi,
    read_sens,
    read_sensor,
)
from readers.cwa import temperature_celsius
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


def test_cwa_keeps_the_light_and_temperature_of_every_sector(tmp_path):
    path = tmp_path / "ax3.cwa"
    values = _accel(120 * 3 + 30)  # four sectors, the last one partial
    light = [300, 310, 1023, 5]
    temperature_raw = [262, 0, 300, 171]  # 0: the sector has no reading
    write_cwa(path, START_LOCAL, values, fractional=True, light=light, temperature_raw=temperature_raw)

    recording = read_cwa(path, utc_offset_minutes=60)
    data = recording.data

    assert list(data.columns) == ["timestamp_unix_ms", "x", "y", "z", "light_raw", "temperature_c", "Vector Magnitude"]
    edges = np.cumsum([0, 120, 120, 120, 30])
    for k, (lo, hi) in enumerate(pairwise(edges)):
        # One reading per sector, repeated on every sample of the sector.
        assert (data["light_raw"].iloc[lo:hi] == light[k]).all()
        celsius = data["temperature_c"].iloc[lo:hi]
        if temperature_raw[k] == 0:
            assert celsius.isna().all()
        else:
            np.testing.assert_allclose(celsius, temperature_raw[k] * 75 / 256 - 50)
    # Axivity's (counts - 171) / 3.413 is the same line, within a tenth of a degree.
    np.testing.assert_allclose(
        temperature_celsius(np.array([171.0, 300.0, 400.0])), [0.0, 129 / 3.413, 229 / 3.413], atol=0.15
    )


def test_matrix_keeps_gyroscope_temperature_and_heart_rate(tmp_path):
    path = tmp_path / "matrix.bin"
    n = 300
    xyz = _accel(n) / 2
    gyro = np.random.default_rng(4).uniform(-1500, 1500, size=(n, 3))
    temperature = np.array([[33.5, 24.0], [34.0, 24.5], [34.5, 25.0]])  # body surface, ambient
    heart = np.array([[1234, 72], [1300, 0], [1400, 75]])  # hr_raw, hr (0: no estimate)
    write_matrix_bin(path, int(START_UTC_S), xyz, gyro_dps=gyro, temperature_c=temperature, heart_rate=heart)

    recording = read_matrix(path)
    data = recording.data

    assert list(data.columns) == [
        "timestamp_unix_ms", "x", "y", "z", "gyro_x", "gyro_y", "gyro_z", "temperature_c",
        "temperature_ambient_c", "hr_bpm", "hr_raw", "Vector Magnitude",
    ]
    assert recording.metadata["gyro_range_dps"] == 2000
    assert recording.metadata["hr_source"] == "matrix_ppg_firmware"
    np.testing.assert_allclose(data[["gyro_x", "gyro_y", "gyro_z"]], gyro, atol=2 * 2000 / 0x7FFF)
    for k in range(3):
        # One temperature and one heart rate per package, repeated on its 100 samples.
        package = data.iloc[k * 100:(k + 1) * 100]
        assert (package["temperature_c"] == temperature[k, 0]).all()
        assert (package["temperature_ambient_c"] == temperature[k, 1]).all()
        assert (package["hr_raw"] == heart[k, 0]).all()
    assert (data["hr_bpm"].iloc[:100] == 72).all() and (data["hr_bpm"].iloc[200:] == 75).all()
    assert data["hr_bpm"].iloc[100:200].isna().all()

    write_matrix_bin(tmp_path / "acc_only.bin", int(START_UTC_S), xyz)
    plain = read_matrix(tmp_path / "acc_only.bin")
    assert list(plain.data.columns) == ["timestamp_unix_ms", "x", "y", "z", "Vector Magnitude"]
    assert "hr_source" not in plain.metadata


def test_sens_keeps_its_other_numeric_columns(tmp_path):
    ts = START_UTC_S * 1000 + np.arange(50) * 80.0
    temperature = np.linspace(30.0, 31.0, 50)
    path = tmp_path / "sens.csv"
    write_sens_csv(path, ts, _accel(50), extra={"Temperature": temperature, "Label": ["walk"] * 50})

    recording = read_sens(path)

    assert list(recording.data.columns) == ["timestamp_unix_ms", "x", "y", "z", "temperature", "Vector Magnitude"]
    np.testing.assert_allclose(recording.data["temperature"], temperature)


def test_polar_companion_readers_name_the_channels_of_the_contract(tmp_path):
    rng = np.random.default_rng(2)
    ts = START_UTC_S * 1000 + np.arange(6) * 1000.0
    sensor_ns = np.arange(6) * 1_000_000_000 + 123
    stamp = "20261021_103000"

    hr_path = tmp_path / f"polar_hr_polar_h10_{stamp}.csv"
    write_polar_hr_csv(hr_path, ts, [60, 61, 62, 63, 64, 65], [[1000.0], [980.4, 990.0], [], [970.0], [960.0], [950.0]])
    hr = read_polar_hr(hr_path)
    assert list(hr.columns) == ["timestamp_unix_ms", "hr_bpm", "rr_intervals_ms", "hr_source"]
    np.testing.assert_allclose(hr["timestamp_unix_ms"], ts)
    assert list(hr["hr_bpm"]) == [60, 61, 62, 63, 64, 65]
    assert list(hr["rr_intervals_ms"][:3]) == ["1000", "980;990", ""]
    assert set(hr["hr_source"]) == {"h10_ecg_firmware"}

    ecg_path = tmp_path / f"polar_ecg_polar_h10_{stamp}.csv"
    ecg = rng.normal(0.0, 500.0, 6)
    write_polar_ecg_csv(ecg_path, ts, ecg, sensor_time_ns=sensor_ns)
    frame = read_polar_ecg(ecg_path)
    assert list(frame.columns) == ["timestamp_unix_ms", "ecg_uv", "sensor_time_ns"]
    np.testing.assert_allclose(frame["ecg_uv"], np.round(ecg))
    assert list(frame["sensor_time_ns"]) == list(sensor_ns)

    ppg_path = tmp_path / f"polar_ppg_verity_arm_{stamp}.csv"
    ppg = rng.uniform(1e5, 4e6, size=(6, 3))
    ambient = rng.uniform(0.0, 1e5, 6)
    write_polar_ppg_csv(ppg_path, ts, ppg, ambient)
    frame = read_polar_ppg(ppg_path)
    assert list(frame.columns) == ["timestamp_unix_ms", "ppg0", "ppg1", "ppg2", "ambient"]
    np.testing.assert_allclose(frame[["ppg0", "ppg1", "ppg2"]], np.round(ppg))
    np.testing.assert_allclose(frame["ambient"], np.round(ambient))

    ppi_path = tmp_path / f"polar_ppi_verity_arm_{stamp}.csv"
    arrival = np.repeat(ts[:2], 3)  # two batches of three pulses
    write_polar_ppi_csv(
        ppi_path, arrival, [900, 910, 920, 930, 940, 950], [66, 65, 65, 64, 63, 63], [5, 6, 7, 8, 9, 10],
        blocker=[0, 0, 1, 0, 0, 0], index_in_frame=[0, 1, 2, 0, 1, 2], sensor_time_ns=sensor_ns,
    )
    frame = read_polar_ppi(ppi_path)
    assert list(frame.columns) == [
        "timestamp_unix_ms", "ppi_ms", "hr_bpm", "error_estimate_ms", "blocker", "skin_contact",
        "skin_contact_supported", "index_in_frame", "hr_source", "sensor_time_ns",
    ]
    assert list(frame["ppi_ms"]) == [900, 910, 920, 930, 940, 950]
    assert list(frame["blocker"]) == [0, 0, 1, 0, 0, 0]
    assert list(frame["index_in_frame"]) == [0, 1, 2, 0, 1, 2]
    assert set(frame["skin_contact"]) == {1} and set(frame["hr_source"]) == {"verity_ppi_firmware"}

    gyro_path = tmp_path / f"polar_gyro_verity_arm_{stamp}.csv"
    gyro = rng.uniform(-200.0, 200.0, size=(6, 3))
    write_polar_gyro_csv(gyro_path, ts, gyro, sensor_time_ns=sensor_ns)
    frame = read_polar_gyro(gyro_path)
    assert list(frame.columns) == ["timestamp_unix_ms", "gyro_x", "gyro_y", "gyro_z", "sensor_time_ns"]
    np.testing.assert_allclose(frame[["gyro_x", "gyro_y", "gyro_z"]], gyro, rtol=1e-9)

    mag_path = tmp_path / f"polar_mag_verity_arm_{stamp}.csv"
    mag = rng.uniform(-0.5, 0.5, size=(6, 3))
    write_polar_mag_csv(mag_path, ts, mag)
    frame = read_polar_mag(mag_path)
    assert list(frame.columns) == ["timestamp_unix_ms", "mag_x", "mag_y", "mag_z"]
    np.testing.assert_allclose(frame[["mag_x", "mag_y", "mag_z"]], mag, rtol=1e-9)

    # The kind comes from the file name; the acceleration is not a companion.
    assert list(read_polar_companion(mag_path).columns) == list(frame.columns)
    with pytest.raises(ValueError):
        read_polar_companion(tmp_path / f"polar_acc_verity_arm_{stamp}.csv")


def test_old_polar_files_without_hr_source_or_frame_index_are_completed(tmp_path):
    ts = START_UTC_S * 1000 + np.arange(4) * 1000.0
    stamp = "20261021_103000"

    old_h10 = tmp_path / f"polar_hr_polar_h10_{stamp}.csv"
    write_polar_hr_csv(old_h10, ts, [60, 61, 62, 63], hr_source=None)
    assert "hr_source" not in old_h10.read_text(encoding="utf-8").splitlines()[0]
    assert set(read_polar_hr(old_h10)["hr_source"]) == {"h10_ecg_firmware"}
    old_verity = tmp_path / f"polar_hr_verity_arm_{stamp}.csv"
    write_polar_hr_csv(old_verity, ts, [70, 71, 72, 73], hr_source=None)
    assert set(read_polar_hr(old_verity)["hr_source"]) == {"verity_ppg_firmware"}
    # A name that says nothing: the device type of the manifest decides, else an unknown Polar.
    anonymous = tmp_path / f"polar_hr_dev1_{stamp}.csv"
    write_polar_hr_csv(anonymous, ts, [70, 71, 72, 73], hr_source=None)
    assert set(read_polar_hr(anonymous)["hr_source"]) == {"polar_firmware"}
    assert set(read_polar_hr(anonymous, hr_source_hint="verity_sense")["hr_source"]) == {"verity_ppg_firmware"}
    assert set(read_polar_hr(anonymous, hr_source_hint="polar_h10 chest")["hr_source"]) == {"h10_ecg_firmware"}
    # A file that has the column keeps it, whatever the hint says.
    write_polar_hr_csv(anonymous, ts, [70, 71, 72, 73], hr_source="verity_ppg_firmware")
    assert set(read_polar_hr(anonymous, hr_source_hint="polar_h10")["hr_source"]) == {"verity_ppg_firmware"}

    old_ppi = tmp_path / f"polar_ppi_verity_arm_{stamp}.csv"
    arrival = np.repeat(ts[:2], [3, 2])  # two batches
    write_polar_ppi_csv(old_ppi, arrival, [900, 910, 920, 930, 940], [66, 65, 65, 64, 63], [5, 6, 7, 8, 9],
                        index_in_frame=None, hr_source=None)
    frame = read_polar_ppi(old_ppi)
    assert "sensor_time_ns" not in frame.columns
    assert list(frame["index_in_frame"]) == [0, 1, 2, 0, 1]
    assert set(frame["hr_source"]) == {"verity_ppi_firmware"}


def test_polar_file_names_split_into_kind_id_and_stamp():
    assert polar_file_parts("polar_acc_polar_h10_20261021_103000.csv") == ("acc", "polar_h10", "20261021_103000")
    assert polar_file_parts("/data/D0/polar_ppi_verity_arm_20261021_103000.csv") == ("ppi", "verity_arm", "20261021_103000")
    assert polar_file_parts("polar_connection_events_verity_arm_20261021_103000.csv") == (
        "connection_events", "verity_arm", "20261021_103000")
    assert polar_file_parts("polar_hr_h10.csv") == ("hr", "h10", "")
    assert polar_file_parts("polar_acc_raw_200hz_demo.csv") == ("acc", "raw_200hz_demo", "")
    assert polar_file_parts("R4H_001_D0_ax3_wnd.cwa") is None
    assert polar_file_parts("polar_light_x_20261021_103000.csv") is None
