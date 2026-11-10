import numpy as np
import pandas as pd

from algorithms.offset_search import (
    HOST_SEARCH_S,
    NETWORK_CLOCK_SEARCH_S,
    OFFLINE_SEARCH_S,
    search_range_for,
    vote_offset,
)
from readers.sens import read_sens


def test_sens_export_unixts_is_utc(tmp_path):
    # Header and first row of a real SENS motion export (August, CEST).
    path = tmp_path / "export_thigh_acc.csv"
    path.write_text(
        "utc, local, unixts, x, y, z\n"
        "2026-08-20T14:53:54.070000Z, 2026-08-20T16:53:54.070000+02:00, 1787237634070, -0.95, 0.38, 0.07\n"
        "2026-08-20T14:53:54.149000Z, 2026-08-20T16:53:54.149000+02:00, 1787237634149, -0.95, 0.38, 0.07\n"
    )
    recording = read_sens(path)
    first = pd.Timestamp(recording.data["timestamp_unix_ms"].iloc[0], unit="ms", tz="UTC")
    assert first == pd.Timestamp("2026-08-20T14:53:54.070Z")


def test_sens_searches_only_minutes_around_its_network_clock():
    assert search_range_for("sens") == NETWORK_CLOCK_SEARCH_S < 600
    assert search_range_for("polar_acc") == HOST_SEARCH_S
    assert search_range_for("cwa") == OFFLINE_SEARCH_S


def test_sens_ignores_a_false_match_an_hour_away():
    # A thigh with steps and sit-to-stands all day: two impacts happen to be
    # as far apart as sync_1 and sync_2, 50 min before the session. Only
    # the jump of sync_2 is felt at its true time (the thigh misses the clap).
    events = [("sync_1", 0.0, 1.0), ("sync_2", 20 * 60_000.0, 1.0)]
    false_pair = [-50 * 60_000.0, -50 * 60_000.0 + 20 * 60_000.0 + 300.0]
    times = np.array([*false_pair, 20 * 60_000.0 + 150.0])
    offline = vote_offset(times, events, OFFLINE_SEARCH_S)
    assert offline.accepted and offline.offset_ms < -40 * 60_000.0
    sens = vote_offset(times, events, search_range_for("sens"))
    assert not sens.accepted  # falls back to the manifest times: offset 0


def test_two_events_that_both_match_still_give_the_offset():
    events = [("sync_1", 0.0, 1.0), ("sync_2", 20 * 60_000.0, 1.0)]
    times = np.array([3_600_000.0, 3_600_000.0 + 20 * 60_000.0 + 200.0, 5_000_000.0])
    result = vote_offset(times, events, OFFLINE_SEARCH_S)
    assert result.accepted
    assert abs(result.offset_ms - 3_600_100.0) < 1.0
