# SAVES

SAVES synchronizes offline wearable sensors with each other and with video after a laboratory session, and lets researchers inspect and label the signals. It reads the session timing written by the live protocol app [lab_protocol_timer](https://github.com/PROFITH/lab_protocol_timer) (see [docs/session_format.md](docs/session_format.md)).

The project has two layers:

- **Headless core** (`readers/`, `algorithms/`, `pipeline/`, `cli.py`): sensor readers, clap detection, offset and drift estimation and quality reports. It needs only numpy, pandas and scipy, so it can process many sessions in batch.
- **Desktop GUI** (`main.py`, `importers/`, `labels/`): built on [MaD GUI](https://github.com/mad-lab-fau/mad-gui) for visual inspection, video playback and manual labelling.

## Supported sensors

| Format | Devices | Reader |
|---|---|---|
| `.cwa` | Axivity AX3, AX6 (accelerometer and gyroscope) | `readers/cwa.py` |
| `.bin` | Parmay Matrix | `readers/matrix.py` |
| `.csv` | SENS motion | `readers/sens.py` |
| `.csv` | Polar H10 / Verity Sense acceleration exported by lab_protocol_timer | `readers/polar.py` |

Axivity clocks store local time; it is converted to UTC with `--timezone` (default `Europe/Madrid`), using the offset at the start of the recording.

## Synchronization report

```sh
pip install -r requirements-core.txt

python cli.py sync \
    --manifest session/session_manifest.json \
    --sensor h10=session/polar_acc.csv \
    --sensor wrist_nd=ax3_wrist.cwa \
    --sensor matrix_nd=matrix_wrist.bin \
    --sensor thigh=sens_thigh.csv:sens \
    --reference h10 \
    --out session/saves_sync
```

For every sync window logged by the protocol app, SAVES detects the clap in each sensor and fits `reference_time = intercept + slope × sensor_time` against the same claps in the reference sensor (or in manual video events with `--reference video:events.csv`, columns `event_id,timestamp_seconds`). Outputs:

- `sync_report.csv`: per sensor, claps detected, offset (ms), drift (ppm), residual RMSE and maximum (ms), share of the session covered by samples and gaps.
- `anchors.csv`: every detected clap with its residual.
- `qc.json`: full fits, residuals and sensor metadata.
- With `--export-aligned`, `<sensor>_aligned.csv.gz` with a `timestamp_aligned_unix_ms` column.

Use `--window-pad` (seconds, default 2) when an offline clock is far from the protocol app clock.

## Desktop GUI

```sh
python main.py
```

Requires MaD GUI and PySide2 in addition to the core requirements.

## Tests

```sh
pip install -r requirements-dev.txt
pytest
```

The tests build synthetic sessions (claps with known clock offset and drift) and synthetic sensor files. Test files are named `*_test.py` because `test_*.py` is reserved for ad-hoc scripts in `.gitignore`.

## Funding

Developed within the RUN4HEALTH project (Universidad de Granada).
