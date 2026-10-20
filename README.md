# SAVES

SAVES synchronizes offline wearable sensors with each other and with video after a laboratory session, and lets researchers inspect and label the signals. It reads the session timing written by the live protocol app [lab_protocol_timer](https://github.com/PROFITH/lab_protocol_timer) (see [docs/session_format.md](docs/session_format.md)).

The project has two layers:

- **Headless core** (`readers/`, `algorithms/`, `pipeline/`, `cli.py`): sensor readers, clap detection, offset and drift estimation and quality reports. It needs only numpy, pandas and scipy, so it can process many sessions in batch.
- **SAVES Studio** (`main.py`, `studio/`): desktop app to synchronize the sensors and the video of a session, check the result and export it. The logic lives in `studio/` without Qt (project file, file discovery, signal cache, synchronization engine, video and audio); `studio/ui/` holds the PySide6 windows.
- **Legacy viewer** (`main_madgui.py`, `importers/`, `labels/`): built on [MaD GUI](https://github.com/mad-lab-fau/mad-gui) for manual labelling.

## Supported sensors

| Format | Devices | Reader |
|---|---|---|
| `.cwa` | Axivity AX3, AX6 (accelerometer and gyroscope) | `readers/cwa.py` |
| `.gt3x` | ActiGraph GT3X+, wGT3X-BT, GT9X Link (`log.bin` and older `activity.bin` layouts; idle-sleep gaps filled with the last sample and flagged in `imputed`) | `readers/gt3x.py` |
| `.bin` | GENEActiv (calibrated with the gains and offsets of the file header; also light, button and temperature) | `readers/geneactiv.py` |
| `.bin` | Parmay Matrix (told apart from GENEActiv by the file content) | `readers/matrix.py` |
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

## SAVES Studio

```sh
pip install -r requirements-core.txt -r requirements-gui.txt   # Python 3.10-3.13, no MaD GUI needed
python main.py            # or: python main.py path/to/session_folder
python main.py --demo     # synthetic session with four sensors and a video
```

Workflow:

1. **Drop the session folder** (or use *Abrir sesión*). Studio finds `session_manifest.json`, the videos, `video_sync_session_*.csv` and every sensor file (`.cwa`, `.gt3x`, GENEActiv or Matrix `.bin`, SENS `.csv`, Polar ACC `.csv`) in the folder and its direct subfolders. Each file takes its sensor ID and location from the manifest when the manifest lists it (matched by serial number or by the sensor ID in the file name); otherwise the location is guessed from the file name. Files kept elsewhere can be dropped on the window or added with *Añadir archivos*.
2. Sensors load in the background. Each file is decoded once and cached in `.saves_cache/` next to it, together with a min/max pyramid so hours of data scroll smoothly.
3. Claps are detected in every sync window and each sensor is fitted against the reference (automatic: Polar ACC, else the video claps, else the sensor with most claps). The banner always says the next step, and the quality table shows offset, drift and residual error per sensor.
4. **Correct**: drag a clap line, right-click to exclude it or to put the clap where you clicked; double-click a row of the *Palmadas* table to jump to it.
5. **Video**: select a clap (N/P), step frame by frame (←/→) to the impact and press **M**; Studio moves on to the next clap. *Detectar palmadas en el audio* does it automatically when the video has sound. Until claps are marked, the video clock comes from `video_sync_session_*.csv`.
6. **Export** (Ctrl+E) writes `sync_report.csv`, `anchors.csv`, `video_events.csv`, `activities_<sensor>.csv` (activity times on each sensor's own clock), `qc.json` and, optionally, the aligned signals.

All decisions (sensors, locations, reference, manual claps, exclusions, video marks) are saved automatically in `saves_project.json` in the session folder, with relative paths, so a session can be reopened or moved.

The old MaD GUI viewer is still available with `python main_madgui.py` (needs MaD GUI and PySide2).

## Tests

```sh
pip install -r requirements-dev.txt
pytest
```

The tests build synthetic sessions (claps with known clock offset and drift) and synthetic sensor files. Test files are named `*_test.py` because `test_*.py` is reserved for ad-hoc scripts in `.gitignore`.

## Funding

Developed within the RUN4HEALTH project (Universidad de Granada).
