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

1. **Drop the session folder** (or use *Abrir sesión*) and type your **technician code** (your initials). Each technician's work is saved apart in the session folder, so you can stop and carry on another day where you left off, and a second technician can review the same video independently (or start from the first one's synchronization and only count). Studio finds `session_manifest.json`, the videos, `video_sync_session_*.csv` and every sensor file (`.cwa`, `.gt3x`, GENEActiv or Matrix `.bin`, SENS `.csv`, Polar ACC `.csv`) in the folder and its direct subfolders. Each file takes its sensor ID and location from the manifest when the manifest lists it (matched by serial number or by the sensor ID in the file name); otherwise the location is guessed from the file name. Files kept elsewhere can be dropped on the window or added with *Añadir archivos*.
2. Sensors load in the background. Each file is decoded once and cached in `.saves_cache/` next to it, together with a min/max pyramid so hours of data scroll smoothly.
3. Claps or jumps (the kind the timer logged for each sync window) are detected in every sensor. Each offline clock is placed first by voting over all the events, so a clock that is minutes off still finds its claps, and then fitted against the reference (automatic: Polar ACC, else the video claps, else the sensor with most claps). The banner always says the next step, and the quality table shows offset, drift and residual error per sensor.
4. **Video**: when the video has sound, Studio finds the claps in the audio as soon as the session opens and places the video with them. To check or fix one, select it (N/P), step frame by frame (←/→) to the impact and press **M**. Until claps are marked, the video clock comes from `video_sync_session_*.csv`.
5. **Correct**: events flagged in red need a look. Drag a clap line, right-click to exclude it or to put the clap where you clicked; double-click a row of the *Palmadas y saltos* table to jump to it.
6. **Count on the video** (tab *Conteo*): choose what you count (*Pasos*, *Sentarse y levantarse* for the 30-s chair stand test, or any other event added with *Otro evento…*), play the video slowly and press **S** at every event (**A** left, **D** right for kinds that tell sides apart, such as steps); **Backspace** removes the nearest mark and **Ctrl+Z** undoes. The table counts each activity and the rate per minute. A session that does not include one of the kinds can drop it with *Quitar*.
7. **Tasks**: the box at the top left lists the synchronization and one counting task per kind of event, each *No iniciada*, *Iniciada* or *Finalizada*, with who and when. Tasks become *Iniciada* on their own as you work on them; only you mark them *Finalizada*, and a change after that sends the task back to *Iniciada*.
8. **Plots**: *👁 Gráficos* (or right-click on a plot or a sensor) hides the plots that do not help the task at hand, for example all but the two shoes while counting steps. Hidden sensors still take part in the synchronization.
9. **Export** (Ctrl+E) writes to `saves_sync/<technician>/` in the session folder (or another folder) and warns first about any task not finished: `signals/<sensor>.csv.gz` with every sample on the timer's clock and its activity, attempt and phase; `heart_rate/<device>.csv`; `activities.csv`, `phases.csv`, `steps.csv` and `steps_per_activity.csv`, and `events_<kind>.csv` and `events_<kind>_per_activity.csv` for other kinds of events; `tasks.csv` with the checklist; `sync_report.csv`, `anchors.csv`, `video_events.csv` and `qc.json`; and a `LEEME.txt` that says who exported, which tasks were left open and what every column means. All files share `time_unix_ms`. Optionally, the whole recording of each sensor, aligned.

All decisions (sensors, locations, reference, manual claps, exclusions, video marks, events counted, tasks and hidden plots) are saved as you go in `saves_project_<technician>.json` in the session folder, with relative paths, so a session can be reopened or moved; *Guardar* (Ctrl+S) saves at once and the toolbar shows when it last saved. A `saves_project.json` from before technician codes becomes the work of the first technician who opens the folder.

The old MaD GUI viewer is still available with `python main_madgui.py` (needs MaD GUI and PySide2).

## Tests

```sh
pip install -r requirements-dev.txt
pytest
```

The tests build synthetic sessions (claps with known clock offset and drift) and synthetic sensor files. Test files are named `*_test.py` because `test_*.py` is reserved for ad-hoc scripts in `.gitignore`.

## Funding

Developed within the RUN4HEALTH project (Universidad de Granada).
