# SAVES session format

A laboratory session is described by one `session_manifest.json` written by the live protocol app (lab_protocol_timer) and read by SAVES. Any other app can produce sessions for SAVES by writing the same file. The machine-readable definition is [`schemas/session_manifest.v1.1.schema.json`](../schemas/session_manifest.v1.1.schema.json) and a complete example is [`examples/session_manifest.v1.1.json`](examples/session_manifest.v1.1.json).

## Rules

- All timestamps are ISO-8601 strings with an explicit offset, written in UTC (`2026-10-27T08:30:20.000Z`).
- Times in the manifest come from the clock of the device running the protocol app. Offline sensors keep their own clocks; SAVES relates both through the synchronization claps.
- `schema_version` changes when a field is renamed or removed. Adding optional fields keeps the version.

## Fields

| Field | Required | Content |
|---|---|---|
| `schema_version` | yes | `"1.1"` |
| `session_id` | yes | Unique identifier of the session |
| `created_by` | yes | App that wrote the file |
| `session_start_time_utc`, `session_end_time_utc` | yes | Start and end of the session |
| `protocol` | no | `protocol_id`, `protocol_version` of the activity battery |
| `participants[]` | no | `participant_id`, `redcap_event_name` |
| `activities[]` | yes | `activity_index`, `activity_start_time_utc`, `activity_end_time_utc`, `duration_seconds`; optional `activity_name`, `compendium_code`, `planned_duration_seconds`, `attempt` (1 unless the activity was repeated), `activity_kind` (`calibration` for the static calibration), `redcap_instance` (the `lab_activity` instance of that attempt in REDCap) |
| `sync_windows[]` | yes | `event_id`, `event_type`, `activity_index`, `phase`, `window_start_time_utc`, `window_end_time_utc` |
| `video.files[]` | no | Video file names in the session folder |
| `sensors{}` | no | Keyed by sensor ID: `present`, `timestamp_domain`; optional `device_name`, `connection` (`live` or `offline`), `type`, `placement`, `dominance`, `side`, `placement_label`, `serial`, `sampling_rate_hz`, `range_g`, `free_living`; for a live device also `signals` and `files` (see [Files of a live device](#files-of-a-live-device)) |

`phase` is one of `static_prep`, `activity_main`, `static_post`, `transition_lap`. The clap selected in each window depends on the phase: the last clap in `static_prep`, the first in `static_post` and the most prominent in `transition_lap`.

`timestamp_domain` says which clock a sensor's timestamps follow: `native_plus_host_utc` (live sensor mapped to the protocol app clock, e.g. Polar H10), `device_clock_utc` (offline sensor with a UTC clock, e.g. Parmay Matrix, SENS), `device_clock_local` (offline sensor with a local wall clock, e.g. Axivity), `video_relative` or `unknown`.

### Sensor types and placement

`type` names the device, and SAVES reads its files with the matching reader:

| `type` | Device | Files | Reader (`readers.FORMAT_BY_SENSOR_TYPE`) |
|---|---|---|---|
| `ax3`, `ax6` | Axivity AX3, AX6 | `.cwa` | `cwa` |
| `actigraph_gt3x` | ActiGraph GT3X+, wGT3X-BT, GT9X Link | `.gt3x` | `gt3x` |
| `geneactiv` | GENEActiv | `.bin` | `geneactiv` |
| `matrix` | Parmay Matrix | `.bin` | `matrix` |
| `sens` | SENS motion | `.csv` | `sens` |
| `polar_h10` | Polar H10 (live) | `polar_acc_*.csv`, plus `polar_hr_*`, `polar_ecg_*` (below) | `polar_acc` |
| `verity_sense` | Polar Verity Sense (live) | `polar_acc_*.csv`, plus `polar_hr_*`, `polar_ppg_*`, `polar_ppi_*`, `polar_gyro_*`, `polar_mag_*` (below) | `polar_acc` |

ActiGraph and GENEActiv clocks store local time, like Axivity (`device_clock_local`). The readers convert it to UTC with the offset stored in the file (`TimeZone` in the .gt3x `info.txt`, `Time Zone` in the GENEActiv header) when there is one.

One device has one clock. SAVES finds the clock fit (offset and drift) of a device from its accelerometer (the claps and jumps) and applies that same fit to every other signal the device recorded, because they share its clock: the gyroscope, light and temperature of an Axivity file; the light, button and temperature of a GENEActiv file; the `lux` records of an ActiGraph file; the gyroscope, temperature and heart rate of a Parmay Matrix file; any other numeric column of a SENS file; and, for a live Polar device, every companion file listed below.

`placement` is the body site (`wrist`, `thigh`, `hip`, `chest`, `upper_arm`, `ankle`, `shoe`, `room`, `other`). Two optional fields complete it, and either may be absent when it does not apply (for example, a chest strap):

- `dominance`: `dominant` or `non_dominant`.
- `side`: `left` or `right`.

`placement_label` is the readable label (e.g. `Muñeca no dominante (izquierda)`). SAVES Studio uses it, and the sensor ID, for a file whose serial number (or whose name, containing the sensor ID) matches a manifest sensor. Older manifests used combined values such as `wrist_non_dominant` or `shoe_left`; they are still read.

## Files of a live device

lab_protocol_timer writes one CSV per stream of each live Polar device, named `polar_<kind>_<id>_<stamp>.csv`: `<id>` is the sensor ID of the session (the key in `sensors{}`, e.g. `polar_h10`, `verity_arm`) and `<stamp>` the session stamp (`YYYYMMDD_HHMMSS`). Every file starts with `timestamp_iso` (UTC, host clock), `participant_id`, `activity_index` and `phase_name`. A stream with no rows gets no file.

| File | Columns after the four shared ones | Device | Rate | Notes |
|---|---|---|---|---|
| `polar_hr_<id>_<stamp>.csv` | `hr_bpm,rr_intervals_ms,hr_source` | H10, Verity | 1 Hz (every 5 s on the Verity while PPI runs) | BLE Heart Rate Service. `rr_intervals_ms` = `;`-joined ms, quoted, empty on the Verity. `hr_source` (older files lack it): `h10_ecg_firmware` (the H10 computes HR from its ECG), `verity_ppg_firmware` (the Verity Sense computes it from its PPG), `polar_firmware` (unknown model). |
| `polar_acc_<id>_<stamp>.csv` | `acc_x_mg,acc_y_mg,acc_z_mg,sensor_time_ns` | H10 (200 Hz ±8 g), Verity (52 Hz ±8 g) | | The synchronization signal. |
| `polar_ecg_<id>_<stamp>.csv` | `ecg_uv,sensor_time_ns` | H10 | 130 Hz | Microvolts. |
| `polar_ppg_<id>_<stamp>.csv` | `ppg0,ppg1,ppg2,ambient,sensor_time_ns` | Verity | 55 Hz | Raw 22-bit counts of the three green PPG channels and the ambient-light channel (PMD PPG frame type 0). |
| `polar_ppi_<id>_<stamp>.csv` | `ppi_ms,hr_bpm,error_estimate_ms,blocker,skin_contact,skin_contact_supported,index_in_frame,sensor_time_ns,hr_source` | Verity | one row per pulse, in batches | Polar's PPI algorithm: pulse-to-pulse interval (ms), its HR (`hr_source=verity_ppi_firmware`), error estimate (ms), `blocker` 1 = movement, skin contact flags (unreliable on the Verity). When the sensor stamps the frame, `timestamp_iso`/`sensor_time_ns` are per pulse (last pulse at the frame time, earlier ones back by the intervals); otherwise `timestamp_iso` is the arrival of the batch and `sensor_time_ns` is empty, so the pulses of a batch share the time and `index_in_frame` orders them. The first batch arrives ~25 s after the start. |
| `polar_gyro_<id>_<stamp>.csv` | `gyro_x_dps,gyro_y_dps,gyro_z_dps,sensor_time_ns` | Verity | 52 Hz ±2000 dps | Degrees per second (raw × factor reported by the sensor). |
| `polar_mag_<id>_<stamp>.csv` | `mag_x_gauss,mag_y_gauss,mag_z_gauss,sensor_time_ns` | Verity | 50 Hz ±50 G | Gauss (raw × factor). |
| `polar_clock_<id>_<stamp>.csv` | `arrival_utc,stream,frame_sensor_time_ns,offset_us` | both | per frame | Clock fit of each frame; `stream` is `acc`, `ecg`, `ppg`, `gyro` or `mag` (PPI batches arrive late and never feed the clock fit). |
| `polar_connection_events_<id>_<stamp>.csv` | `timestamp_utc,event,detail` | both | | Connection log. A stream the device refused appears as `sensor_start_failed`. |

`sensor_time_ns` is the sensor's own clock (PMD timestamp) before the mapping to the host clock; `timestamp_iso` is that time mapped to the host clock (`native_plus_host_utc`).

For each live sensor, `sensors{}` may also carry:

- `signals`: the streams requested from the device, e.g. `["acc", "ecg", "heart_rate", "rr"]` for an H10 or `["acc", "gyro", "heart_rate", "mag", "ppg", "ppi"]` for a Verity Sense.
- `files`: the names (no folder) of the device's CSV files that got rows.

SAVES Studio synchronizes the device with its `polar_acc_*` file (`readers/polar.py`, `read_polar_acc`) and reads the companions with `read_polar_companion` (`readers/polar.py`); they are never sensors of their own. The export writes them with the clock fit of the device's acceleration: `signals/<sensor>_ecg.csv.gz`, `signals/<sensor>_ppg.csv.gz`, `signals/<sensor>_gyro.csv.gz`, `signals/<sensor>_mag.csv.gz`, `heart_rate/<sensor>.csv` and `heart_rate/<sensor>_ppi.csv`, where `<sensor>` is the sensor ID of the device's acceleration file; a device whose acceleration is not among the synchronized sensors keeps the host clock as written (identity fit) and is named by its `<id>`, as `LEEME.txt` says. A heart-rate file without `hr_source` gets it from the device `type` or `device_name` of the manifest, else from the file name (`polar_h10*`/`h10` → `h10_ecg_firmware`, `verity*`/`sense` → `verity_ppg_firmware`, else `polar_firmware`).

## Changes

- **1.1, optional fields** (22 October 2026): sensor `signals` and `files` of a live device, and the catalogue of its files (`polar_ppg_*`, `polar_ppi_*`, `polar_gyro_*`, `polar_mag_*` and the `hr_source` column of `polar_hr_*` are new); new type `verity_sense`.

- **1.1, optional fields** (20 October 2026): sensor `type`, `dominance`, `side`, `placement_label`, `serial`, `sampling_rate_hz`, `range_g`, `free_living`; new types `actigraph_gt3x` and `geneactiv`.
- **1.1** (October 2026): sync windows use `window_start_time_utc` / `window_end_time_utc`, as SAVES always expected (lab_protocol_timer 1.0 wrote `window_start` / `window_end`, so its sessions failed to import). Optional activity metadata, `protocol` and the sensor fields above. SAVES still reads 1.0 manifests with either spelling.
- **1.0**: first version.
