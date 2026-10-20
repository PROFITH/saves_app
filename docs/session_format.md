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
| `activities[]` | yes | `activity_index`, `activity_start_time_utc`, `activity_end_time_utc`, `duration_seconds`; optional `activity_name`, `compendium_code`, `planned_duration_seconds` |
| `sync_windows[]` | yes | `event_id`, `event_type`, `activity_index`, `phase`, `window_start_time_utc`, `window_end_time_utc` |
| `video.files[]` | no | Video file names in the session folder |
| `sensors{}` | no | Keyed by sensor ID: `present`, `timestamp_domain`; optional `device_name`, `connection` (`live` or `offline`), `type`, `placement`, `dominance`, `side`, `placement_label`, `serial`, `sampling_rate_hz`, `range_g`, `free_living` |

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
| `polar_h10` | Polar H10 (live) | `polar_acc_*.csv` | `polar_acc` |

ActiGraph and GENEActiv clocks store local time, like Axivity (`device_clock_local`). The readers convert it to UTC with the offset stored in the file (`TimeZone` in the .gt3x `info.txt`, `Time Zone` in the GENEActiv header) when there is one.

`placement` is the body site (`wrist`, `thigh`, `hip`, `chest`, `upper_arm`, `ankle`, `shoe`, `room`, `other`). Two optional fields complete it, and either may be absent when it does not apply (for example, a chest strap):

- `dominance`: `dominant` or `non_dominant`.
- `side`: `left` or `right`.

`placement_label` is the readable label (e.g. `Muñeca no dominante (izquierda)`). SAVES Studio uses it, and the sensor ID, for a file whose serial number (or whose name, containing the sensor ID) matches a manifest sensor. Older manifests used combined values such as `wrist_non_dominant` or `shoe_left`; they are still read.

## Changes

- **1.1, optional fields** (20 October 2026): sensor `type`, `dominance`, `side`, `placement_label`, `serial`, `sampling_rate_hz`, `range_g`, `free_living`; new types `actigraph_gt3x` and `geneactiv`.
- **1.1** (October 2026): sync windows use `window_start_time_utc` / `window_end_time_utc`, as SAVES always expected (lab_protocol_timer 1.0 wrote `window_start` / `window_end`, so its sessions failed to import). Optional activity metadata, `protocol` and the sensor fields above. SAVES still reads 1.0 manifests with either spelling.
- **1.0**: first version.
