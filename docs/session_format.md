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
| `sensors{}` | no | Keyed by sensor ID: `present`, `timestamp_domain`; optional `device_name`, `placement`, `connection` (`live` or `offline`) |

`phase` is one of `static_prep`, `activity_main`, `static_post`, `transition_lap`. The clap selected in each window depends on the phase: the last clap in `static_prep`, the first in `static_post` and the most prominent in `transition_lap`.

`timestamp_domain` says which clock a sensor's timestamps follow: `native_plus_host_utc` (live sensor mapped to the protocol app clock, e.g. Polar H10), `device_clock_utc` (offline sensor with a UTC clock, e.g. Parmay Matrix, SENS), `device_clock_local` (offline sensor with a local wall clock, e.g. Axivity), `video_relative` or `unknown`.

## Changes

- **1.1** (October 2026): sync windows use `window_start_time_utc` / `window_end_time_utc`, as SAVES always expected (lab_protocol_timer 1.0 wrote `window_start` / `window_end`, so its sessions failed to import). Optional activity metadata, `protocol` and the sensor fields above. SAVES still reads 1.0 manifests with either spelling.
- **1.0**: first version.
