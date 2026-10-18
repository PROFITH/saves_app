"""
Command-line entry point for headless SAVES processing.

Example
-------
    python cli.py sync \\
        --manifest session/session_manifest.json \\
        --sensor h10=session/polar_acc.csv \\
        --sensor wrist_nd=ax3_1234.cwa \\
        --sensor thigh=sens_thigh.csv:sens \\
        --reference h10 \\
        --out session/saves_sync

Writes ``sync_report.csv`` (one row per sensor), ``anchors.csv`` (one row
per detected clap) and ``qc.json`` (full fits and residuals). With
``--export-aligned`` it also writes each sensor with a
``timestamp_aligned_unix_ms`` column.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from importers.manifest_importer import ManifestImporter
from pipeline.sync_report import (
    anchors_table,
    apply_alignment,
    report_table,
    synchronize_session,
)
from readers import SENSOR_FORMATS, read_sensor


def _parse_sensor_argument(value: str) -> tuple[str, str, str | None]:
    """Parse ``ID=PATH`` or ``ID=PATH:FORMAT``."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Expected ID=PATH[:FORMAT], got '{value}'"
        )

    sensor_id, rest = value.split("=", 1)
    sensor_format = None

    # Split an explicit format without breaking Windows drive letters.
    head, sep, tail = rest.rpartition(":")
    if sep and tail in SENSOR_FORMATS:
        rest, sensor_format = head, tail

    if not sensor_id or not rest:
        raise argparse.ArgumentTypeError(
            f"Expected ID=PATH[:FORMAT], got '{value}'"
        )

    return sensor_id, rest, sensor_format


def _read_video_events(path: Path) -> dict[str, float]:
    """Read a CSV with columns event_id, timestamp_seconds."""
    events = pd.read_csv(path)
    missing = {"event_id", "timestamp_seconds"}.difference(events.columns)
    if missing:
        raise ValueError(
            "Video events file is missing columns: " + ", ".join(sorted(missing))
        )
    return dict(zip(events["event_id"].astype(str), events["timestamp_seconds"].astype(float)))


def run_sync(args: argparse.Namespace) -> int:
    session = ManifestImporter().load(args.manifest)

    recordings = {}
    for sensor_id, path, sensor_format in args.sensor:
        recordings[sensor_id] = read_sensor(path, sensor_format, timezone=args.timezone)

    video_events = None
    reference = args.reference
    if reference.startswith("video:"):
        video_events = _read_video_events(Path(reference.split(":", 1)[1]))
        reference = "video"

    results = synchronize_session(
        session,
        recordings,
        reference=reference,
        video_events=video_events,
        pad_seconds=args.window_pad,
        min_prominence=args.min_prominence,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = report_table(session, results)
    report.to_csv(out / "sync_report.csv", index=False)
    anchors_table(session, results).to_csv(out / "anchors.csv", index=False)

    qc = {
        "session_id": session.session_id,
        "manifest": str(args.manifest),
        "reference": args.reference,
        "window_pad_seconds": args.window_pad,
        "min_prominence": args.min_prominence,
        "timezone": args.timezone,
        "sensors": {
            sensor_id: {
                "sensor_type": result.sensor_type,
                "sampling_rate_hz": result.sampling_rate_hz,
                "metadata": recordings[sensor_id].metadata,
                "n_anchors": len(result.anchors),
                "fit": result.fit,
                "coverage": result.coverage,
                "note": result.note,
            }
            for sensor_id, result in results.items()
        },
    }
    with open(out / "qc.json", "w", encoding="utf-8") as file:
        json.dump(qc, file, indent=2, default=str)

    if args.export_aligned:
        if reference == "video":
            print("--export-aligned needs a sensor reference; skipped.", file=sys.stderr)
        else:
            for sensor_id, result in results.items():
                if result.fit is None:
                    continue
                aligned = apply_alignment(recordings[sensor_id], result.fit)
                aligned.to_csv(out / f"{sensor_id}_aligned.csv.gz", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(report.drop(columns=["session_id"]).to_string(index=False))

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="saves", description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser(
        "sync",
        help="Detect synchronization claps and report offset, drift and residual error.",
    )
    sync.add_argument("--manifest", required=True, type=Path, help="session_manifest.json")
    sync.add_argument(
        "--sensor",
        required=True,
        action="append",
        type=_parse_sensor_argument,
        help=f"ID=PATH[:FORMAT]; FORMAT is one of {', '.join(SENSOR_FORMATS)} (inferred if omitted).",
    )
    sync.add_argument(
        "--reference",
        required=True,
        help="Sensor ID used as the reference timeline, or video:EVENTS.csv.",
    )
    sync.add_argument("--out", required=True, type=Path, help="Output directory.")
    sync.add_argument("--timezone", default="Europe/Madrid", help="Time zone of the Axivity device clocks.")
    sync.add_argument("--window-pad", type=float, default=2.0, help="Seconds added around each sync window.")
    sync.add_argument("--min-prominence", type=float, default=1.0, help="Minimum clap prominence (g).")
    sync.add_argument("--export-aligned", action="store_true", help="Write each sensor with aligned timestamps.")
    sync.set_defaults(func=run_sync)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
