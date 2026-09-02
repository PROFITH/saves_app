import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from models import Activity, Participant, Session, SyncWindow


class ManifestImporter:
    """Load a SAVES session manifest into domain models."""

    def load(self, file_path: Union[str, Path]) -> Session:  # noqa: UP007
        """
        Load and parse a session_manifest.json file.

        Parameters
        ----------
        file_path
            Path to the session manifest.

        Returns
        -------
        Session
            Parsed session metadata.

        Raises
        ------
        FileNotFoundError
            If the manifest does not exist.
        ValueError
            If the manifest is invalid or uses unsupported schema information.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Session manifest not found: {path}"
            )

        with path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        self._validate_manifest(manifest)

        return self._build_session(manifest)

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        """Parse an ISO-8601 timestamp and return a UTC-aware datetime."""
        if not isinstance(value, str):
            raise TypeError(
                f"Expected ISO-8601 timestamp as string, got {type(value).__name__}"
            )

        normalized = value.replace("Z", "+00:00")

        timestamp = datetime.fromisoformat(normalized)

        if timestamp.tzinfo is None:
            raise ValueError(
                f"Timestamp must contain timezone information: {value}"
            )

        return timestamp.astimezone(timezone.utc)

    @classmethod
    def _parse_activity(cls, data: dict) -> Activity:
        """Convert one manifest activity into an Activity model."""
        return Activity(
            activity_index=int(data["activity_index"]),
            start_time=cls._parse_datetime(
                data["activity_start_time_utc"]
            ),
            end_time=cls._parse_datetime(
                data["activity_end_time_utc"]
            ),
            duration_seconds=float(data["duration_seconds"]),
        )

    @classmethod
    def _parse_sync_window(cls, data: dict) -> SyncWindow:
        """Convert one manifest synchronization window into a model."""
        return SyncWindow(
            event_id=str(data["event_id"]),
            event_type=str(data["event_type"]),
            activity_index=int(data["activity_index"]),
            phase=str(data["phase"]),
            window_start=cls._parse_datetime(
                data["window_start_time_utc"]
            ),
            window_end=cls._parse_datetime(
                data["window_end_time_utc"]
            ),
        )

    @staticmethod
    def _parse_participant(data: dict) -> Participant:
        """Convert manifest participant metadata into a Participant model."""
        return Participant(
            participant_id=str(data["participant_id"]),
            redcap_event_name=str(data["redcap_event_name"]),
        )

    def _build_session(self, manifest: dict) -> Session:
        """Build a Session domain object from parsed manifest data."""

        participants = [
            self._parse_participant(item)
            for item in manifest.get("participants", [])
        ]

        activities = [
            self._parse_activity(item)
            for item in manifest.get("activities", [])
        ]

        sync_windows = [
            self._parse_sync_window(item)
            for item in manifest.get("sync_windows", [])
        ]

        video = manifest.get("video", {})

        return Session(
            schema_version=str(manifest["schema_version"]),
            session_id=str(manifest["session_id"]),
            created_by=str(manifest["created_by"]),
            start_time=self._parse_datetime(
                manifest["session_start_time_utc"]
            ),
            end_time=self._parse_datetime(
                manifest["session_end_time_utc"]
            ),
            participants=participants,
            activities=activities,
            sync_windows=sync_windows,
            video_files=[
                str(path)
                for path in video.get("files", [])
            ],
            sensors=manifest.get("sensors", {}),
        )

    @staticmethod
    def _validate_manifest(manifest: dict) -> None:
        """Validate the minimum structure required by SAVES."""

        if not isinstance(manifest, dict):
            raise TypeError(
                "Session manifest must contain a JSON object."
            )

        required_fields = [
            "schema_version",
            "session_id",
            "session_start_time_utc",
            "session_end_time_utc",
            "activities",
            "sync_windows",
        ]

        missing = [
            field
            for field in required_fields
            if field not in manifest
        ]

        if missing:
            raise ValueError(
                "Session manifest is missing required fields: "
                + ", ".join(missing)
            )

        if manifest["schema_version"] != "1.0":
            raise ValueError(
                f"Unsupported manifest schema version: "
                f"{manifest['schema_version']}"
            )