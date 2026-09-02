from typing import Optional

from models import SyncAnchor, SyncWindow


class SyncEventDetector:
    """
    Identify the synchronization event within a synchronization window.

    This class does not detect acceleration peaks itself. Instead, it uses
    the candidates produced by ClapDetector and applies protocol-level
    rules to select the event that will become the synchronization anchor.

    Selection rules:

    - ``static_prep``: select the last sufficiently prominent candidate.
    - ``static_post``: select the first sufficiently prominent candidate.

    No fixed temporal interval between claps is assumed.
    """

    def __init__(self, min_prominence: float = 1.0):
        """
        Parameters
        ----------
        min_prominence
            Minimum peak prominence required for a candidate to be
            considered a synchronization clap.
        """
        if min_prominence < 0:
            raise ValueError(
                "min_prominence must be greater than or equal to zero."
            )

        self.min_prominence = min_prominence

    def detect(
        self,
        window: SyncWindow,
        clap_result: dict,
    ) -> Optional[SyncAnchor]:  # noqa: UP045
        """
        Select the synchronization event from detected clap candidates.

        Parameters
        ----------
        window
            Synchronization window from the session manifest.

        clap_result
            Result returned by ClapDetector.detect().

        Returns
        -------
        SyncAnchor or None
            Detected synchronization anchor, or None if no suitable
            synchronization event was found.
        """
        candidates = clap_result.get("candidates", [])

        prominent_candidates = [
            candidate
            for candidate in candidates
            if candidate["prominence"] >= self.min_prominence
        ]

        # ClapDetector sorts candidates by prominence. Restore
        # chronological order before applying the protocol rule.
        prominent_candidates.sort(
            key=lambda candidate: candidate["timestamp_unix_ms"]
        )

        if not prominent_candidates:
            return None

        if window.phase == "static_prep":
            selected = prominent_candidates[-1]
            selection = "last_prominent_candidate"

        elif window.phase == "static_post":
            selected = prominent_candidates[0]
            selection = "first_prominent_candidate"

        else:
            raise ValueError(
                "Unsupported synchronization window phase: "
                f"{window.phase}"
            )

        return SyncAnchor(
            event_id=window.event_id,
            event_type=window.event_type,
            phase=window.phase,
            sensor_timestamp_unix_ms=selected[
                "timestamp_unix_ms"
            ],
            selection_rule=selection,
            prominence=selected["prominence"],
            peak_value=selected["peak_value"],
            candidate_count=len(prominent_candidates),
        )