from typing import Optional

from models import SyncAnchor, SyncWindow

# Around the expected time of a cued event, how early and how late (ms) a
# candidate may be to be taken as that event.
CUE_EARLY_MS = 1200.0
CUE_LATE_MS = 2500.0


def candidate_time(candidate: dict) -> float:
    """Onset of the impact when known, otherwise its peak."""
    return float(candidate.get("onset_unix_ms", candidate["timestamp_unix_ms"]))


class SyncEventDetector:
    """
    Identify the synchronization event within a synchronization window.

    This class does not detect acceleration peaks itself. Instead, it uses
    the candidates produced by ClapDetector or JumpDetector and applies
    protocol-level rules to select the event that will become the
    synchronization anchor.

    When the protocol app logged the 3-2-1 cue, the candidate closest to the
    expected time of the clap or landing is chosen. Otherwise:

    - ``static_prep``: select the last sufficiently prominent candidate.
    - ``static_post``: select the first sufficiently prominent candidate.
    - ``transition_lap`` and any other phase: select the most prominent
      candidate.

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
        expected_ms: Optional[float] = None,  # noqa: UP045
    ) -> Optional[SyncAnchor]:  # noqa: UP045
        """
        Select the synchronization event from detected candidates.

        Parameters
        ----------
        window
            Synchronization window from the session manifest.

        clap_result
            Result returned by ClapDetector.detect() or JumpDetector.detect().

        expected_ms
            Expected time of the event in the sensor clock (Unix ms), when
            the cue time is known.

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

        # The detectors sort candidates by prominence. Restore
        # chronological order before applying the protocol rule.
        prominent_candidates.sort(key=candidate_time)

        if not prominent_candidates:
            return None

        selected = None
        if expected_ms is not None:
            near = [
                candidate
                for candidate in prominent_candidates
                if expected_ms - CUE_EARLY_MS <= candidate_time(candidate) <= expected_ms + CUE_LATE_MS
            ]
            if near:
                selected = min(near, key=lambda c: abs(candidate_time(c) - expected_ms))
                selection = "closest_to_cue"

        if selected is not None:
            pass
        elif window.phase == "static_prep":
            selected = prominent_candidates[-1]
            selection = "last_prominent_candidate"

        elif window.phase == "static_post":
            selected = prominent_candidates[0]
            selection = "first_prominent_candidate"

        else:
            selected = max(
                prominent_candidates,
                key=lambda candidate: candidate["prominence"],
            )
            selection = "most_prominent_candidate"

        return SyncAnchor(
            event_id=window.event_id,
            event_type=window.event_type,
            phase=window.phase,
            sensor_timestamp_unix_ms=int(round(candidate_time(selected))),
            selection_rule=selection,
            prominence=selected["prominence"],
            peak_value=selected["peak_value"],
            candidate_count=len(prominent_candidates),
        )
