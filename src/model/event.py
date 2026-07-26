"""Domain events detected during tracking."""

from dataclasses import dataclass


@dataclass
class ShotEvent:
    """A shot detected from ball acceleration and direction change.

    Attributes:
        frame_number: Frame index where the shot was detected.
        timestamp: Not populated in Phase 5 (requires frame timing
            integration).  Reserved for future use.
        ball_speed_kmh: Ball speed at the moment of the shot, in km/h.
    """

    frame_number: int
    timestamp: float | None
    ball_speed_kmh: float
