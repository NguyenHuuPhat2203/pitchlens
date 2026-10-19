"""Domain events detected during tracking."""

from dataclasses import dataclass


@dataclass
class ShotEvent:
    """A shot detected from ball acceleration and direction change.

    Attributes:
        frame_number: Video frame index where the shot was detected.
        timestamp: Seconds from video start.
        ball_speed_kmh: Ball speed at the shot in km/h; None without a
            pitch calibration (pixels cannot be converted to metres).
    """

    frame_number: int
    timestamp: float | None
    ball_speed_kmh: float | None
