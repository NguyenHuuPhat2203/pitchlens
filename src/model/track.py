"""A tracked player across consecutive frames."""

from dataclasses import dataclass


@dataclass
class PlayerTrack:
    """One tracked person with a persistent identity.

    Attributes:
        track_id: Integer ID assigned by the tracker, stable across
            frames while the person remains in view.
        bbox: Pixel coordinates (x1, y1, x2, y2).
        class_id: Model class index for the detection.
        confidence: Detection confidence in [0.0, 1.0].
    """

    track_id: int
    bbox: tuple[float, float, float, float]
    class_id: int
    confidence: float
