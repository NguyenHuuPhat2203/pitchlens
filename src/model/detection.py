"""A single-frame observation from the object detector."""

from dataclasses import dataclass


@dataclass
class Detection:
    """One detected object in a frame.

    Attributes:
        bbox: Pixel coordinates (x1, y1, x2, y2) — top-left and
            bottom-right corners of the bounding box.
        class_id: Integer label from the model's class index.
        confidence: Detection confidence in [0.0, 1.0].
    """

    bbox: tuple[float, float, float, float]
    class_id: int
    confidence: float
