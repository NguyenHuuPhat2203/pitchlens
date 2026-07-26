"""Tracked objects across consecutive frames."""

from __future__ import annotations

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
        team: Team label — ``"A"``, ``"B"``, ``"referee"``, or
            ``None`` if not yet classified.
        speed_kmh: Real-world speed in km/h, populated after
            calibration (Phase 4).
        direction_deg: Movement direction in degrees, 0 = right.
    """

    track_id: int
    bbox: tuple[float, float, float, float]
    class_id: int
    confidence: float
    team: str | None = None
    speed_kmh: float | None = None
    direction_deg: float | None = None


@dataclass
class BallTrack:
    """The football's tracked position and motion across frames.

    Attributes:
        track_id: Always 0 — there is only one ball.
        bbox: Pixel bounding box (may be estimated during occlusion).
        centroid: Center (cx, cy) of the ball in pixels.
        velocity: Pixel velocity vector (vx, vy) from the Kalman filter.
        state: ``"in_play"`` when actively tracked, ``"lost"`` after
            prolonged occlusion.
    """

    track_id: int
    bbox: tuple[float, float, float, float]
    centroid: tuple[float, float]
    velocity: tuple[float, float]
    state: str
