"""Tests for ShotDetector — acceleration-based shot event detection."""

import math

import pytest

from src.model.track import BallTrack
from src.serve.pipeline.shot_detector import ShotDetector


def _make_ball(
    centroid: tuple[float, float],
    velocity: tuple[float, float],
    state: str = "in_play",
) -> BallTrack:
    """Create a BallTrack with the given centroid and velocity."""
    return BallTrack(
        track_id=0,
        bbox=(centroid[0] - 5, centroid[1] - 5,
              centroid[0] + 5, centroid[1] + 5),
        centroid=centroid,
        velocity=velocity,
        state=state,
    )


class TestShotDetector:
    """Unit tests for the shot detector."""

    def test_no_shot_on_steady_motion(self) -> None:
        """Ball moving at constant velocity does not trigger a shot."""
        detector = ShotDetector()
        # Feed steady rightward motion.
        for i in range(5):
            result = detector.update(
                _make_ball((float(i * 10), 100.0), (10.0, 0.0))
            )
        assert result is None

    def test_shot_on_sudden_direction_change_and_acceleration(self) -> None:
        """A sharp turn with high acceleration triggers a shot."""
        detector = ShotDetector(angle_threshold=20.0, accel_threshold=30.0)
        # Ball moving gently right.
        detector.update(_make_ball((100.0, 100.0), (2.0, 0.0)))
        detector.update(_make_ball((102.0, 100.0), (2.0, 0.0)))
        # Sudden kick upward with large velocity.
        result = detector.update(_make_ball((102.0, 102.0), (0.0, 50.0)))
        assert result is not None
        assert result.frame_number == 2

    def test_no_shot_when_ball_is_lost(self) -> None:
        """A lost ball never triggers a shot."""
        detector = ShotDetector()
        detector.update(_make_ball((100.0, 100.0), (5.0, 0.0)))
        result = detector.update(
            _make_ball((0.0, 0.0), (0.0, 0.0), state="lost")
        )
        assert result is None

    def test_detector_needs_history(self) -> None:
        """First call to update never returns a shot (needs prior state)."""
        detector = ShotDetector()
        result = detector.update(_make_ball((100.0, 100.0), (50.0, 0.0)))
        assert result is None
