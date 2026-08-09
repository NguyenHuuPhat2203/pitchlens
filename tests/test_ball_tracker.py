"""Tests for BallTracker — Kalman-filtered ball trajectory."""

import numpy as np
import pytest

from src.model.detection import Detection
from src.serve.pipeline.ball_tracker import BallTracker


def _make_detection(x: float, y: float, w: float = 10, h: float = 10) -> Detection:
    """Create a synthetic ball detection at (x, y)."""
    return Detection(
        bbox=(x, y, x + w, y + h),
        class_id=32,
        confidence=0.9,
    )


class TestBallTracker:
    """Unit tests for the Kalman-based ball tracker."""

    def test_initial_state_is_lost(self) -> None:
        """Before any detection, ball state is 'lost'."""
        tracker = BallTracker()
        ball = tracker.update(None)
        assert ball.state == "lost"

    def test_state_becomes_in_play_after_detection(self) -> None:
        """After receiving a detection, state transitions to 'in_play'."""
        tracker = BallTracker()
        det = _make_detection(100, 100)
        ball = tracker.update(det)
        assert ball.state == "in_play"

    def test_centroid_is_computed(self) -> None:
        """BallTrack centroid is the center of the bbox."""
        tracker = BallTracker()
        det = _make_detection(100, 100, 20, 20)
        ball = tracker.update(det)
        assert ball.centroid == pytest.approx((110.0, 110.0))

    def test_prediction_during_missing_detection(self) -> None:
        """Kalman filter predicts position when detection is missing."""
        tracker = BallTracker()
        # Feed a few detections so the filter has a velocity estimate.
        tracker.update(_make_detection(100, 100))
        tracker.update(_make_detection(110, 100))  # moving right
        tracker.update(_make_detection(120, 100))  # still moving right
        # Now miss one.
        ball = tracker.update(None)
        # Should still be in_play with a predicted position.
        assert ball.state == "in_play"
        # Kalman filter predicts forward motion beyond the last seen x.
        assert ball.centroid[0] > 120

    def test_state_returns_to_lost_after_prolonged_absence(self) -> None:
        """After too many consecutive misses, state goes back to 'lost'."""
        tracker = BallTracker(max_lost_frames=3)
        tracker.update(_make_detection(100, 100))
        for _ in range(4):
            ball = tracker.update(None)
        assert ball.state == "lost"
