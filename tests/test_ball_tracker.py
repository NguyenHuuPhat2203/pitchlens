"""Tests for BallTracker — Kalman-filtered ball trajectory."""

import pytest

from src.model.detection import Detection
from src.serve.pipeline.ball_tracker import BallTracker


def _det(x: float, y: float, w: float = 10, h: float = 10, conf: float = 0.9) -> Detection:
    """Create a synthetic ball detection with top-left corner (x, y)."""
    return Detection(bbox=(x, y, x + w, y + h), class_id=0, confidence=conf)


def test_initial_state_is_lost() -> None:
    assert BallTracker().update([]).state == "lost"


def test_first_detection_starts_track_at_its_centre() -> None:
    ball = BallTracker().update([_det(100, 100, 20, 20)])
    assert ball.state == "in_play"
    assert ball.centroid == pytest.approx((110.0, 110.0))


def test_prediction_during_missing_detection() -> None:
    tracker = BallTracker()
    for x in (100, 110, 120):
        tracker.update([_det(x, 100)])
    ball = tracker.update([])
    assert ball.state == "in_play"
    assert ball.centroid[0] > 125  # last seen centre was 125


def test_state_returns_to_lost_after_prolonged_absence() -> None:
    tracker = BallTracker(max_lost_frames=3)
    tracker.update([_det(100, 100)])
    for _ in range(4):
        ball = tracker.update([])
    assert ball.state == "lost"


def test_far_false_positive_is_ignored_even_with_higher_confidence() -> None:
    tracker = BallTracker(gate_px=50)
    for x in (100, 104, 108):
        tracker.update([_det(x, 100)])
    ball = tracker.update([_det(600, 400, conf=0.99), _det(112, 100, conf=0.4)])
    assert ball.centroid[0] == pytest.approx(117, abs=3)
    assert ball.centroid[1] == pytest.approx(105, abs=3)


def test_lone_far_detection_counts_as_a_miss() -> None:
    tracker = BallTracker(max_lost_frames=1, gate_px=50)
    tracker.update([_det(100, 100)])
    tracker.update([_det(600, 400)])
    assert tracker.update([_det(600, 400)]).state == "lost"


def test_reacquires_at_new_detection_after_loss() -> None:
    tracker = BallTracker(max_lost_frames=2, gate_px=50)
    for x in (100, 110, 120):
        tracker.update([_det(x, 100)])
    for _ in range(10):
        tracker.update([])
    ball = tracker.update([_det(600, 400)])
    assert ball.state == "in_play"
    assert ball.centroid == pytest.approx((605.0, 405.0))
