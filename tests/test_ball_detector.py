"""Tests for BallDetector — small-object detection for the football."""

import numpy as np
import pytest

from src.backbone.ball_detector import BallDetector
from src.model.detection import Detection


class TestBallDetector:
    """Unit tests for the ball detector wrapper."""

    @pytest.fixture
    def detector(self) -> BallDetector:
        """Load a custom single-class ball model (class 0)."""
        return BallDetector(model_name="yolo26s.pt", device="cpu")

    def test_detect_balls_returns_detections(
        self, detector: BallDetector,
    ) -> None:
        """detect_balls returns a list of Detections (possibly empty)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_balls(frame)
        assert isinstance(result, list)
        assert all(isinstance(d, Detection) for d in result)

    def test_ball_detections_are_well_formed(
        self, detector: BallDetector,
    ) -> None:
        """Every candidate has class 0 and a valid in-frame bounding box."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for det in detector.detect_balls(frame):
            assert det.class_id == 0
            x1, y1, x2, y2 = det.bbox
            assert 0 <= x1 < x2 <= 640
            assert 0 <= y1 < y2 <= 480
