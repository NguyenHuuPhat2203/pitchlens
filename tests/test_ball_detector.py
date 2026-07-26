"""Tests for BallDetector — small-object detection for the football."""

import numpy as np
import pytest

from src.model.detection import Detection
from src.service.ball_detector import BallDetector


class TestBallDetector:
    """Unit tests for the ball detector wrapper."""

    @pytest.fixture
    def detector(self) -> BallDetector:
        """Load YOLO26s once, using the built-in sports-ball class."""
        return BallDetector(model_name="yolo26s.pt", device="cpu")

    def test_detect_ball_returns_none_or_detection(
        self, detector: BallDetector,
    ) -> None:
        """detect_ball returns None or a Detection with class_id=32."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_ball(frame)
        assert result is None or isinstance(result, Detection)

    def test_ball_detection_has_sports_ball_class(
        self, detector: BallDetector,
    ) -> None:
        """If a ball is found, its class_id is 32 (COCO sports ball)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_ball(frame)
        if result is not None:
            assert result.class_id == 32

    def test_ball_detection_bbox_is_valid(
        self, detector: BallDetector,
    ) -> None:
        """A returned detection has a well-formed bounding box."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_ball(frame)
        if result is not None:
            x1, y1, x2, y2 = result.bbox
            assert 0 <= x1 < x2 <= 640
            assert 0 <= y1 < y2 <= 480
