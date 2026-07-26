"""Tests for Detector — YOLO26s wrapper that returns Detection objects."""

import numpy as np
import pytest

from src.model.detection import Detection
from src.service.detector import Detector


class TestDetector:
    """Unit tests for the YOLO detector wrapper."""

    @pytest.fixture
    def detector(self) -> Detector:
        """Load YOLO26s once for the test class."""
        return Detector(model_name="yolo26s.pt", device="cpu")

    def test_detect_returns_list_on_blank_frame(self, detector: Detector) -> None:
        """detect() returns a list (possibly empty) for any image."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert isinstance(result, list)

    def test_detection_has_valid_bbox(self, detector: Detector) -> None:
        """Every Detection has a well-formed bounding box."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            assert 0 <= x1 < x2 <= 640, f"x coords out of range: {d.bbox}"
            assert 0 <= y1 < y2 <= 480, f"y coords out of range: {d.bbox}"
            assert isinstance(d.class_id, int)
            assert 0.0 <= d.confidence <= 1.0
