"""Tests for Tracker — OC-SORT wrapper producing PlayerTrack objects."""

import numpy as np
import pytest

from src.model.track import PlayerTrack
from src.service.tracker import Tracker


def _make_synthetic_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Return a blank BGR frame for tracker testing."""
    return np.zeros((height, width, 3), dtype=np.uint8)


class TestTracker:
    """Unit tests for the OC-SORT tracker wrapper."""

    @pytest.fixture
    def tracker(self) -> Tracker:
        """Create a tracker with CPU YOLO26s."""
        return Tracker(
            model_name="yolo26s.pt",
            tracker_config="ocsort.yaml",
            device="cpu",
        )

    def test_track_returns_list(self, tracker: Tracker) -> None:
        """track() returns a list of PlayerTrack objects."""
        frame = _make_synthetic_frame()
        tracks = tracker.track(frame)
        assert isinstance(tracks, list)
        for p in tracks:
            assert isinstance(p, PlayerTrack)

    def test_track_has_valid_ids(self, tracker: Tracker) -> None:
        """Every PlayerTrack has a non-negative integer track_id."""
        frame = _make_synthetic_frame()
        tracks = tracker.track(frame)
        for p in tracks:
            assert isinstance(p.track_id, int)
            assert p.track_id >= 0
