"""Tests for Tracker — OC-SORT wrapper producing PlayerTrack objects."""

import numpy as np
import pytest

from src.model.track import PlayerTrack
from src.serve.pipeline.tracker import Tracker


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


class _FakeBoxes:
    def __init__(self, ids):
        import torch

        n = 2
        self.xyxy = torch.tensor([[0.0, 0.0, 10.0, 20.0]] * n)
        self.conf = torch.tensor([0.9] * n)
        self.cls = torch.tensor([0.0] * n)
        self.id = None if ids is None else torch.tensor(ids, dtype=torch.float32)


class _FakeModel:
    """Stands in for YOLO: returns canned results and exposes a predictor."""

    def __init__(self, ids, trackers=()):
        self._ids = ids
        self.predictor = type("P", (), {"trackers": list(trackers)})()

    def track(self, frame, **kwargs):
        return [type("R", (), {"boxes": _FakeBoxes(self._ids)})()]


def _tracker_with(model) -> Tracker:
    t = object.__new__(Tracker)
    t._model, t._tracker_config, t._device = model, "ocsort.yaml", "cpu"
    t._half, t._conf = False, 0.3
    return t


def test_track_drops_detections_without_track_id() -> None:
    """Untracked detections (no ID yet) never reach the pipeline."""
    tracker = _tracker_with(_FakeModel(ids=None))
    assert tracker.track(_make_synthetic_frame()) == []
    tracks = _tracker_with(_FakeModel(ids=[4, 5])).track(_make_synthetic_frame())
    assert [t.track_id for t in tracks] == [4, 5]


def test_reset_clears_underlying_trackers() -> None:
    """reset() resets every Ultralytics tracker (ids restart between videos)."""
    class Fake:
        resets = 0

        def reset(self):
            self.resets += 1

    fake = Fake()
    _tracker_with(_FakeModel(ids=[1, 2], trackers=[fake])).reset()
    assert fake.resets == 1
    _tracker_with(_FakeModel(ids=[1, 2])).reset()  # no predictor trackers yet
