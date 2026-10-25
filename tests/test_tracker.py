"""Tests for Tracker: YOLO person detections in, confirmed OC-SORT tracks out."""

import numpy as np
import torch

from src.serve.pipeline.tracker import Tracker

FRAME = np.zeros((360, 640, 3), dtype=np.uint8)


class _FakeYOLO:
    """Stands in for ultralytics.YOLO: predict() returns canned (N, 6) boxes, one list per frame."""

    def __init__(self, frames: list[list[list[float]]]) -> None:
        self._frames = iter(frames)

    def predict(self, frame, **kwargs):
        data = torch.tensor(next(self._frames), dtype=torch.float32).reshape(-1, 6)
        return [type("R", (), {"boxes": type("B", (), {"data": data})()})()]


def _tracker(frames) -> Tracker:
    tracker = Tracker("yolo26s.pt", device="cpu")
    tracker._model = _FakeYOLO(frames)
    return tracker


def _player(x: float, conf: float = 0.9) -> list[float]:
    return [x, 100.0, x + 30.0, 170.0, conf, 0.0]


def test_player_keeps_its_id_and_newcomers_show_from_their_second_frame() -> None:
    tracker = _tracker([[_player(100)], [_player(104), _player(400)], [_player(108), _player(404)]])

    first = tracker.track(FRAME)
    second = tracker.track(FRAME)
    third = tracker.track(FRAME)

    assert [t.track_id for t in first] == [1]  # the first frame confirms immediately
    assert [t.track_id for t in second] == [1]  # the newcomer is not confirmed yet
    assert sorted(t.track_id for t in third) == [1, 2]
    assert abs(third[0].bbox[0] - 108) < 3  # Kalman-filtered box follows the player


def test_boxes_stay_inside_the_frame() -> None:
    tracker = _tracker([[[620.0, 100.0, 660.0, 170.0, 0.9, 0.0]]])
    (track,) = tracker.track(FRAME)
    assert track.bbox[2] == 640.0


def test_reset_restarts_ids() -> None:
    tracker = _tracker([[_player(100)], [_player(400)]])
    tracker.track(FRAME)
    tracker.reset()
    assert [t.track_id for t in tracker.track(FRAME)] == [1]
