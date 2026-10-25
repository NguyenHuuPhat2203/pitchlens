"""OCSort must reproduce Ultralytics' OC-SORT exactly: same tracks, IDs, boxes, scores."""

import numpy as np
import pytest
from ultralytics.engine.results import Boxes
from ultralytics.trackers.oc_sort import OCSORT
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from src.serve.pipeline.ocsort import OCSort

W, H = 1280, 720


def _reference() -> OCSORT:
    return OCSORT(IterableSimpleNamespace(**YAML.load(check_yaml("ocsort.yaml"))))


def _scene(seed: int, frames: int = 400, players: int = 24) -> list[np.ndarray]:
    """Broadcast-like detections per frame as (N, 6) float32 [x1, y1, x2, y2, conf, cls].

    Covers what the tracker branches on: crossing and turning players, occlusion gaps either
    side of the 30-frame removal limit, players stopping while hidden (observation-centric
    re-update), leaving and re-entering, boxes clipped at the image edge, duplicate boxes,
    low-score clutter, zero-width boxes, and frames without detections.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform([40, 120], [W - 40, H - 40], size=(players, 2))
    speed = np.where(np.arange(players) % 2 == 0, 4.0, 0.4)[:, None]  # sprinters and walkers
    vel = rng.normal(0, 1, size=(players, 2)) * speed
    wh = rng.uniform([16, 40], [34, 95], size=(players, 2))
    hidden_until = np.zeros(players, dtype=int)
    scene = []
    for f in range(frames):
        vel = 0.9 * vel + rng.normal(0, 0.4, size=vel.shape) * speed
        pos += vel
        pos[:, 0] %= W  # leave on one side, re-enter on the other
        pos[:, 1] = pos[:, 1].clip(60, H - 30)
        start = (hidden_until <= f) & (rng.random(players) < 0.015)
        hidden_until[start] = f + rng.choice([1, 2, 5, 12, 29, 30, 31, 32, 45], size=start.sum())
        seen = hidden_until <= f
        centre = pos[seen] + rng.normal(0, 1.0, size=(seen.sum(), 2))
        size = wh[seen] * rng.uniform(0.95, 1.05, size=(seen.sum(), 2))
        rows = [np.c_[centre - size / 2, centre + size / 2, rng.uniform(0.3, 0.95, seen.sum())]]
        if seen.any() and rng.random() < 0.15:  # second box on the same player
            k = rng.integers(len(rows[0]))
            rows.append(np.r_[rows[0][k, :4] + rng.normal(0, 1.5, 4), 0.4][None])
        if rng.random() < 0.1:  # low-score clutter and a zero-width box
            x, y = rng.uniform([0, 0], [W - 30, H - 60])
            rows.append(np.array([[x, y, x + 20, y + 50, 0.15], [x, y, x, y + 50, 0.9]]))
        det = np.concatenate(rows) if rng.random() > 0.02 else np.zeros((0, 5))
        det[:, [0, 2]] = det[:, [0, 2]].clip(0, W)  # Ultralytics clips boxes to the image
        det[:, [1, 3]] = det[:, [1, 3]].clip(0, H)
        scene.append(np.c_[det, np.zeros(len(det))].astype(np.float32))
    return scene


@pytest.mark.parametrize("seed", range(4))
def test_matches_ultralytics_ocsort_frame_by_frame(seed: int) -> None:
    reference, tracker = _reference(), OCSort()
    for frame, det in enumerate(_scene(seed)):
        expected = np.asarray(reference.update(Boxes(det, (H, W)), None), dtype=np.float32).reshape(-1, 8)
        got = tracker.update(det[:, :4], det[:, 4], det[:, 5])
        assert np.array_equal(got, expected), f"seed {seed}, frame {frame}"


def test_reset_restarts_ids_and_forgets_tracks() -> None:
    scene = _scene(seed=7, frames=60)
    tracker = OCSort()
    first = [tracker.update(d[:, :4], d[:, 4], d[:, 5]) for d in scene]
    tracker.reset()
    second = [tracker.update(d[:, :4], d[:, 4], d[:, 5]) for d in scene]
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
