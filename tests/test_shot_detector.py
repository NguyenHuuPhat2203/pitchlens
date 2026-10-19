"""Tests for ShotDetector — velocity-change shot event detection."""

import math

import numpy as np
import pytest

from src.model.detection import Detection
from src.model.track import BallTrack
from src.serve.pipeline.ball_tracker import BallTracker
from src.serve.pipeline.shot_detector import ShotDetector

FPS = 25.0


def _ball(velocity, state="in_play") -> BallTrack:
    return BallTrack(0, (0, 0, 10, 10), (5.0, 5.0), velocity, state)


def _simulate(kick_px_per_frame, seed, kick_frame=40, turn_deg=45.0, frames=90):
    """Ball dribbled at 4 px/frame, optionally kicked; detected every other
    frame with +-1 px noise.  Returns the frame indices where shots fired."""
    rng = np.random.default_rng(seed)
    tracker, detector = BallTracker(), ShotDetector()
    x, y, speed, heading = 100.0, 300.0, 4.0, 0.0
    shots = []
    for f in range(frames):
        if kick_px_per_frame and f == kick_frame:
            speed, heading = kick_px_per_frame, math.radians(turn_deg)
        x += speed * math.cos(heading)
        y += speed * math.sin(heading)
        dets = []
        if f % 2 == 0:
            nx, ny = x + rng.uniform(-1, 1), y + rng.uniform(-1, 1)
            dets = [Detection((nx - 5, ny - 5, nx + 5, ny + 5), 0, 0.9)]
        if detector.update(tracker.update(dets), f, FPS):
            shots.append(f)
    return shots


@pytest.mark.parametrize("kick", [20.0, 40.0])
@pytest.mark.parametrize("seed", range(5))
def test_kick_from_dribble_fires_exactly_once_near_the_kick(kick, seed) -> None:
    shots = _simulate(kick, seed)
    assert len(shots) == 1
    assert 40 <= shots[0] <= 48


@pytest.mark.parametrize("seed", range(10))
def test_steady_dribble_with_noise_never_fires(seed) -> None:
    assert _simulate(None, seed) == []


def test_event_carries_real_frame_index_and_no_fake_speed() -> None:
    detector = ShotDetector()
    event = None
    for f in range(100, 110):
        v = (2.0, 0.0) if f < 105 else (0.0, 30.0)
        event = detector.update(_ball(v), f, FPS) or event
    assert event is not None
    assert event.frame_number >= 105
    assert event.ball_speed_kmh is None


def test_threshold_is_in_px_per_second_squared() -> None:
    """The same px/frame jump is a shot at 25 fps but not at 100 fps with the
    window fixed in seconds: 100 fps frames are 4x closer in time."""
    def fires(fps):
        d = ShotDetector(angle_threshold=20.0, accel_threshold=2000.0)
        out = None
        for f in range(40):
            v = (2.0, 0.0) if f < 20 else (0.0, 30.0 * 25.0 / fps)
            out = d.update(_ball(v), f, fps) or out
        return out

    assert fires(25.0) is not None
    assert fires(100.0) is not None  # same physical px/s velocities -> still fires


def test_no_shot_when_ball_is_lost() -> None:
    detector = ShotDetector()
    for f in range(10):
        detector.update(_ball((5.0, 0.0)), f, FPS)
    assert detector.update(_ball((0.0, 0.0), state="lost"), 10, FPS) is None


def test_first_update_never_fires() -> None:
    assert ShotDetector().update(_ball((50.0, 0.0)), 0, FPS) is None
