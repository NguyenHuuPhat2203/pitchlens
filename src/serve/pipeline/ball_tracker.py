"""Ball tracker — Kalman-filtered trajectory for the football."""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.model.detection import Detection
from src.model.track import BallTrack


class BallTracker:
    """Tracks the football across frames with a Kalman filter.

    Uses a constant-velocity model (4 states: x, y, vx, vy; velocity in
    px/frame).  Each frame the filter predicts, then corrects with the
    candidate detection nearest the prediction, provided it lies within
    ``gate_px``; farther candidates are treated as false positives.
    After ``max_lost_frames`` consecutive misses the state becomes
    ``"lost"`` and the filter is re-initialised from the next detection.

    Attributes:
        max_lost_frames: Consecutive misses before declaring the
            ball lost.  Default 15 (roughly 0.5 s at 30 fps).
        gate_px: Max distance (px) between the predicted position and an
            accepted detection.
    """

    def __init__(self, max_lost_frames: int = 15, gate_px: float = 150.0) -> None:
        """Initialise the Kalman filter.

        Args:
            max_lost_frames: Number of consecutive misses tolerated.
            gate_px: Association gate in pixels.
        """
        self._max_lost = max_lost_frames
        self._gate = gate_px
        self._missed = 0
        self._kf = cv2.KalmanFilter(4, 2)
        self._kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32,
        )
        self._kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32,
        )
        # Velocity noise is high enough to follow a kick within a few
        # frames; position measurements are trusted to about +-1 px.
        self._kf.processNoiseCov = np.diag([0.1, 0.1, 2.0, 2.0]).astype(np.float32)
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
        self._initialised = False

    def update(self, detections: list[Detection]) -> BallTrack:
        """Advance the filter by one frame.

        Args:
            detections: Ball candidates from the current frame (empty
                if the detector did not run or found nothing).

        Returns:
            A BallTrack with the current (possibly predicted) state.
        """
        if not self._initialised:
            if not detections:
                return self._track("lost")
            best = max(detections, key=lambda d: d.confidence)
            cx, cy = self._centroid(best.bbox)
            self._kf.statePost = np.array(
                [[cx], [cy], [0], [0]], dtype=np.float32,
            )
            self._kf.errorCovPost = np.eye(4, dtype=np.float32) * 100
            self._initialised = True
            self._missed = 0
        else:
            # Predict step — advance state using the transition model.
            px, py = self._kf.predict()[:2, 0]
            nearest = min(
                (self._centroid(d.bbox) for d in detections),
                key=lambda c: math.hypot(c[0] - px, c[1] - py),
                default=None,
            )
            if nearest and math.hypot(nearest[0] - px, nearest[1] - py) <= self._gate:
                self._kf.correct(np.array([[nearest[0]], [nearest[1]]], dtype=np.float32))
                self._missed = 0
            else:
                self._missed += 1

        if self._missed > self._max_lost:
            self._initialised = False  # next detection starts a fresh track
            return self._track("lost")
        return self._track("in_play")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _track(self, state: str) -> BallTrack:
        """Snapshot the filter state as a BallTrack."""
        cx, cy, vx, vy = (float(v) for v in self._kf.statePost[:, 0])
        return BallTrack(
            track_id=0,
            bbox=(cx - 5, cy - 5, cx + 5, cy + 5),
            centroid=(cx, cy),
            velocity=(vx, vy),
            state=state,
        )

    @staticmethod
    def _centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        """Return the center (cx, cy) of a bounding box."""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
