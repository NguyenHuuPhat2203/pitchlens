"""Ball tracker — Kalman-filtered trajectory for the football."""

from __future__ import annotations

import cv2
import numpy as np

from src.model.detection import Detection
from src.model.track import BallTrack


class BallTracker:
    """Tracks the football across frames with a Kalman filter.

    Uses a constant-velocity model (4 states: x, y, vx, vy).
    When a detection is available the filter corrects; when it is
    missing the filter predicts.  After ``max_lost_frames``
    consecutive misses the state transitions to ``"lost"``.

    Attributes:
        max_lost_frames: Consecutive misses before declaring the
            ball lost.  Default 15 (roughly 0.5 s at 30 fps).
    """

    def __init__(self, max_lost_frames: int = 15) -> None:
        """Initialise the Kalman filter.

        Args:
            max_lost_frames: Number of consecutive misses tolerated.
        """
        self._max_lost = max_lost_frames
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
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * 100
        self._initialised = False

    def update(self, detection: Detection | None) -> BallTrack:
        """Advance the filter by one frame.

        Args:
            detection: A ball Detection from the current frame,
                or None if the ball was not found.

        Returns:
            A BallTrack with the current (possibly predicted) state.
        """
        if not self._initialised:
            if detection is None:
                return BallTrack(
                    track_id=0,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    centroid=(0.0, 0.0),
                    velocity=(0.0, 0.0),
                    state="lost",
                )
            cx, cy = self._centroid(detection.bbox)
            self._kf.statePost = np.array(
                [[cx], [cy], [0], [0]], dtype=np.float32,
            )
            self._initialised = True
            self._missed = 0
        else:
            # Predict step — advance state using the transition model.
            self._kf.predict()
            if detection is not None:
                cx, cy = self._centroid(detection.bbox)
                measurement = np.array([[cx], [cy]], dtype=np.float32)
                self._kf.correct(measurement)
                self._missed = 0
            else:
                self._missed += 1

        state = self._kf.statePost
        cx = float(state[0, 0])
        cy = float(state[1, 0])
        vx = float(state[2, 0])
        vy = float(state[3, 0])
        ball_state = "lost" if self._missed > self._max_lost else "in_play"

        return BallTrack(
            track_id=0,
            bbox=(cx - 5, cy - 5, cx + 5, cy + 5),
            centroid=(cx, cy),
            velocity=(vx, vy),
            state=ball_state,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        """Return the center (cx, cy) of a bounding box."""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
