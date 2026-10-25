"""Shot detector — detects kicks from ball acceleration and direction change."""

from __future__ import annotations

import math
from collections import deque

from src.model.event import ShotEvent
from src.model.track import BallTrack

_WINDOW_S = 0.25    # velocity is compared against this far back
_COOLDOWN_S = 1.0   # one kick fires once, however long the Kalman lags


class ShotDetector:
    """Detects shot events from the ball's velocity change over a window.

    The ball's smoothed velocity now is compared with its velocity
    ``_WINDOW_S`` ago (a kick is spread over several Kalman frames, so
    consecutive frames never differ enough).  A shot fires when, in
    that window:

    1. the velocity direction turned by at least ``angle_threshold``;
    2. the mean acceleration is at least ``accel_threshold`` px/s²;
    3. the acceleration points along the new velocity (struck forward,
       not braked or deflected).

    After a shot the detector is quiet for ``_COOLDOWN_S``.  All
    thresholds are in image pixels; ``ball_speed_kmh`` is left ``None``
    (the pipeline fills it in when a pitch calibration exists).
    """

    def __init__(
        self,
        angle_threshold: float = 30.0,
        accel_threshold: float = 800.0,
    ) -> None:
        """Create the detector.

        Args:
            angle_threshold: Minimum velocity direction change, degrees.
            accel_threshold: Minimum mean acceleration over the window,
                in px/s².
        """
        self._angle_threshold = angle_threshold
        self._accel_threshold = accel_threshold
        # (frame_index, vx, vy) with velocity in px/s, in-play frames only.
        self._history: deque[tuple[int, float, float]] = deque()
        self._last_shot = -math.inf

    def update(
        self, ball: BallTrack, frame_index: int, fps: float = 25.0
    ) -> ShotEvent | None:
        """Advance the detector by one frame.

        Args:
            ball: The current BallTrack (velocity in px/frame).
            frame_index: Real video frame index of this update.
            fps: Video frame rate, converting px/frame to px/s.

        Returns:
            A ShotEvent for ``frame_index`` if a shot fired, else None.
        """
        if ball.state != "in_play":
            self._history.clear()
            return None

        vx, vy = ball.velocity[0] * fps, ball.velocity[1] * fps
        self._history.append((frame_index, vx, vy))

        # Keep the newest sample that is at least one window old at the left.
        window = max(1, round(_WINDOW_S * fps))
        while len(self._history) > 1 and self._history[1][0] <= frame_index - window:
            self._history.popleft()
        ref_frame, pvx, pvy = self._history[0]
        if ref_frame > frame_index - window:
            return None
        if frame_index - self._last_shot < _COOLDOWN_S * fps:
            return None

        ax, ay = vx - pvx, vy - pvy
        accel = math.hypot(ax, ay) * fps / (frame_index - ref_frame)
        if accel < self._accel_threshold:
            return None

        prev_mag, curr_mag = math.hypot(pvx, pvy), math.hypot(vx, vy)
        if prev_mag < 1e-6 or curr_mag < 1e-6:
            return None
        cos_angle = max(-1.0, min(1.0, (pvx * vx + pvy * vy) / (prev_mag * curr_mag)))
        if math.degrees(math.acos(cos_angle)) < self._angle_threshold:
            return None

        if ax * vx + ay * vy <= 0:
            return None

        self._last_shot = frame_index
        return ShotEvent(frame_number=frame_index, timestamp=None, ball_speed_kmh=None)
