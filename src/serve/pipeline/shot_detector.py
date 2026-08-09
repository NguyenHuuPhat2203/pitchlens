"""Shot detector — detects kicks from ball acceleration and direction change."""

from __future__ import annotations

import math

from src.model.event import ShotEvent
from src.model.track import BallTrack


class ShotDetector:
    """Detects shot events by monitoring ball velocity and acceleration.

    A shot is triggered when both conditions hold:
    1. The ball's velocity direction changes by more than
       ``angle_threshold`` degrees compared to the previous frame.
    2. The acceleration magnitude exceeds ``accel_threshold`` AND
       the acceleration direction roughly aligns with the velocity
       (dot product > 0), indicating the ball is being struck
       forward rather than deflected.
    """

    def __init__(
        self,
        angle_threshold: float = 30.0,
        accel_threshold: float = 50.0,
    ) -> None:
        """Create the detector.

        Args:
            angle_threshold: Minimum velocity direction change in
                degrees to consider a shot.
            accel_threshold: Minimum acceleration magnitude (px/s²)
                to consider a shot.
        """
        self._angle_threshold = angle_threshold
        self._accel_threshold = accel_threshold
        self._prev_velocity: tuple[float, float] | None = None
        self._frame_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, ball: BallTrack) -> ShotEvent | None:
        """Advance the detector by one frame.

        Args:
            ball: The current BallTrack from the ball tracker.

        Returns:
            A ShotEvent if a shot was detected on this frame, or None.
        """
        self._frame_counter += 1

        if ball.state != "in_play":
            self._prev_velocity = None
            return None

        vx, vy = ball.velocity

        if self._prev_velocity is None:
            self._prev_velocity = (vx, vy)
            return None

        pvx, pvy = self._prev_velocity
        self._prev_velocity = (vx, vy)

        # Acceleration = change in velocity.
        ax = vx - pvx
        ay = vy - pvy
        accel_mag = math.hypot(ax, ay)

        if accel_mag < self._accel_threshold:
            return None

        # Velocity direction change.
        prev_mag = math.hypot(pvx, pvy)
        curr_mag = math.hypot(vx, vy)
        if prev_mag < 1e-6 or curr_mag < 1e-6:
            return None

        dot = pvx * vx + pvy * vy
        cos_angle = max(-1.0, min(1.0, dot / (prev_mag * curr_mag)))
        angle_delta = math.degrees(math.acos(cos_angle))

        if angle_delta < self._angle_threshold:
            return None

        # Acceleration must roughly align with current velocity
        # (the ball is struck forward, not deflected sideways).
        accel_dot_vel = ax * vx + ay * vy
        if accel_dot_vel <= 0:
            return None

        speed_px_s = curr_mag
        speed_kmh = speed_px_s * 3.6 / 100.0  # rough px→km/h

        return ShotEvent(
            frame_number=self._frame_counter - 1,
            timestamp=None,
            ball_speed_kmh=round(speed_kmh, 1),
        )
