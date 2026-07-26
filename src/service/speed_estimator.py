"""Speed estimator — pixel displacement to real-world km/h."""

from __future__ import annotations

import math

import cv2
import numpy as np


class SpeedEstimator:
    """Converts frame-to-frame displacement into speed and direction.

    Requires a homography matrix from the Calibrator so pixel deltas
    are projected into pitch metres before computing speed.
    """

    def __init__(self, fps: float = 25.0) -> None:
        """Create the estimator.

        Args:
            fps: Video frame rate, used to convert per-frame
                displacement to per-second speed.
        """
        self._fps = float(fps)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        prev: tuple[float, float],
        curr: tuple[float, float],
        homography: np.ndarray,
    ) -> tuple[float, float]:
        """Compute speed (km/h) and direction (degrees) between two points.

        Points are pixel coordinates; they are projected to pitch space
        before computing the displacement.

        Args:
            prev: (x, y) pixel position in the previous frame.
            curr: (x, y) pixel position in the current frame.
            homography: 3×3 perspective transform from Calibrator.

        Returns:
            (speed_kmh, direction_deg) tuple.  Direction is 0° for
            rightward movement, 90° for downward (image coords).
        """
        pts = np.array([[prev, curr]], dtype=np.float32).reshape(-1, 1, 2)
        pitch_pts = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)
        dx = pitch_pts[1, 0] - pitch_pts[0, 0]
        dy = pitch_pts[1, 1] - pitch_pts[0, 1]

        distance_m = math.hypot(dx, dy)
        speed_ms = distance_m * self._fps
        speed_kmh = speed_ms * 3.6

        direction_deg = math.degrees(math.atan2(dy, dx))

        return speed_kmh, direction_deg
