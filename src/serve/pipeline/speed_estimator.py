"""Speed estimator — pixel displacement to real-world km/h."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


class SpeedEstimator:
    """Converts displacement between two frames into speed and direction.

    Pixel positions are projected into pitch metres with the calibrator's
    transform (homography or SynLoc camera model), so the displacement is
    in real-world units.  The transform is static: a moving camera needs
    a per-frame transform, which this estimator does not provide.
    """

    def estimate(
        self,
        prev: tuple[float, float],
        curr: tuple[float, float],
        transform: Callable[[np.ndarray], np.ndarray],
        dt: float,
    ) -> tuple[float, float]:
        """Compute speed (km/h) and direction (degrees) between two points.

        Args:
            prev: (x, y) pixel position at the earlier time.
            curr: (x, y) pixel position at the later time.
            transform: Maps an N×2 pixel array to N×2 pitch metres
                (``Calibrator.transform``).
            dt: Seconds elapsed between ``prev`` and ``curr`` (> 0).

        Returns:
            (speed_kmh, direction_deg) tuple.  Direction is 0° for
            rightward movement, 90° for downward (image coords).
        """
        pitch = transform(np.array([prev, curr], dtype=np.float32))
        dx = pitch[1, 0] - pitch[0, 0]
        dy = pitch[1, 1] - pitch[0, 1]
        speed_kmh = math.hypot(dx, dy) / dt * 3.6
        return speed_kmh, math.degrees(math.atan2(dy, dx))
