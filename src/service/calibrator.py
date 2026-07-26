"""Pitch calibrator — computes pixel-to-pitch homography."""

from __future__ import annotations

import cv2
import numpy as np


class Calibrator:
    """Maps pixel coordinates to real-world pitch coordinates (meters).

    Computes a perspective transform (homography) from at least four
    point correspondences between the image and the pitch template.

    FIFA regulation pitch: 105 × 68 metres (configurable).
    """

    PITCH_LENGTH = 105.0
    PITCH_WIDTH = 68.0

    def __init__(self) -> None:
        """Create an uncalibrated calibrator."""
        self._homography: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        """True once ``fit()`` has succeeded."""
        return self._homography is not None

    @property
    def homography(self) -> np.ndarray:
        """The 3×3 homography matrix, or raises if not calibrated."""
        if self._homography is None:
            raise RuntimeError("Calibrator is not calibrated; call fit() first")
        return self._homography

    def fit(
        self,
        image_points: np.ndarray,
        pitch_points: np.ndarray,
    ) -> None:
        """Compute the homography from N ≥ 4 point correspondences.

        Args:
            image_points: N×2 array of (x, y) pixel coordinates.
            pitch_points: N×2 array of (x, y) pitch coordinates in metres.

        Raises:
            ValueError: If fewer than 4 points are provided.
        """
        n = len(image_points)
        if n < 4:
            raise ValueError(f"At least 4 points required, got {n}")

        self._homography, _ = cv2.findHomography(
            image_points.astype(np.float32),
            pitch_points.astype(np.float32),
            cv2.RANSAC,
            5.0,
        )

    def transform(self, points: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to pitch coordinates.

        Args:
            points: N×2 array of (x, y) pixel coordinates.

        Returns:
            N×2 array of (x, y) pitch coordinates in metres.

        Raises:
            RuntimeError: If ``fit()`` has not been called.
        """
        if self._homography is None:
            raise RuntimeError("Calibrator is not calibrated; call fit() first")

        pts = points.astype(np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self._homography)
        return transformed.reshape(-1, 2)
