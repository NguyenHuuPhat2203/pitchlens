"""SynLoc camera-model calibrator — auto pitch coordinates via sskit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.serve.pipeline.calibrator import Calibrator


class SynLocCalibrator(Calibrator):
    """Extends Calibrator with SynLoc camera-model support.

    When a camera model directory is available, uses ``sskit.camera``
    for direct pixel→pitch projection (no homography needed).
    Falls back to manual click-based homography when no camera model
    is loaded.
    """

    def __init__(self) -> None:
        """Create the calibrator.  Camera model is loaded lazily."""
        super().__init__()
        self._camera_matrix: np.ndarray | None = None
        self._undistortion_poly: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_camera(self, camera_dir: str | Path) -> None:
        """Load a SynLoc camera model from a directory.

        Expects ``camera_matrix.npy`` (4×4) and ``lens.json`` in the
        directory.  Uses ``sskit.camera.load_camera``.

        Args:
            camera_dir: Path to the camera calibration directory.
        """
        from sskit.camera import load_camera as _load

        matrix, poly = _load(Path(camera_dir))
        self._camera_matrix = np.asarray(matrix, dtype=np.float32)
        self._undistortion_poly = np.asarray(poly, dtype=np.float32)

    @property
    def has_camera(self) -> bool:
        """True if a SynLoc camera model has been loaded."""
        return self._camera_matrix is not None

    def image_to_ground(self, points: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to pitch coordinates using the camera model.

        Args:
            points: N×2 array of (x, y) pixel coordinates.

        Returns:
            N×2 array of (x, y) pitch coordinates in metres.

        Raises:
            RuntimeError: If no camera model has been loaded.
        """
        if self._camera_matrix is None:
            raise RuntimeError("No camera model loaded; call load_camera() first")

        pts = torch.as_tensor(points, dtype=torch.float32)
        cam = torch.as_tensor(self._camera_matrix)
        poly = torch.as_tensor(self._undistortion_poly)

        from sskit.camera import image_to_ground as _i2g

        result = _i2g(cam, poly, pts)
        return result.numpy(force=True).reshape(-1, 2)

    def transform(self, points: np.ndarray) -> np.ndarray:
        """Convert pixel→pitch: camera model if available, else homography."""
        if self.has_camera:
            return self.image_to_ground(points)
        return super().transform(points)
