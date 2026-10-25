"""SynLoc camera-model calibrator: pitch coordinates for a fixed, known camera via sskit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class SynLocCalibrator:
    """Projects pixels onto the pitch with a SynLoc camera model (no homography needed)."""

    def __init__(self) -> None:
        self._camera_matrix: np.ndarray | None = None
        self._undistortion_poly: np.ndarray | None = None

    def load_camera(self, camera_dir: str | Path) -> None:
        """Load ``camera_matrix.npy`` (4×4) and ``lens.json`` from a directory (``sskit.camera``)."""
        from sskit.camera import load_camera as _load

        matrix, poly = _load(Path(camera_dir))
        self._camera_matrix = np.asarray(matrix, dtype=np.float32)
        self._undistortion_poly = np.asarray(poly, dtype=np.float32)

    def transform(self, points: np.ndarray) -> np.ndarray:
        """N×2 pixel coordinates -> N×2 pitch coordinates in metres.

        Raises:
            RuntimeError: If no camera model has been loaded.
        """
        if self._camera_matrix is None:
            raise RuntimeError("No camera model loaded; call load_camera() first")

        from sskit.camera import image_to_ground as _i2g

        result = _i2g(
            torch.as_tensor(self._camera_matrix),
            torch.as_tensor(self._undistortion_poly),
            torch.as_tensor(np.asarray(points), dtype=torch.float32),
        )
        return result.numpy(force=True).reshape(-1, 2)
