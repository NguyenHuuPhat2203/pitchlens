"""Tests for Calibrator — pixel-to-pitch homography."""

import numpy as np
import pytest

from src.service.calibrator import Calibrator


class TestCalibrator:
    """Unit tests for the pitch calibrator."""

    def test_is_calibrated_false_by_default(self) -> None:
        """A new calibrator is not yet calibrated."""
        calib = Calibrator()
        assert not calib.is_calibrated

    def test_fit_sets_calibrated_flag(self) -> None:
        """After fit() with valid points, is_calibrated is True."""
        calib = Calibrator()
        image_pts = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        pitch_pts = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=np.float32)
        calib.fit(image_pts, pitch_pts)
        assert calib.is_calibrated

    def test_transform_converts_pixel_to_meters(self) -> None:
        """A known point is transformed to the expected pitch location."""
        calib = Calibrator()
        image_pts = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        pitch_pts = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=np.float32)
        calib.fit(image_pts, pitch_pts)

        # Top-left corner → should map near (0, 0) in pitch space.
        result = calib.transform(np.array([[0.0, 0.0]]))
        assert result.shape == (1, 2)
        assert np.linalg.norm(result[0] - [0, 0]) < 2.0

    def test_transform_raises_before_fit(self) -> None:
        """transform() raises RuntimeError if calibrator is not fitted."""
        calib = Calibrator()
        with pytest.raises(RuntimeError, match="not calibrated"):
            calib.transform(np.array([[0.0, 0.0]]))

    def test_fit_rejects_fewer_than_four_points(self) -> None:
        """fit() requires at least 4 point correspondences."""
        calib = Calibrator()
        with pytest.raises(ValueError, match="At least 4"):
            calib.fit(
                np.array([[0, 0], [100, 0], [100, 50]], dtype=np.float32),
                np.array([[0, 0], [105, 0], [105, 68]], dtype=np.float32),
            )
