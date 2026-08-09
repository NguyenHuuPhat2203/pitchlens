"""Tests for SpeedEstimator — pixel displacement to real-world km/h."""

import numpy as np
import pytest

from src.serve.pipeline.calibrator import Calibrator
from src.serve.pipeline.speed_estimator import SpeedEstimator


class TestSpeedEstimator:
    """Unit tests for the speed estimator."""

    @pytest.fixture
    def calibrated(self) -> Calibrator:
        """A calibrator with a simple pixel→meter mapping (1 px = 1 m)."""
        calib = Calibrator()
        img = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        pitch = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        calib.fit(img, pitch)
        return calib

    def test_zero_displacement_gives_zero_speed(self, calibrated: Calibrator) -> None:
        """A stationary object has speed 0."""
        estimator = SpeedEstimator(fps=25.0)
        speed, direction = estimator.estimate(
            prev=(50.0, 50.0),
            curr=(50.0, 50.0),
            homography=calibrated.homography,
        )
        assert speed == pytest.approx(0.0, abs=0.01)

    def test_known_displacement_gives_expected_speed(
        self, calibrated: Calibrator,
    ) -> None:
        """Moving 1 meter per frame at 25 fps = 25 m/s = 90 km/h."""
        estimator = SpeedEstimator(fps=25.0)
        speed, direction = estimator.estimate(
            prev=(50.0, 50.0),
            curr=(51.0, 50.0),  # 1 pixel right
            homography=calibrated.homography,
        )
        # 1 px/frame × 25 fps = 25 px/s = 25 m/s = 90 km/h
        assert speed == pytest.approx(90.0, abs=1.0)

    def test_direction_is_degrees(self, calibrated: Calibrator) -> None:
        """Direction is returned in degrees (0 = right, 90 = down)."""
        estimator = SpeedEstimator(fps=25.0)
        _, direction = estimator.estimate(
            prev=(50.0, 50.0),
            curr=(60.0, 50.0),  # pure right
            homography=calibrated.homography,
        )
        assert direction == pytest.approx(0.0, abs=10.0)

    def test_estimator_uses_default_fps(self) -> None:
        """Default FPS is 25 if not specified."""
        estimator = SpeedEstimator()
        assert estimator._fps == 25.0
