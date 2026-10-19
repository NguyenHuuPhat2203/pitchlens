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
        speed, _ = SpeedEstimator().estimate(
            (50.0, 50.0), (50.0, 50.0), calibrated.transform, dt=0.04
        )
        assert speed == pytest.approx(0.0, abs=0.01)

    def test_known_displacement_gives_expected_speed(
        self, calibrated: Calibrator,
    ) -> None:
        """1 m in 1/25 s = 25 m/s = 90 km/h."""
        speed, _ = SpeedEstimator().estimate(
            (50.0, 50.0), (51.0, 50.0), calibrated.transform, dt=1 / 25
        )
        assert speed == pytest.approx(90.0, abs=1.0)

    def test_speed_divides_by_real_elapsed_time(
        self, calibrated: Calibrator,
    ) -> None:
        """Same 1 m displacement over 11 frames at 25 fps is 11x slower."""
        speed, _ = SpeedEstimator().estimate(
            (50.0, 50.0), (51.0, 50.0), calibrated.transform, dt=11 / 25
        )
        assert speed == pytest.approx(90.0 / 11, abs=0.1)

    def test_direction_is_degrees(self, calibrated: Calibrator) -> None:
        """Direction is returned in degrees (0 = right, 90 = down)."""
        _, direction = SpeedEstimator().estimate(
            (50.0, 50.0), (60.0, 50.0), calibrated.transform, dt=0.04
        )
        assert direction == pytest.approx(0.0, abs=10.0)
