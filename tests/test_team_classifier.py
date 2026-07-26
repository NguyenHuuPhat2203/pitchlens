"""Tests for TeamClassifier — HSV-based jersey color clustering."""

import cv2
import numpy as np
import pytest

from src.service.team_classifier import TeamClassifier


def _make_jersey_patch(hue: int, sat: int = 200, val: int = 200) -> np.ndarray:
    """Return a 44×44 BGR image of a single HSV color.

    Args:
        hue: Hue value in [0, 179] (OpenCV range).
        sat: Saturation.
        val: Value / brightness.
    """
    hsv = np.full((44, 44, 3), [hue, sat, val], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class TestTeamClassifier:
    """Unit tests for the HSV team classifier."""

    def test_fit_succeeds_with_enough_crops(self) -> None:
        """fit() completes when given enough distinct-color crops."""
        patches = [
            _make_jersey_patch(0),    # red
            _make_jersey_patch(0),
            _make_jersey_patch(60),   # green
            _make_jersey_patch(60),
            _make_jersey_patch(120),  # blue
            _make_jersey_patch(120),
        ]
        clf = TeamClassifier()
        clf.fit(patches)
        assert clf._is_fitted

    def test_same_color_gets_same_label(self) -> None:
        """Two crops of the same color receive the same team label."""
        red1 = _make_jersey_patch(0)
        red2 = _make_jersey_patch(2)
        green = _make_jersey_patch(60)
        blue = _make_jersey_patch(120)

        clf = TeamClassifier()
        clf.fit([red1, red2, green, green, blue, blue, red1, red2])
        label_a = clf.predict(red1)
        label_b = clf.predict(red2)
        assert label_a == label_b, (
            f"Same color got different labels: {label_a} vs {label_b}"
        )

    def test_different_colors_get_different_labels(self) -> None:
        """Two clearly different jersey colors get different labels."""
        red = _make_jersey_patch(0)
        green = _make_jersey_patch(60)
        blue = _make_jersey_patch(120)

        clf = TeamClassifier()
        clf.fit([red, red, green, green, blue, blue])
        label_red = clf.predict(red)
        label_green = clf.predict(green)
        assert label_red != label_green

    def test_predict_raises_before_fit(self) -> None:
        """predict() raises RuntimeError if fit() was never called."""
        clf = TeamClassifier()
        patch = _make_jersey_patch(0)
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(patch)

    def test_fit_rejects_too_few_crops(self) -> None:
        """fit() raises ValueError with fewer crops than clusters."""
        clf = TeamClassifier()
        with pytest.raises(ValueError, match="at least"):
            clf.fit([_make_jersey_patch(0), _make_jersey_patch(60)])
