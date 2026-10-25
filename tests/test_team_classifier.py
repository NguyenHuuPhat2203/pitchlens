"""Tests for TeamClassifier — HSV-based jersey color clustering."""

import cv2
import numpy as np
import pytest

from src.features.team_classifier import TeamClassifier


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
        assert clf.is_fitted

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

    def test_failed_fit_leaves_classifier_unfitted_and_retryable(self) -> None:
        """A fit() that raised must not leave a half-fitted model."""
        clf = TeamClassifier()
        with pytest.raises(ValueError):
            clf.fit([_make_jersey_patch(0)])
        assert not clf.is_fitted
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(_make_jersey_patch(0))
        clf.fit([_make_jersey_patch(h) for h in (0, 0, 120, 120, 100, 100)])
        assert clf.is_fitted

    def test_white_and_black_kits_get_different_labels(self) -> None:
        """Colourless kits differ only in value; they must not be merged."""
        white = _make_jersey_patch(0, sat=5, val=250)
        black = _make_jersey_patch(0, sat=5, val=20)
        red = _make_jersey_patch(0)
        clf = TeamClassifier()
        clf.fit([white] * 10 + [black] * 10 + [red] * 2)
        assert clf.predict(white) != clf.predict(black)

    def test_grass_pixels_do_not_dominate(self) -> None:
        """A kit mostly surrounded by pitch keeps the kit's label."""
        grass = _make_jersey_patch(60, sat=150, val=140)

        def with_grass(kit: np.ndarray, frac: float) -> np.ndarray:
            img = kit.copy()
            img[:, : int(44 * frac)] = grass[:, : int(44 * frac)]
            return img

        white = _make_jersey_patch(0, sat=5, val=250)
        black = _make_jersey_patch(0, sat=5, val=20)
        clf = TeamClassifier()
        clf.fit(
            [white] * 10 + [black] * 10
            + [_make_jersey_patch(0)] * 2
        )
        assert clf.predict(with_grass(white, 0.7)) == clf.predict(white)
        assert clf.predict(with_grass(black, 0.7)) == clf.predict(black)

    def test_referee_is_small_outlier_cluster(self) -> None:
        """The minority colour is the referee, the two big groups are teams."""
        white = _make_jersey_patch(0, sat=5, val=250)
        black = _make_jersey_patch(0, sat=5, val=20)
        yellow = _make_jersey_patch(25)
        clf = TeamClassifier()
        clf.fit([white] * 10 + [black] * 10 + [yellow] * 2)
        assert clf.predict(yellow) == "referee"
        assert {clf.predict(white), clf.predict(black)} == {"A", "B"}
