"""Team classifier — HSV histogram clustering on jersey crops."""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    """Assigns players to Team A, Team B, or referee by jersey color.

    Extracts hue histograms from player image crops, clusters them
    with k-means (k=3), and labels the two largest clusters as teams
    and the smallest as referee.

    Usage:
        clf = TeamClassifier()
        clf.fit(crops)          # once, on a sample of frames
        label = clf.predict(crop)  # per-player, per-frame
    """

    _HUE_BINS = 32

    def __init__(self, n_clusters: int = 3) -> None:
        """Create the classifier.

        Args:
            n_clusters: Number of k-means clusters (default 3 for
                two teams + referee).
        """
        self._n_clusters = n_clusters
        self._kmeans: KMeans | None = None
        self._cluster_to_label: dict[int, str] | None = None

    @property
    def _is_fitted(self) -> bool:
        """True after a successful ``fit()`` call."""
        return self._kmeans is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, crops: list[np.ndarray]) -> None:
        """Learn cluster centers from player image crops.

        Args:
            crops: List of BGR image patches (player bounding boxes).

        Raises:
            ValueError: If fewer crops are provided than clusters.
        """
        if len(crops) < self._n_clusters:
            raise ValueError(
                f"Need at least {self._n_clusters} crops, got {len(crops)}"
            )
        features = np.stack([self._extract_features(c) for c in crops])
        self._kmeans = KMeans(
            n_clusters=self._n_clusters,
            n_init=10,
            random_state=42,
        ).fit(features)

        # Label mapping: largest clusters → teams, remainder → referee.
        _, counts = np.unique(self._kmeans.labels_, return_counts=True)
        sorted_clusters = np.argsort(counts)[::-1]
        actual_n = len(sorted_clusters)
        self._cluster_to_label = {
            sorted_clusters[0]: "A",
        }
        if actual_n >= 2:
            self._cluster_to_label[sorted_clusters[1]] = "B"
        if actual_n >= 3:
            self._cluster_to_label[sorted_clusters[2]] = "referee"

    def predict(self, crop: np.ndarray) -> str:
        """Return the team label for a single player crop.

        Args:
            crop: BGR image patch of a player.

        Returns:
            ``"A"``, ``"B"``, or ``"referee"``.

        Raises:
            RuntimeError: If ``fit()`` has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError("TeamClassifier is not fitted; call fit() first")
        features = self._extract_features(crop)
        cluster_id = int(self._kmeans.predict([features])[0])
        return self._cluster_to_label[cluster_id]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_features(self, crop: np.ndarray) -> np.ndarray:
        """Extract a normalized hue histogram from a BGR crop."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Focus on the lower third (torso, not face/head).
        torso = hsv[crop.shape[0] // 2:, :, :]
        hist = cv2.calcHist(
            [torso], [0], None, [self._HUE_BINS], [0, 180],
        )
        hist = hist.flatten().astype(np.float64)
        hist /= hist.sum() + 1e-9
        return hist
