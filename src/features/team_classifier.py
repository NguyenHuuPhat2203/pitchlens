"""Team classifier — Lab colour clustering on pitch-masked jersey crops."""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    """Assigns players to Team A, Team B, or referee by jersey color.

    Takes the median Lab colour of non-grass pixels, clusters with k-means
    (k=3), and labels the two largest clusters as teams and a clearly
    smaller / outlying one as referee.

    Usage:
        clf = TeamClassifier()
        clf.fit(crops)          # once, on a sample of frames
        label = clf.predict(crop)  # per-player, per-frame
    """

    _GRASS_HUE = (35, 85)  # OpenCV hue range of the pitch
    _GRASS_MIN_SAT = 60
    _REFEREE_MAX_RATIO = 0.5  # referee cluster <= this x size of 2nd team

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
    def is_fitted(self) -> bool:
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
        kmeans = KMeans(
            n_clusters=self._n_clusters,
            n_init=10,
            random_state=42,
        ).fit(features)
        counts = np.bincount(kmeans.labels_, minlength=self._n_clusters)
        # Assign only after everything succeeded so a failed fit leaves no model.
        self._cluster_to_label = self._label_clusters(counts, kmeans.cluster_centers_)
        self._kmeans = kmeans

    def predict(self, crop: np.ndarray) -> str:
        """Return the team label for a single player crop.

        Args:
            crop: BGR image patch of a player.

        Returns:
            ``"A"``, ``"B"``, or ``"referee"``.

        Raises:
            RuntimeError: If ``fit()`` has not been called yet.
        """
        if not self.is_fitted:
            raise RuntimeError("TeamClassifier is not fitted; call fit() first")
        features = self._extract_features(crop)
        cluster_id = int(self._kmeans.predict([features])[0])
        return self._cluster_to_label[cluster_id]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_features(self, crop: np.ndarray) -> np.ndarray:
        """Median Lab colour of the non-pitch pixels of a BGR crop.

        Lab keeps lightness, so white and black kits are far apart; green
        (pitch) pixels are masked so grass doesn't dominate the colour.
        """
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        grass = (
            (hsv[..., 0] >= self._GRASS_HUE[0])
            & (hsv[..., 0] <= self._GRASS_HUE[1])
            & (hsv[..., 1] > self._GRASS_MIN_SAT)
        )
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        kept = lab[~grass.reshape(-1)]
        if len(kept) == 0:  # all grass-coloured (e.g. green kit): use everything
            kept = lab
        return np.median(kept, axis=0).astype(np.float64)

    def _label_clusters(self, counts: np.ndarray, centers: np.ndarray) -> dict[int, str]:
        """Map cluster ids to A/B/referee.

        The referee is the smallest cluster when it is clearly smaller than
        the others; otherwise (no clear minority) the cluster whose centroid
        is farthest from every other centroid.
        """
        n = len(counts)
        order = [int(i) for i in np.argsort(counts)[::-1]]
        if n < 3:
            return {c: lbl for c, lbl in zip(order, ("A", "B"))}
        if counts[order[2]] <= self._REFEREE_MAX_RATIO * counts[order[1]]:
            referee = order[2]
        else:
            dist = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
            np.fill_diagonal(dist, np.inf)
            referee = int(np.argmax(dist.min(axis=1)))
        team_a, team_b = [c for c in order if c != referee]
        return {team_a: "A", team_b: "B", referee: "referee"}
