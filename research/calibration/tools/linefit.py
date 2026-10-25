"""Training-free registration pieces: white-line centre mask, line alignment, frame-to-frame tracking.

`LineAligner.refine(H, frame)` moves a homography so the projected pitch markings sit on the
centres of the painted lines (robust least squares on a distance transform). `Tracker` predicts
the next frame's homography from ORB camera motion, then refines it on that frame's lines.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares

from fieldreg import Pitch


def field_mask(image: np.ndarray, erode_px: int = 5) -> np.ndarray:
    """The playing surface: largest grass-coloured region, holes (lines, players) filled."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grass, connectivity=8)
    if n < 2:
        return np.zeros(image.shape[:2], bool)
    field = (lab == 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())).astype(np.uint8)
    # Fill interior holes: anything not connected to the image border through non-field pixels.
    flood = np.pad(1 - field, 1, constant_values=1)  # a non-field frame links every border pixel
    cv2.floodFill(flood, None, (0, 0), 2)
    field = (flood[1:-1, 1:-1] != 2).astype(np.uint8)
    return cv2.erode(field, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1,) * 2)) > 0


def line_centres(image: np.ndarray, min_extent: int = 40, min_elongation: float = 8.0) -> np.ndarray:
    """Binary mask of painted-line centres (1-2 px wide) on the playing surface."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    r = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    r = cv2.GaussianBlur(r * (hsv[:, :, 1] < 90), (3, 3), 0)
    strong = (r > 20) & field_mask(image)
    # Orientation coherence of the structure tensor: one dominant direction on a painted line,
    # two crossing directions on the goal net's mesh. Cutting low-coherence pixels separates the
    # net (and line crossings) from the lines it touches before components are judged.
    gx, gy = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = (cv2.GaussianBlur(a, (0, 0), 2.0) for a in (gx * gx, gy * gy, gx * gy))
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    strong &= coherence > 0.6
    # Keep line-like pieces: long, or thin and elongated, and never thick on average. Drops white
    # shirts and the ball (short), the goal net and screen logos (dense: many pixels per length).
    n, lab, stats, _ = cv2.connectedComponentsWithStats(strong.astype(np.uint8), connectivity=8)
    w, h, area = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_AREA]
    extent = np.maximum(w, h)
    keep = (extent >= min_extent) | ((extent >= 15) & (extent * extent / np.maximum(area, 1) >= min_elongation))
    keep &= area / np.maximum(extent, 1) <= 12.0
    keep[0] = False
    strong &= keep[lab]
    # Non-maximum suppression across the line: a centre is a local max horizontally or vertically.
    pad = np.pad(r, 1, mode="edge")
    hmax = (r >= pad[1:-1, :-2]) & (r >= pad[1:-1, 2:])
    vmax = (r >= pad[:-2, 1:-1]) & (r >= pad[2:, 1:-1])
    return strong & (hmax | vmax)


class LineAligner:
    """Refines image -> pitch homographies against a frame's line-centre mask."""

    def __init__(self, pitch: Pitch, step: float = 0.25, cap_px: float = 6.0) -> None:
        pts = np.concatenate(pitch.polylines(step))
        self.model = np.c_[pts, np.ones(len(pts))]  # homogeneous pitch samples on every marking
        self.cap = cap_px

    def _prepare(self, image: np.ndarray) -> None:
        self.h, self.w = image.shape[:2]
        mask = line_centres(image)
        self.dt = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)

    def _front(self, H: np.ndarray) -> float:
        bottom = H @ np.array([self.w / 2, self.h - 1.0, 1.0])
        return float(np.sign((np.linalg.inv(H) @ bottom)[2]))

    def _project(self, g8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        G = np.append(g8, 1.0).reshape(3, 3)  # pitch -> image
        p = self.model @ G.T
        xy = p[:, :2] / p[:, 2:3]
        ok = (p[:, 2] * self.front > 0) & (xy[:, 0] > 2) & (xy[:, 0] < self.w - 3) & (xy[:, 1] > 2) & (xy[:, 1] < self.h - 3)
        return xy, ok

    def _residuals(self, g8: np.ndarray) -> np.ndarray:
        xy, ok = self._project(g8)
        d = np.full(len(xy), self.cap, np.float32)
        if ok.any():
            q = xy[ok].astype(np.float32)
            d[ok] = cv2.remap(self.dt, q[:, 0][None], q[:, 1][None], cv2.INTER_LINEAR)[0]
        # Off-screen points keep a constant residual, so leaving the image is never rewarded.
        return np.minimum(d, self.cap)

    def inliers(self, H: np.ndarray, image: np.ndarray | None = None, tol: float = 2.0) -> float:
        """Share of on-screen model samples within `tol` px of a line centre (alignment quality)."""
        if image is not None:
            self._prepare(image)
        self.front = self._front(H)
        G = np.linalg.inv(H)
        xy, ok = self._project((G / G[2, 2]).ravel()[:8])
        if ok.sum() == 0:
            return 0.0
        q = xy[ok].astype(np.float32)
        d = cv2.remap(self.dt, q[:, 0][None], q[:, 1][None], cv2.INTER_LINEAR)[0]
        return float((d < tol).mean())

    def refine(self, H: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Align H (image -> pitch) to the frame's lines; returns (H, inlier share)."""
        self._prepare(image)
        self.front = self._front(H)
        G = np.linalg.inv(H)
        g = (G / G[2, 2]).ravel()[:8]
        for f_scale in (3.0, 1.5):  # coarse then fine robust scale
            g = least_squares(self._residuals, g, loss="soft_l1", f_scale=f_scale, method="trf",
                              x_scale="jac", max_nfev=60).x
        H = np.linalg.inv(np.append(g, 1.0).reshape(3, 3))
        H /= H[2, 2]
        return H, self.inliers(H)


def camera_motion(src: np.ndarray, dst: np.ndarray, mask: np.ndarray | None = None,
                  features: int = 3000) -> np.ndarray | None:
    """Global homography src -> dst from ORB matches (a camera on a fixed head rotates and
    zooms, so every scene point moves by one homography). `mask` hides static screen overlays."""
    orb = cv2.ORB_create(features)
    k1, d1 = orb.detectAndCompute(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), mask)
    k2, d2 = orb.detectAndCompute(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), mask)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        return None
    a = np.float32([k1[m.queryIdx].pt for m in good])
    b = np.float32([k2[m.trainIdx].pt for m in good])
    G, _ = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
    return G


class Tracker:
    """Carries a homography from frame to frame: ORB camera motion, then line alignment."""

    def __init__(self, pitch: Pitch, mask: np.ndarray | None = None) -> None:
        self.aligner, self.mask = LineAligner(pitch), mask
        self.prev: np.ndarray | None = None
        self.H: np.ndarray | None = None

    def start(self, image: np.ndarray, H: np.ndarray) -> None:
        self.prev, self.H = image, H

    def step(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        G = camera_motion(self.prev, image, self.mask)
        pred = self.H if G is None else self.H @ np.linalg.inv(G)  # pitch = H_prev * G^-1 * x_now
        self.H, score = self.aligner.refine(pred / pred[2, 2], image)
        self.prev = image
        return self.H, score
