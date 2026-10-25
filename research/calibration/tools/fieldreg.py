"""Shared tools for the auto-calibration study: pitch model, ground truth, metrics, overlays.

Pitch frame (same as the product): metres, origin at the far-left corner seen from the
main camera, x along the length (0..L), y along the width (0..W, near touchline at W).
Every homography here maps image pixels -> pitch metres.
"""

from __future__ import annotations

import glob
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "calibration"
YARD = 0.9144
# WC14 / TS-WorldCup homographies map pixels to a 115 x 74 yard template.
WC_LENGTH, WC_WIDTH = 115 * YARD, 74 * YARD


@dataclass(frozen=True)
class Pitch:
    """Standard markings (IFAB) on a pitch of the given length and width."""

    length: float = 105.0
    width: float = 68.0
    box_depth: float = 16.5
    box_width: float = 40.32
    goal_area_depth: float = 5.5
    goal_area_width: float = 18.32
    circle_radius: float = 9.15
    spot_distance: float = 11.0

    def keypoints32(self) -> np.ndarray:
        """Roboflow `sports` 32-keypoint layout (same order), in metres."""
        L, W = self.length, self.width
        b0, b1 = (W - self.box_width) / 2, (W + self.box_width) / 2
        g0, g1 = (W - self.goal_area_width) / 2, (W + self.goal_area_width) / 2
        bd, gd, r, s = self.box_depth, self.goal_area_depth, self.circle_radius, self.spot_distance
        return np.array([
            (0, 0), (0, b0), (0, g0), (0, g1), (0, b1), (0, W),
            (gd, g0), (gd, g1), (s, W / 2),
            (bd, b0), (bd, g0), (bd, g1), (bd, b1),
            (L / 2, 0), (L / 2, W / 2 - r), (L / 2, W / 2 + r), (L / 2, W),
            (L - bd, b0), (L - bd, g0), (L - bd, g1), (L - bd, b1),
            (L - s, W / 2), (L - gd, g0), (L - gd, g1),
            (L, 0), (L, b0), (L, g0), (L, g1), (L, b1), (L, W),
            (L / 2 - r, W / 2), (L / 2 + r, W / 2),
        ], dtype=np.float64)

    def polylines(self, step: float = 0.5) -> list[np.ndarray]:
        """Every painted marking as a densely sampled polyline (metres)."""
        L, W = self.length, self.width
        b0, b1 = (W - self.box_width) / 2, (W + self.box_width) / 2
        g0, g1 = (W - self.goal_area_width) / 2, (W + self.goal_area_width) / 2
        bd, gd = self.box_depth, self.goal_area_depth

        def seg(a, b):
            n = max(2, int(math.dist(a, b) / step) + 1)
            return np.linspace(a, b, n)

        def arc(c, r, t0, t1):
            n = max(8, int(abs(t1 - t0) * r / step) + 1)
            t = np.linspace(t0, t1, n)
            return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], axis=1)

        lines = [seg((0, 0), (L, 0)), seg((L, 0), (L, W)), seg((L, W), (0, W)), seg((0, W), (0, 0)),
                 seg((L / 2, 0), (L / 2, W)), arc((L / 2, W / 2), self.circle_radius, 0, 2 * math.pi)]
        for x0, sgn in ((0.0, 1.0), (L, -1.0)):
            lines += [seg((x0, b0), (x0 + sgn * bd, b0)), seg((x0 + sgn * bd, b0), (x0 + sgn * bd, b1)),
                      seg((x0 + sgn * bd, b1), (x0, b1)),
                      seg((x0, g0), (x0 + sgn * gd, g0)), seg((x0 + sgn * gd, g0), (x0 + sgn * gd, g1)),
                      seg((x0 + sgn * gd, g1), (x0, g1))]
            # Penalty arc: the part of the spot-centred circle outside the box.
            half = math.acos((self.box_depth - self.spot_distance) / self.circle_radius)
            centre = (x0 + sgn * self.spot_distance, W / 2)
            lines.append(arc(centre, self.circle_radius, -half, half) if sgn > 0
                         else arc(centre, self.circle_radius, math.pi - half, math.pi + half))
        return lines


def apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map Nx2 points through a homography (no clipping)."""
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def wc_to_metres(H_wc: np.ndarray) -> np.ndarray:
    """WC template (yards) homography -> pitch metres homography."""
    H = np.diag([YARD, YARD, 1.0]) @ H_wc
    return H / H[2, 2]


def wc14_test() -> list[tuple[Path, np.ndarray]]:
    """WorldCup 2014 test images with image -> metres homographies."""
    items = []
    for f in sorted(glob.glob(str(DATA / "wc14" / "raw" / "test" / "*.jpg")), key=lambda p: int(Path(p).stem)):
        items.append((Path(f), wc_to_metres(np.loadtxt(f.replace(".jpg", ".homographyMatrix")))))
    return items


def tswc(split: str = "test") -> dict[str, list[tuple[Path, np.ndarray]]]:
    """TS-WorldCup sequences: video -> ordered (image, image -> metres homography)."""
    base = DATA / "TS-WorldCup"
    videos = {}
    for v in (base / f"{split}.txt").read_text().split():
        imgs = sorted((base / "Dataset" / "80_95" / v).glob("*.jpg"))
        videos[v] = [(p, wc_to_metres(np.load(base / "Annotations" / "80_95" / v / f"{p.stem}_homography.npy")))
                     for p in imgs]
    return videos


def visible_samples(H_gt: np.ndarray, w: int, h: int, pitch: Pitch, n: int = 2500, seed: int = 0) -> np.ndarray:
    """Pixels sampled uniformly from the image area that shows the pitch (per GT)."""
    rng = np.random.default_rng(seed)
    pts = np.c_[rng.uniform(0, w, 20 * n), rng.uniform(0, h, 20 * n)]
    g = apply(H_gt, pts)
    inside = (g[:, 0] >= 0) & (g[:, 0] <= pitch.length) & (g[:, 1] >= 0) & (g[:, 1] <= pitch.width)
    return pts[inside][:n]


def projection_error(H_pred: np.ndarray, H_gt: np.ndarray, w: int, h: int, pitch: Pitch) -> float:
    """Mean ground distance (m) between pred and GT projections of visible-pitch pixels.

    Unlike KpSFR's script, points whose prediction lands off the pitch are kept, so a
    wrong homography cannot hide its worst errors.
    """
    pts = visible_samples(H_gt, w, h, pitch)
    if len(pts) == 0:
        return float("nan")
    return float(np.linalg.norm(apply(H_pred, pts) - apply(H_gt, pts), axis=1).mean())


def displacement_error(Hp0, Hp1, Hg0, Hg1, w: int, h: int, pitch: Pitch) -> float:
    """Error (m) in the ground motion of fixed pixels between two frames.

    Speed is a difference of positions, so this is the calibration's contribution to
    a speed error: |(P1(x) - P0(x)) - (G1(x) - G0(x))| averaged over visible pixels.
    """
    pts = visible_samples(Hg0, w, h, pitch)
    if len(pts) == 0:
        return float("nan")
    d = (apply(Hp1, pts) - apply(Hp0, pts)) - (apply(Hg1, pts) - apply(Hg0, pts))
    return float(np.linalg.norm(d, axis=1).mean())


def draw(image: np.ndarray, H: np.ndarray, pitch: Pitch, color=(0, 255, 255), thickness: int = 2) -> np.ndarray:
    """Overlay the pitch markings projected with an image -> pitch homography."""
    out = image.copy()
    Hinv = np.linalg.inv(H)
    h, w = image.shape[:2]
    # Homographies are defined up to scale, sign included: take "in front of the camera"
    # from the sign of a pitch point known to be visible (the bottom-centre pixel's).
    front = np.sign((Hinv @ np.r_[apply(H, np.array([[w / 2, h - 1.0]]))[0], 1.0])[2])
    for line in pitch.polylines():
        p = np.c_[line, np.ones(len(line))] @ Hinv.T
        ok = p[:, 2] * front > 1e-9
        xy = p[:, :2] / np.where(ok, p[:, 2], 1)[:, None]
        ok &= (np.abs(xy[:, 0]) < 4 * w) & (np.abs(xy[:, 1]) < 4 * h)
        for a, b, oa, ob in zip(xy[:-1], xy[1:], ok[:-1], ok[1:]):
            if oa and ob:
                cv2.line(out, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), color, thickness, cv2.LINE_AA)
    return out
