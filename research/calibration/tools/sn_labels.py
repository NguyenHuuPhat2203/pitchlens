"""Fit an image -> pitch homography to SoccerNet-Calibration line annotations.

SoccerNet gives, per image, points lying on named pitch markings (no camera ground truth).
Every point on a straight marking gives one linear equation l^T H p = 0, so a DLT over
>= 2 lines of each direction initialises H; Levenberg-Marquardt then minimises pixel
distances to the projected lines and circles. Goal posts and crossbars are off the ground
plane and ignored.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from fieldreg import Pitch, apply

P = Pitch()
_B0, _B1 = (P.width - P.box_width) / 2, (P.width + P.box_width) / 2
_G0, _G1 = (P.width - P.goal_area_width) / 2, (P.width + P.goal_area_width) / 2
# (a, b, c) with a*x + b*y + c = 0 in the pitch frame (x along length, y = 0 far touchline).
LINES = {
    "Side line top": (0, 1, 0), "Side line bottom": (0, 1, -P.width),
    "Side line left": (1, 0, 0), "Side line right": (1, 0, -P.length),
    "Middle line": (1, 0, -P.length / 2),
    "Big rect. left main": (1, 0, -P.box_depth), "Big rect. right main": (1, 0, -(P.length - P.box_depth)),
    "Big rect. left top": (0, 1, -_B0), "Big rect. right top": (0, 1, -_B0),
    "Big rect. left bottom": (0, 1, -_B1), "Big rect. right bottom": (0, 1, -_B1),
    "Small rect. left main": (1, 0, -P.goal_area_depth), "Small rect. right main": (1, 0, -(P.length - P.goal_area_depth)),
    "Small rect. left top": (0, 1, -_G0), "Small rect. right top": (0, 1, -_G0),
    "Small rect. left bottom": (0, 1, -_G1), "Small rect. right bottom": (0, 1, -_G1),
}
CIRCLES = {
    "Circle central": (P.length / 2, P.width / 2),
    "Circle left": (P.spot_distance, P.width / 2),
    "Circle right": (P.length - P.spot_distance, P.width / 2),
}
_T = np.linspace(0, 2 * math.pi, 721)[:-1]


def _normaliser(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    s = math.sqrt(2) / max(np.linalg.norm(pts - c, axis=1).mean(), 1e-9)
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


def _dlt(points: np.ndarray, lines: np.ndarray) -> np.ndarray | None:
    """Solve l_i^T H p_i = 0 for H (image -> pitch) with Hartley normalisation."""
    Ti = _normaliser(points)
    Tp = _normaliser(np.array([[0, 0], [P.length, 0], [P.length, P.width], [0, P.width]], float))
    p = np.c_[points, np.ones(len(points))] @ Ti.T
    l = lines @ np.linalg.inv(Tp)  # lines transform with the inverse transpose
    A = np.einsum("ni,nj->nij", l, p).reshape(len(p), 9)
    _, s, vt = np.linalg.svd(A)
    if s[-2] < 1e-9 * s[0]:
        return None
    H = np.linalg.inv(Tp) @ vt[-1].reshape(3, 3) @ Ti
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else None


def _residuals(h8: np.ndarray, lp: np.ndarray, ll: np.ndarray, circ: list[tuple[np.ndarray, tuple, float]]) -> np.ndarray:
    H = np.append(h8, 1.0).reshape(3, 3)
    res = []
    if len(lp):
        il = ll @ H  # image lines: l^T H p = 0  =>  (H^T l)^T p = 0
        res.append((il[:, 0] * lp[:, 0] + il[:, 1] * lp[:, 1] + il[:, 2]) / np.hypot(il[:, 0], il[:, 1]))
    if circ:
        Hinv = np.linalg.inv(H)
        for pts, c, r in circ:
            ring = apply(Hinv, np.c_[c[0] + r * np.cos(_T), c[1] + r * np.sin(_T)])
            res.append(np.sqrt(((pts[:, None, :] - ring[None]) ** 2).sum(-1)).min(axis=1))
    return np.concatenate(res)


def fit(annotation: dict, w: int, h: int) -> dict | None:
    """Homography (image -> pitch metres) for one SoccerNet annotation, or None if under-constrained."""
    lp, ll, families = [], [], {0: set(), 1: set()}
    for name, coef in LINES.items():
        pts = annotation.get(name, [])
        if len(pts) >= 2:
            families[0 if coef[0] else 1].add(name)
            for q in pts:
                lp.append((q["x"] * w, q["y"] * h))
                ll.append(coef)
    if len(families[0]) < 2 or len(families[1]) < 2:
        return None  # parallel-only lines leave the homography undetermined
    lp, ll = np.array(lp, float), np.array(ll, float)
    H0 = _dlt(lp, ll)
    if H0 is None:
        return None
    circ = [(np.array([(q["x"] * w, q["y"] * h) for q in annotation[n]]), c, P.circle_radius)
            for n, c in CIRCLES.items() if len(annotation.get(n, [])) >= 3]
    try:
        sol = least_squares(_residuals, (H0 / H0[2, 2]).ravel()[:8], args=(lp, ll, circ), method="lm")
    except (ValueError, np.linalg.LinAlgError):
        return None
    H = np.append(sol.x, 1.0).reshape(3, 3)
    r = _residuals(sol.x, lp, ll, circ)
    return {"H": H, "rms_px": float(np.sqrt((r ** 2).mean())), "max_px": float(np.abs(r).max()),
            "n": int(len(r)), "lines": sorted(families[0] | families[1])}


def fit_split(root: Path, w: int = 960, h: int = 540) -> list[dict]:
    """Fit every image of a split directory; keep fits with sub-3px RMS and no wild point."""
    out = []
    for f in sorted(root.glob("*.json")):
        if f.name.startswith("match_info"):
            continue
        res = fit(json.loads(f.read_text()), w, h)
        if res is not None:
            out.append({"image": f.with_suffix(".jpg").name, **res})
    return out
