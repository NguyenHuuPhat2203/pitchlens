"""Ground truth for frames without any: snap rough line annotations to the painted lines, then fit.

The annotator (a human, or a careful look at a gridded frame) names each visible marking and
gives two or more rough points on it. Points are resampled every few pixels along the rough
polyline and each is moved, along the polyline's normal, to the centre of the white ridge
within +/- `search` px; points with no clear ridge (players, shadows) are dropped. The snapped
points feed the same SoccerNet-style fitter (`sn_labels.fit`), so this GT is built exactly
like SoccerNet's.
"""

from __future__ import annotations

import cv2
import numpy as np

from sn_labels import fit


def ridge_response(image: np.ndarray) -> np.ndarray:
    """White-line strength: luminance top-hat (thin bright structures) on low-saturation pixels."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return cv2.GaussianBlur(tophat * (hsv[:, :, 1] < 90), (3, 3), 0)


def snap(image: np.ndarray, rough: dict[str, list], step: float = 6.0,
         search: int = 7, min_peak: float = 25.0) -> dict[str, list[tuple[float, float]]]:
    """Move rough points onto ridge centres; returns {marking: [(x, y), ...]} in pixels.

    A `None` in a rough polyline breaks it (skip an occluded stretch, e.g. behind the goal net).
    """
    resp = ridge_response(image)
    h, w = resp.shape
    offs = np.arange(-search, search + 0.5, 0.5)
    out = {}
    for name, poly in rough.items():
        pts = []
        for a, b in zip(poly[:-1], poly[1:]):
            if a is None or b is None:
                continue
            a, b = np.asarray(a, float), np.asarray(b, float)
            d = b - a
            normal = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
            for t in np.linspace(0, 1, max(2, int(np.linalg.norm(d) / step))):
                cand = (a + t * d)[None] + offs[:, None] * normal[None]
                ok = (cand[:, 0] >= 0) & (cand[:, 0] < w - 1) & (cand[:, 1] >= 0) & (cand[:, 1] < h - 1)
                if ok.sum() < len(offs) // 2:
                    continue
                vals = cv2.remap(resp, cand[:, 0].astype(np.float32)[None], cand[:, 1].astype(np.float32)[None],
                                 cv2.INTER_LINEAR)[0]
                vals[~ok] = 0
                k = int(vals.argmax())
                if vals[k] < min_peak or k in (0, len(offs) - 1):
                    continue  # no ridge, or the ridge lies outside the search window
                # Sub-pixel centre: intensity-weighted mean around the peak.
                lo, hi = max(0, k - 4), min(len(offs), k + 5)
                wts = np.clip(vals[lo:hi] - 0.5 * vals[k], 0, None)
                shift = float((offs[lo:hi] * wts).sum() / (wts.sum() + 1e-9))
                pts.append(tuple(a + t * d + shift * normal))
        if len(pts) >= 2:
            out[name] = pts
    return out


def fit_frame(image: np.ndarray, rough: dict[str, list]) -> tuple[dict | None, dict]:
    """Snap then fit; returns (fit result with H image->pitch, snapped points)."""
    h, w = image.shape[:2]
    snapped = snap(image, rough)
    ann = {k: [{"x": x / w, "y": y / h} for x, y in v] for k, v in snapped.items()}
    return fit(ann, w, h), snapped
