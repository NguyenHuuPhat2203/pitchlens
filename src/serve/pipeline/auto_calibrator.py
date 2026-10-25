"""Automatic pitch calibration from the painted lines: no clicks, no trained model.

A shot is registered by a line-family search: straight markings are grouped into the two
vanishing-point families (parallel to the touchlines, parallel to the goal lines), every ordered
assignment of two lines per family to model lines gives a homography, the candidates are scored
against the line centres, refined, and accepted only if they explain enough of the markings.
The homography is then tracked frame to frame by aligning the projected markings to the line
centres at reduced resolution. A frame whose markings cannot be explained gets no homography.

Measured on data/sample.mp4 (research/calibration/experiments): the search never returned a wrong
homography on the end-on test frames (median error 0.23 m), and tracking kept every one of 425
frames within 1 m (median 0.25 m). Side-on broadcast shots with one family of lines mostly get
no answer.

Every pixel parameter is set for a 1280-pixel-wide frame and scales with the frame width.
"""

from __future__ import annotations

import itertools
import math
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Pitch:
    """Standard markings (IFAB) on a pitch of the given length and width, in metres.

    Frame: x along the length (0..length), y across (0 = far touchline seen from the main camera).
    """

    length: float = 105.0
    width: float = 68.0
    box_depth: float = 16.5
    box_width: float = 40.32
    goal_area_depth: float = 5.5
    goal_area_width: float = 18.32
    circle_radius: float = 9.15
    spot_distance: float = 11.0

    def polylines(self, step: float = 0.5) -> list[np.ndarray]:
        """Every painted marking as a densely sampled polyline."""
        L, W = self.length, self.width
        b0, b1 = (W - self.box_width) / 2, (W + self.box_width) / 2
        g0, g1 = (W - self.goal_area_width) / 2, (W + self.goal_area_width) / 2
        bd, gd = self.box_depth, self.goal_area_depth

        def seg(a, b):
            return np.linspace(a, b, max(2, int(math.dist(a, b) / step) + 1))

        def arc(c, r, t0, t1):
            t = np.linspace(t0, t1, max(8, int(abs(t1 - t0) * r / step) + 1))
            return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], axis=1)

        lines = [seg((0, 0), (L, 0)), seg((L, 0), (L, W)), seg((L, W), (0, W)), seg((0, W), (0, 0)),
                 seg((L / 2, 0), (L / 2, W)), arc((L / 2, W / 2), self.circle_radius, 0, 2 * math.pi)]
        half = math.acos((self.box_depth - self.spot_distance) / self.circle_radius)
        for x0, sgn in ((0.0, 1.0), (L, -1.0)):
            lines += [seg((x0, b0), (x0 + sgn * bd, b0)), seg((x0 + sgn * bd, b0), (x0 + sgn * bd, b1)),
                      seg((x0 + sgn * bd, b1), (x0, b1)),
                      seg((x0, g0), (x0 + sgn * gd, g0)), seg((x0 + sgn * gd, g0), (x0 + sgn * gd, g1)),
                      seg((x0 + sgn * gd, g1), (x0, g1))]
            centre = (x0 + sgn * self.spot_distance, W / 2)  # penalty arc: the part outside the box
            lines.append(arc(centre, self.circle_radius, -half, half) if sgn > 0
                         else arc(centre, self.circle_radius, math.pi - half, math.pi + half))
        return lines

    def line_values(self) -> tuple[np.ndarray, np.ndarray]:
        """x of every x = const marking, y of every y = const marking."""
        L, W = self.length, self.width
        xs = [0, self.goal_area_depth, self.box_depth, L / 2, L - self.box_depth, L - self.goal_area_depth, L]
        ys = [0, (W - self.box_width) / 2, (W - self.goal_area_width) / 2,
              (W + self.goal_area_width) / 2, (W + self.box_width) / 2, W]
        return np.array(xs, float), np.array(ys, float)


def _odd(v: float) -> int:
    return max(3, round(v) | 1)


def field_mask(image: np.ndarray, erode_px: float = 5.0) -> np.ndarray:
    """The playing surface: largest grass-coloured region with holes (lines, players) filled."""
    s = image.shape[1] / 1280
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(25 * s),) * 2))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grass, connectivity=8)
    if n < 2:
        return np.zeros(image.shape[:2], bool)
    field = (lab == 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())).astype(np.uint8)
    flood = np.pad(1 - field, 1, constant_values=1)  # a non-field frame links every border pixel
    cv2.floodFill(flood, None, (0, 0), 2)
    field = (flood[1:-1, 1:-1] != 2).astype(np.uint8)
    k = max(1, round(erode_px * s))
    return cv2.erode(field, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1,) * 2)) > 0


def line_centres(image: np.ndarray) -> np.ndarray:
    """Mask of painted-line centres (1-2 px wide) on the playing surface.

    Thin bright structures (top-hat) on low-saturation pixels, with one dominant orientation (the
    goal net's mesh has two), in long or elongated pieces of limited mean thickness (drops shirts,
    the ball, net and screen logos), thinned by non-maximum suppression across the line.
    """
    s = image.shape[1] / 1280
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    r = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(15 * s),) * 2))
    r = cv2.GaussianBlur(r * (hsv[:, :, 1] < 90), (3, 3), 0)
    strong = (r > 20) & field_mask(image)
    gx, gy = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = (cv2.GaussianBlur(a, (0, 0), max(1.0, 2.0 * s)) for a in (gx * gx, gy * gy, gx * gy))
    strong &= np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6) > 0.6
    n, lab, stats, _ = cv2.connectedComponentsWithStats(strong.astype(np.uint8), connectivity=8)
    w, h, area = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_AREA]
    extent = np.maximum(w, h)
    keep = (extent >= 40 * s) | ((extent >= 15 * s) & (extent * extent / np.maximum(area, 1) >= 8.0))
    # Mean thickness 2 * area / boundary pixels: a line's width, also for connected markings
    # (a box outline), where area / extent would count the whole outline as one thick blob.
    edge = strong & ~cv2.erode(strong.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    keep &= 2 * area / np.maximum(np.bincount(lab[edge], minlength=n), 1) <= 12.0 * s
    keep[0] = False
    strong &= keep[lab]
    pad = np.pad(r, 1, mode="edge")
    hmax = (r >= pad[1:-1, :-2]) & (r >= pad[1:-1, 2:])
    vmax = (r >= pad[:-2, 1:-1]) & (r >= pad[2:, 1:-1])
    return strong & (hmax | vmax)


def _front(H: np.ndarray, w: int, h: int) -> float:
    """Sign of the homogeneous scale, under pitch -> image G = inv(H) with G[2, 2] = 1, of the
    pitch points in front of the camera (homographies carry any sign)."""
    G = np.linalg.inv(H)
    P = H @ np.array([w / 2, h - 1.0, 1.0])  # pitch point seen at the bottom-centre pixel
    return float(np.sign(((G / G[2, 2]) @ (P / P[2]))[2]))


class LineAligner:
    """Moves an image -> pitch homography so the projected markings sit on the line centres."""

    def __init__(self, pitch: Pitch, step: float = 0.5) -> None:
        pts = np.concatenate(pitch.polylines(step))
        self.model = np.c_[pts, np.ones(len(pts))]

    def prepare(self, image: np.ndarray) -> None:
        """Compute the line maps of an image; ``refine``/``inliers`` without an image reuse them."""
        self.h, self.w = image.shape[:2]
        self.s = self.w / 1280
        self.cap = 6.0 * self.s
        self.mask = line_centres(image)
        self.dt = cv2.distanceTransform((~self.mask).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        self.gx = cv2.Sobel(self.dt, cv2.CV_32F, 1, 0, ksize=3) / 8
        self.gy = cv2.Sobel(self.dt, cv2.CV_32F, 0, 1, ksize=3) / 8

    def _project(self, g8: np.ndarray):
        G = np.append(g8, 1.0).reshape(3, 3)  # pitch -> image
        p = self.model @ G.T
        xy = p[:, :2] / p[:, 2:3]
        ok = (p[:, 2] * self.front > 0) & (xy[:, 0] > 2) & (xy[:, 0] < self.w - 3) & (xy[:, 1] > 2) & (xy[:, 1] < self.h - 3)
        return p, xy, ok

    def _sample(self, img: np.ndarray, xy: np.ndarray) -> np.ndarray:
        q = xy.astype(np.float32)
        return cv2.remap(img, q[:, 0][None], q[:, 1][None], cv2.INTER_LINEAR)[0]

    def _residuals(self, g8: np.ndarray) -> np.ndarray:
        _, xy, ok = self._project(g8)
        d = np.full(len(xy), self.cap, np.float32)
        if ok.any():
            d[ok] = self._sample(self.dt, xy[ok])
        return np.minimum(d, self.cap)  # off-screen points never reward leaving the image

    def _jacobian(self, g8: np.ndarray) -> np.ndarray:
        p, xy, ok = self._project(g8)
        J = np.zeros((len(xy), 8))
        if not ok.any():
            return J
        d = self._sample(self.dt, xy[ok])
        active = np.flatnonzero(ok)[d < self.cap]
        if len(active) == 0:
            return J
        dx, dy = self._sample(self.gx, xy[active]), self._sample(self.gy, xy[active])
        X, Y, w = self.model[active, 0], self.model[active, 1], p[active, 2]
        x, y = xy[active, 0], xy[active, 1]
        J[active] = np.stack([dx * X / w, dx * Y / w, dx / w, dy * X / w, dy * Y / w, dy / w,
                              -(dx * x + dy * y) * X / w, -(dx * x + dy * y) * Y / w], axis=1)
        return J

    def inliers(self, H: np.ndarray, image: np.ndarray | None = None) -> float:
        """Share of on-screen model samples within 2 px (at 1280 wide) of a line centre."""
        if image is not None:
            self.prepare(image)
        self.front = _front(H, self.w, self.h)
        G = np.linalg.inv(H)
        _, xy, ok = self._project((G / G[2, 2]).ravel()[:8])
        if ok.sum() == 0:
            return 0.0
        return float((self._sample(self.dt, xy[ok]) < 2.0 * self.s).mean())

    def refine(self, H: np.ndarray, image: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """Aligned homography and its inlier share."""
        if image is not None:
            self.prepare(image)
        self.front = _front(H, self.w, self.h)
        G = np.linalg.inv(H)
        g = (G / G[2, 2]).ravel()[:8]
        for f_scale in (3.0 * self.s, 1.5 * self.s):  # coarse, then fine robust scale
            g = least_squares(self._residuals, g, jac=self._jacobian, loss="soft_l1", f_scale=f_scale,
                              method="trf", x_scale="jac", max_nfev=30).x
        H = np.linalg.inv(np.append(g, 1.0).reshape(3, 3))
        H /= H[2, 2]
        return H, self.inliers(H)


def _detect_lines(mask: np.ndarray, s: float) -> np.ndarray:
    """Straight lines as rows (a, b, c, support), a^2 + b^2 = 1, refitted on mask pixels."""
    segs = cv2.HoughLinesP(mask.astype(np.uint8) * 255, 1, np.pi / 720, max(10, int(30 * s)),
                           minLineLength=40 * s, maxLineGap=12 * s)
    if segs is None:
        return np.zeros((0, 4))
    segs = segs.reshape(-1, 4).astype(np.float64)  # OpenCV 4 returns (N, 1, 4), OpenCV 5 (N, 4)
    segs = segs[np.argsort(-np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]))][:200]
    free = mask.astype(np.uint8).copy()
    h, w = mask.shape
    band = np.zeros_like(free)
    min_support, lines = 60 * s, []
    for x1, y1, x2, y2 in segs:
        mx, my = round((x1 + x2) / 2), round((y1 + y2) / 2)
        if not free[min(max(my, 0), h - 1), min(max(mx, 0), w - 1)]:
            continue  # already explained by an earlier line
        d = np.array([x2 - x1, y2 - y1]) / (np.hypot(x2 - x1, y2 - y1) + 1e-9)
        far = 2.0 * (w + h)
        band[:] = 0
        cv2.line(band, tuple(np.round([x1, y1] - far * d).astype(int)), tuple(np.round([x2, y2] + far * d).astype(int)),
                 1, max(3, round(5 * s)))
        ys, xs = np.nonzero(band & free)
        if len(xs) < min_support:
            continue
        q = np.c_[xs, ys].astype(np.float64)
        c = q.mean(axis=0)
        n = np.linalg.svd(q - c, full_matrices=False)[2][1]
        inl = np.abs((q - c) @ n) < max(0.8, 1.2 * s)
        if inl.sum() < min_support:
            continue
        free[ys[inl], xs[inl]] = 0
        lines.append([n[0], n[1], -n @ c, inl.sum()])
    return np.array(lines) if lines else np.zeros((0, 4))


def _concurrent(lines: np.ndarray, vp: np.ndarray, max_sin: float = 0.02) -> np.ndarray:
    l = lines[:, :3]
    d = np.c_[l[:, 1], -l[:, 0]]
    foot = -l[:, 2:3] * l[:, :2]
    to_vp = np.tile(vp[:2], (len(l), 1)) if abs(vp[2]) < 1e-9 else vp[:2] / vp[2] - foot
    to_vp = to_vp / (np.linalg.norm(to_vp, axis=1, keepdims=True) + 1e-12)
    return np.abs(d[:, 0] * to_vp[:, 1] - d[:, 1] * to_vp[:, 0]) < max_sin


def _families(lines: np.ndarray, per_family: int = 5) -> tuple[np.ndarray, np.ndarray] | None:
    """The two best-supported pencils of concurrent lines (images of the two marking families)."""
    if len(lines) < 4:
        return None
    w = lines[:, 3]
    picked, free = [], np.ones(len(lines), bool)
    for _ in range(2):
        best, best_sup = None, 0.0
        cand = [i for i in np.argsort(-w) if free[i]][:12]
        for i, j in itertools.combinations(cand, 2):
            members = _concurrent(lines, np.cross(lines[i, :3], lines[j, :3])) & free
            if members.sum() >= 2 and w[members].sum() > best_sup:
                best, best_sup = members, w[members].sum()
        if best is None:
            return None
        idx = np.flatnonzero(best)
        picked.append(idx[np.argsort(-w[idx])][:per_family])
        free &= ~best
    return picked[0], picked[1]


def _ordered(fam: np.ndarray, lines: np.ndarray, transversal: np.ndarray) -> np.ndarray:
    """A family's lines sorted by where they cross a line of the other family."""
    t_dir = np.array([transversal[1], -transversal[0]])
    keys = []
    for i in fam:
        p = np.cross(lines[i, :3], transversal[:3])
        keys.append((p[:2] / p[2]) @ t_dir if abs(p[2]) > 1e-12 else np.sign(p[:2] @ t_dir) * 1e9)
    return fam[np.argsort(keys)]


class LineSearch:
    """Registers a single frame from scratch (seconds of CPU), or returns None."""

    def __init__(self, pitch: Pitch, accept_f: float = 0.55, min_matched: int = 80) -> None:
        self.pitch, self.accept_f, self.min_matched = pitch, accept_f, min_matched
        xs, ys = pitch.line_values()
        px, py = list(itertools.permutations(xs, 2)), list(itertools.permutations(ys, 2))
        self.quads_xy = np.array([[(a, c), (a, d), (b, c), (b, d)] for a, b in px for c, d in py], np.float32)
        self.quads_yx = np.array([[(c, a), (d, a), (c, b), (d, b)] for a, b in py for c, d in px], np.float32)
        self.coarse = np.concatenate(pitch.polylines(2.0))
        self.polys = pitch.polylines(0.5)
        self.fine = np.concatenate(self.polys)
        self.aligner = LineAligner(pitch, step=0.25)
        self.last: dict = {}

    @staticmethod
    def _score(G: np.ndarray, pts: np.ndarray, dt: np.ndarray, field: np.ndarray, tol: float):
        """Matched model samples and precision for a batch of pitch -> image homographies."""
        h, w = dt.shape
        p = np.einsum("nij,kj->nki", G, np.c_[pts, np.ones(len(pts))].astype(G.dtype))
        z = p[..., 2]
        xy = p[..., :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[..., None]
        xi, yi = np.round(xy[..., 0]).astype(np.int64), np.round(xy[..., 1]).astype(np.int64)
        on = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        xi, yi = np.clip(xi, 0, w - 1), np.clip(yi, 0, h - 1)
        on &= field[yi, xi]
        matched = (on & (dt[yi, xi] < tol)).sum(axis=1)
        return matched, matched / np.maximum(on.sum(axis=1), 1)

    def _recall(self, G: np.ndarray, mask: np.ndarray, tol: float) -> float:
        """Share of line-centre pixels within tolerance of the projected markings."""
        h, w = mask.shape
        canvas = np.zeros((h, w), np.uint8)
        for poly in self.polys:
            p = np.c_[poly, np.ones(len(poly))] @ G.T
            ok = p[:, 2] * np.sign(p[len(p) // 2, 2]) > 1e-9
            xy = p[:, :2] / np.where(ok, p[:, 2], 1)[:, None]
            ok &= np.all(np.abs(xy) < 4 * max(w, h), axis=1)
            for a, b, oa, ob in zip(xy[:-1], xy[1:], ok[:-1], ok[1:]):
                if oa and ob:
                    cv2.line(canvas, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), 255, 1)
        dist = cv2.distanceTransform(255 - canvas, cv2.DIST_L2, 3)
        return float((dist[mask] < tol).mean()) if mask.any() else 0.0

    def _plausible(self, G: np.ndarray, probe: np.ndarray) -> np.ndarray:
        """Camera above the pitch (no mirror image) and the visible grass on or near the pitch."""
        ok = np.abs(np.linalg.det(G)) > 1e-12
        H = np.zeros_like(G)
        H[ok] = np.linalg.inv(G[ok])
        q = np.einsum("nij,kj->nki", H, np.c_[probe, np.ones(len(probe))].astype(H.dtype))
        z = q[..., 2]
        ok &= np.all(z * np.sign(z[:, :1]) > 1e-12, axis=1)
        xy = q[..., :2] / np.where(np.abs(z) < 1e-12, 1e-12, z)[..., None]
        L, W = self.pitch.length, self.pitch.width
        ok &= np.all((xy[..., 0] > -15) & (xy[..., 0] < L + 15) & (xy[..., 1] > -15) & (xy[..., 1] < W + 15), axis=1)
        c = np.c_[np.nan_to_num(xy.mean(axis=1)), np.ones(len(G))]

        def img(off):
            p = np.einsum("nij,nj->ni", G, c + np.array(off))
            return p[:, :2] / np.where(np.abs(p[:, 2:3]) < 1e-12, 1e-12, p[:, 2:3])

        u, v, o = img([1.0, 0, 0]), img([0, 1.0, 0]), img([0, 0, 0])
        return ok & (((u - o)[:, 0] * (v - o)[:, 1] - (u - o)[:, 1] * (v - o)[:, 0]) > 0)

    def __call__(self, image: np.ndarray) -> np.ndarray | None:
        h, w = image.shape[:2]
        s = w / 1280
        field = field_mask(image, erode_px=3.0)
        self.aligner.prepare(image)  # line maps shared by scoring and every refinement
        mask, dt = self.aligner.mask, self.aligner.dt
        lines = _detect_lines(mask, s)
        fam = _families(lines)
        self.last = {"lines": len(lines)}
        if fam is None:
            return None
        A = _ordered(fam[0], lines, lines[fam[1][0]])
        B = _ordered(fam[1], lines, lines[fam[0][0]])
        step = max(4, round(16 * s))
        fy, fx = np.nonzero(field[::step, ::step])
        probe = np.c_[fx * float(step), fy * float(step)][:: max(1, len(fx) // 40)]
        Gs = []
        for (i, j), (k, m) in itertools.product(itertools.combinations(range(len(A)), 2),
                                                itertools.combinations(range(len(B)), 2)):
            pts = np.array([np.cross(lines[a, :3], lines[b, :3]) for a in A[[i, j]] for b in B[[k, m]]])
            if np.any(np.abs(pts[:, 2]) < 1e-9):
                continue
            img = (pts[:, :2] / pts[:, 2:3]).astype(np.float32)
            if np.any(np.abs(img) > 6 * max(w, h)):
                continue
            for quads in (self.quads_xy, self.quads_yx):  # family A = x-lines, or family A = y-lines
                Gs.extend(cv2.getPerspectiveTransform(q, img) for q in quads)
        if not Gs:
            return None
        G = np.array(Gs)
        G = G[self._plausible(G, probe)]
        if len(G) == 0:
            return None
        matched, prec = np.zeros(len(G), int), np.zeros(len(G))
        for b in range(0, len(G), 8000):
            matched[b:b + 8000], prec[b:b + 8000] = self._score(G[b:b + 8000].astype(np.float32), self.coarse, dt, field, 2.5 * s)
        best, best_f = None, -1.0
        for t in np.argsort(-(matched * prec))[:30]:
            H0 = np.linalg.inv(G[t])
            H, _ = self.aligner.refine(H0 / H0[2, 2])
            Gr = np.linalg.inv(H)
            Gr /= Gr[2, 2]
            if not self._plausible(Gr[None], probe)[0]:
                continue
            m, pr = self._score(Gr[None], self.fine, dt, field, 2.5 * s)
            rc = self._recall(Gr, mask & field, 2.5 * s)
            f = 2 * pr[0] * rc / (pr[0] + rc + 1e-9)
            if f > best_f:
                best, best_f = H, f
                self.last.update(matched=int(m[0]), f=round(float(f), 3))
        if best is None or best_f < self.accept_f or self.last["matched"] < self.min_matched:
            return None
        return best


class AutoCalibrator:
    """Per-frame image -> pitch homography for one video, or None where the pitch is not readable.

    Assumes the standard 105 x 68 m pitch: on another size, distances scale by the size ratio.
    """

    def __init__(self, track_width: int = 640, retry_frames: int = 50, min_inliers: float = 0.3) -> None:
        pitch = Pitch()
        self._search = LineSearch(pitch)
        self._aligner = LineAligner(pitch)
        self._track_width, self._retry, self._min_inliers = track_width, retry_frames, min_inliers
        self.reset()

    def reset(self) -> None:
        """Forget the current shot (new video)."""
        self._H: np.ndarray | None = None
        self._next_search, self._gap = 0, self._retry

    @property
    def is_calibrated(self) -> bool:
        """True when the latest frame has a homography."""
        return self._H is not None

    def update(self, image: np.ndarray, frame_index: int) -> np.ndarray | None:
        """Calibrate this frame: track the previous homography, or search when there is none."""
        if self._H is not None:
            s = self._track_width / image.shape[1]
            small = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else image
            S = np.diag([small.shape[1] / image.shape[1], small.shape[0] / image.shape[0], 1.0])
            H, inliers = self._aligner.refine(self._H @ np.linalg.inv(S), small)
            if inliers >= self._min_inliers:
                H = H @ S
                self._H = H / H[2, 2]
                return self._H
            self._H, self._next_search = None, frame_index  # lost: search again right away
        if frame_index < self._next_search:
            return None
        self._H = self._search(image)
        if self._H is not None:
            self._gap = self._retry
        else:  # back off (up to 16x) so footage without a readable pitch costs little search time
            self._next_search = frame_index + self._gap
            self._gap = min(2 * self._gap, 16 * self._retry)
        return self._H


_worker: AutoCalibrator | None = None  # the child process's calibrator


def _worker_update(image: np.ndarray, frame_index: int, fresh: bool) -> np.ndarray | None:
    global _worker
    if fresh or _worker is None:
        _worker = AutoCalibrator()
    return _worker.update(image, frame_index)


class CalibrationProcess:
    """``AutoCalibrator`` in a child process.

    Its CPU work (much of it Python holding the GIL) then runs beside the pipeline's
    threads instead of between them: in a thread it slowed the whole pipeline by 26 ms
    per frame on data/sample.mp4. Frames are calibrated in submission order.
    """

    def __init__(self) -> None:
        self._pool = ProcessPoolExecutor(max_workers=1)  # the process starts with the first frame
        self._fresh = True

    def reset(self) -> None:
        """The next frame starts a new video."""
        self._fresh = True

    def submit(self, image: np.ndarray, frame_index: int) -> Future:
        """Future of the frame's image -> pitch homography (or None)."""
        fresh, self._fresh = self._fresh, False
        return self._pool.submit(_worker_update, image, frame_index, fresh)
