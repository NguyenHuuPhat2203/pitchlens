"""Training-free initialisation: find the image -> pitch homography from straight painted lines.

1. Straight lines: Hough on the line-centre mask, each refitted by total least squares on the
   mask pixels in a band around it.
2. Two families by vanishing point: images of x = const and y = const markings.
3. Hypotheses: two of the (at most five strongest) lines of each family assigned, in order, to
   two model lines of each orientation; the four intersections give a homography. Mirror images
   and views that put the visible grass off the pitch are discarded.
4. Vectorised chamfer score on the distance transform; the best candidates are refined by line
   alignment, and the winner must explain enough of the markings, otherwise: no answer.
"""

from __future__ import annotations

import itertools

import cv2
import numpy as np

from fieldreg import Pitch
from linefit import LineAligner, field_mask, line_centres

PER_FAMILY = 5


def _model_values(p: Pitch) -> tuple[np.ndarray, np.ndarray]:
    """x of every x = const marking, y of every y = const marking (pitch metres)."""
    xs = [0, p.goal_area_depth, p.box_depth, p.length / 2, p.length - p.box_depth, p.length - p.goal_area_depth, p.length]
    ys = [0, (p.width - p.box_width) / 2, (p.width - p.goal_area_width) / 2,
          (p.width + p.goal_area_width) / 2, (p.width + p.box_width) / 2, p.width]
    return np.array(xs, float), np.array(ys, float)


def detect_lines(mask: np.ndarray, min_support: int = 60, max_segments: int = 200) -> np.ndarray:
    """Straight lines as rows (a, b, c, support) with a*x + b*y + c = 0 and a^2 + b^2 = 1."""
    segs = cv2.HoughLinesP(mask.astype(np.uint8) * 255, 1, np.pi / 720, 30, minLineLength=40, maxLineGap=12)
    if segs is None:
        return np.zeros((0, 4))
    segs = segs.reshape(-1, 4).astype(np.float64)  # OpenCV 4 returns (N, 1, 4), OpenCV 5 (N, 4)
    segs = segs[np.argsort(-np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]))][:max_segments]
    free = mask.astype(np.uint8).copy()
    h, w = mask.shape
    band = np.zeros_like(free)
    lines = []
    for x1, y1, x2, y2 in segs:
        mx, my = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
        if not free[min(max(my, 0), h - 1), min(max(mx, 0), w - 1)]:
            continue  # already explained by an earlier line
        # Pixels of the mask in a 5 px band along the whole (extended) line through the segment.
        d = np.array([x2 - x1, y2 - y1]) / (np.hypot(x2 - x1, y2 - y1) + 1e-9)
        far = 2.0 * (w + h)
        band[:] = 0
        cv2.line(band, tuple(np.round([x1, y1] - far * d).astype(int)), tuple(np.round([x2, y2] + far * d).astype(int)), 1, 5)
        ys, xs = np.nonzero(band & free)
        if len(xs) < min_support:
            continue
        q = np.c_[xs, ys].astype(np.float64)
        c = q.mean(axis=0)
        n = np.linalg.svd(q - c, full_matrices=False)[2][1]
        inl = np.abs((q - c) @ n) < 1.2
        if inl.sum() < min_support:
            continue
        free[ys[inl], xs[inl]] = 0
        lines.append([n[0], n[1], -n @ c, inl.sum()])
    return np.array(lines) if lines else np.zeros((0, 4))


def _concurrent(lines: np.ndarray, vp: np.ndarray, max_sin: float = 0.02) -> np.ndarray:
    """Lines passing through vanishing point vp (homogeneous), by angle at the line's foot."""
    l = lines[:, :3]
    d = np.c_[l[:, 1], -l[:, 0]]
    foot = -l[:, 2:3] * l[:, :2]
    to_vp = np.tile(vp[:2], (len(l), 1)) if abs(vp[2]) < 1e-9 else vp[:2] / vp[2] - foot
    to_vp = to_vp / (np.linalg.norm(to_vp, axis=1, keepdims=True) + 1e-12)
    return np.abs(d[:, 0] * to_vp[:, 1] - d[:, 1] * to_vp[:, 0]) < max_sin


def families(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Indices of the two best-supported pencils of concurrent lines (strongest first)."""
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
        picked.append(idx[np.argsort(-w[idx])][:PER_FAMILY])
        free &= ~best
    return picked[0], picked[1]


def _ordered(fam: np.ndarray, lines: np.ndarray, transversal: np.ndarray) -> np.ndarray:
    """Sort a family's lines by where they cross a line of the other family."""
    t_dir = np.array([transversal[1], -transversal[0]])
    keys = []
    for i in fam:
        p = np.cross(lines[i, :3], transversal[:3])
        keys.append((p[:2] / p[2]) @ t_dir if abs(p[2]) > 1e-12 else np.sign(p[:2] @ t_dir) * 1e9)
    return fam[np.argsort(keys)]


class LineSearch:
    """Initialise a pitch homography on a single frame, or return None."""

    def __init__(self, pitch: Pitch, accept_f: float = 0.55, min_matched: int = 80) -> None:
        self.pitch, self.accept_f, self.min_matched = pitch, accept_f, min_matched
        xs, ys = _model_values(pitch)
        # Every ordered pair of x-lines with every ordered pair of y-lines: 4 model points each.
        px, py = list(itertools.permutations(xs, 2)), list(itertools.permutations(ys, 2))
        self.quads_xy = np.array([[(a, c), (a, d), (b, c), (b, d)] for a, b in px for c, d in py], np.float32)
        self.quads_yx = np.array([[(c, a), (d, a), (c, b), (d, b)] for a, b in py for c, d in px], np.float32)
        self.coarse = np.concatenate(pitch.polylines(2.0))
        self.polys = pitch.polylines(0.5)
        self.fine = np.concatenate(self.polys)
        self.aligner = LineAligner(pitch)
        self.last: dict = {}

    def _recall(self, G: np.ndarray, mask: np.ndarray) -> float:
        """Share of the frame's line-centre pixels lying within 2.5 px of the projected markings."""
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
        return float((dist[mask] < 2.5).mean()) if mask.any() else 0.0

    @staticmethod
    def _score(G: np.ndarray, pts: np.ndarray, dt: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Matched model samples and precision for a batch of pitch -> image homographies."""
        h, w = dt.shape
        p = np.einsum("nij,kj->nki", G, np.c_[pts, np.ones(len(pts))].astype(G.dtype))
        z = p[..., 2]
        xy = p[..., :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[..., None]
        xi, yi = np.round(xy[..., 0]).astype(np.int64), np.round(xy[..., 1]).astype(np.int64)
        on = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        xi, yi = np.clip(xi, 0, w - 1), np.clip(yi, 0, h - 1)
        on &= field[yi, xi]
        matched = (on & (dt[yi, xi] < 2.5)).sum(axis=1)
        return matched, matched / np.maximum(on.sum(axis=1), 1)

    def __call__(self, image: np.ndarray) -> np.ndarray | None:
        h, w = image.shape[:2]
        field = field_mask(image, erode_px=3)
        mask = line_centres(image)
        dt = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
        lines = detect_lines(mask)
        fam = families(lines)
        self.last = {"lines": len(lines), "families": None if fam is None else (len(fam[0]), len(fam[1]))}
        if fam is None:
            return None
        A = _ordered(fam[0], lines, lines[fam[1][0]])
        B = _ordered(fam[1], lines, lines[fam[0][0]])
        fy, fx = np.nonzero(field[::16, ::16])
        probe = np.c_[fx * 16.0, fy * 16.0][:: max(1, len(fx) // 40)]  # grass pixels for sanity checks
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
        self.last["hypotheses"] = len(Gs)
        self.last["plausible"] = len(G)
        if len(G) == 0:
            return None
        matched, prec = np.zeros(len(G), int), np.zeros(len(G))
        for s in range(0, len(G), 8000):
            matched[s:s + 8000], prec[s:s + 8000] = self._score(G[s:s + 8000].astype(np.float32), self.coarse, dt, field)
        best, best_f = None, -1.0
        for t in np.argsort(-(matched * prec))[:30]:
            H0 = np.linalg.inv(G[t])
            H, _ = self.aligner.refine(H0 / H0[2, 2], image)
            Gr = np.linalg.inv(H)
            Gr /= Gr[2, 2]
            if not self._plausible(Gr[None], probe)[0]:
                continue
            m, pr = self._score(Gr[None], self.fine, dt, field)
            rc = self._recall(Gr, mask & field)
            f = 2 * pr[0] * rc / (pr[0] + rc + 1e-9)
            if f > best_f:
                best, best_f = H, f
                self.last.update(matched=int(m[0]), precision=round(float(pr[0]), 3), recall=round(rc, 3),
                                 f=round(float(f), 3))
        if best is None or best_f < self.accept_f or self.last["matched"] < self.min_matched:
            return None
        return best

    def _plausible(self, G: np.ndarray, probe: np.ndarray) -> np.ndarray:
        """Camera above the pitch (no mirror image) and the visible grass maps onto or near the pitch."""
        det = np.linalg.det(G)
        ok = np.abs(det) > 1e-12
        H = np.zeros_like(G)
        H[ok] = np.linalg.inv(G[ok])
        q = np.einsum("nij,kj->nki", H, np.c_[probe, np.ones(len(probe))].astype(H.dtype))
        z = q[..., 2]
        ok &= np.all(z * np.sign(z[:, :1]) > 1e-12, axis=1)  # every grass pixel in front of the camera
        xy = q[..., :2] / np.where(np.abs(z) < 1e-12, 1e-12, z)[..., None]
        L, W = self.pitch.length, self.pitch.width
        ok &= np.all((xy[..., 0] > -15) & (xy[..., 0] < L + 15) & (xy[..., 1] > -15) & (xy[..., 1] < W + 15), axis=1)
        # Handedness: +x then +y on the pitch must turn the way they do for a real camera.
        c = np.c_[np.nan_to_num(xy.mean(axis=1)), np.ones(len(G))]

        def img(off):
            p = np.einsum("nij,nj->ni", G, c + np.array(off))
            return p[:, :2] / np.where(np.abs(p[:, 2:3]) < 1e-12, 1e-12, p[:, 2:3])

        u, v, o = img([1.0, 0, 0]), img([0, 1.0, 0]), img([0, 0, 0])
        ok &= ((u - o)[:, 0] * (v - o)[:, 1] - (u - o)[:, 1] * (v - o)[:, 0]) > 0
        return ok
