"""OC-SORT player tracker with batched numpy maths.

A port of Ultralytics' ``OCSORT`` (ultralytics 8.4.182 with ``ocsort.yaml``: no ByteTrack second pass, no
camera-motion compensation, axis-aligned boxes). It keeps Ultralytics' track pools, association order and
edge cases, so the same detections give the same tracks, IDs and boxes. The difference is that it runs the
Kalman filter, IoU and direction costs for all tracks at once, instead of one Python object per track.
``tests/test_ocsort.py`` checks the equivalence against Ultralytics; re-run it before upgrading ultralytics.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH
from ultralytics.trackers.utils.matching import linear_assignment
from ultralytics.utils.metrics import bbox_ioa

TRACKED, LOST, REMOVED = 1, 2, 3
_REMOVED_BUFFER = 1000


class _Track:
    """Per-track state; the maths over many tracks is batched by ``OCSort``."""

    __slots__ = (
        "cls",
        "cov",
        "frame_id",
        "idx",
        "is_activated",
        "last_obs",
        "mean",
        "obs",
        "saved_cov",
        "saved_mean",
        "score",
        "start_frame",
        "state",
        "track_id",
        "velocity",
    )


class _Detections:
    """One frame's associable detections, in the box formats Ultralytics' ``STrack`` derives.

    All conversions keep Ultralytics' dtypes and operation order, so the
    values are bit-identical to the reference tracker's.
    """

    def __init__(self, xywh: np.ndarray, score: np.ndarray, cls: np.ndarray, idx: np.ndarray) -> None:
        x = xywh.astype(np.float64)  # parse_bboxes appends an int index -> float64
        tlwh = x.copy()
        tlwh[:, 0] = x[:, 0] - x[:, 2] / 2
        tlwh[:, 1] = x[:, 1] - x[:, 3] / 2
        self.tlwh = tlwh.astype(np.float32)
        self.xyxy = self.tlwh.copy()
        self.xyxy[:, 2:] += self.xyxy[:, :2]
        self.xyah = self.tlwh.copy()
        self.xyah[:, :2] += self.xyah[:, 2:] / 2
        self.xyah[:, 2] /= self.xyah[:, 3]
        self.center = np.stack(
            [(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2, (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2], axis=1
        )
        self.score = score
        self.cls = cls
        self.idx = idx

    def __len__(self) -> int:
        return len(self.score)

    def subset(self, rows) -> _Detections:
        """Rows ``rows`` of these detections, in that order."""
        sub = object.__new__(_Detections)
        rows = np.asarray(rows, dtype=np.int64)
        for name in ("tlwh", "xyxy", "xyah", "center", "score", "cls", "idx"):
            setattr(sub, name, getattr(self, name)[rows])
        return sub


class OCSort:
    """Ultralytics OC-SORT with ``ocsort.yaml`` defaults, batched.

    ``update`` takes one frame's detections and returns
    ``[x1, y1, x2, y2, track_id, score, cls, idx]`` rows (float32) for the
    confirmed tracks, like Ultralytics' ``OCSORT.update``. Boxes are the
    Kalman-filtered state; ``idx`` indexes the matched input detection.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.25,
        new_track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        fuse_score: bool = True,
        delta_t: int = 3,
        inertia: float = 0.2,
    ) -> None:
        self.track_high_thresh = track_high_thresh
        self.new_track_thresh = new_track_thresh
        self.max_frames_lost = track_buffer
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score
        self.delta_t = delta_t
        self.inertia = inertia
        self.reset()

    def reset(self) -> None:
        """Forget all tracks and restart IDs at 1."""
        self.frame_id = 0
        self._next_id = 1
        self._kf = KalmanFilterXYAH()
        self._tracked: list[_Track] = []
        self._lost: list[_Track] = []
        self._removed_ids: list[int] = []  # Ultralytics' removed_stracks, by ID

    # ------------------------------------------------------------------
    # Frame update (mirrors BYTETracker.update + OCSORT overrides)
    # ------------------------------------------------------------------

    def update(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> np.ndarray:
        """Advance one frame.

        Args:
            xyxy: (N, 4) float32 detection boxes in pixels.
            conf: (N,) float32 confidences.
            cls: (N,) float32 class ids.

        Returns:
            (K, 8) float32 rows ``[x1, y1, x2, y2, track_id, score, cls, idx]``.
        """
        self.frame_id += 1
        activated: list[_Track] = []
        refind: list[_Track] = []
        lost: list[_Track] = []
        removed: list[_Track] = []

        xywh = np.empty_like(xyxy)  # ops.xyxy2xywh
        xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
        xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
        high = (xywh[:, 2] > 0) & (xywh[:, 3] > 0) & (conf >= self.track_high_thresh)
        rows = np.flatnonzero(high)
        dets = _Detections(xywh[rows], conf[rows], cls[rows], rows)
        # Low-score detections only feed the ByteTrack pass, which ocsort.yaml disables.

        unconfirmed = [t for t in self._tracked if not t.is_activated]
        pool = _joint([t for t in self._tracked if t.is_activated], self._lost)
        self._multi_predict(pool)

        # First association: predicted boxes + direction consistency.
        matches, u_track, u_det = linear_assignment(self._dists(pool, dets), thresh=self.match_thresh)
        self._apply(matches, pool, dets, activated, refind)

        # Observation-centric recovery: last observations, Tracked tracks first.
        if len(u_det):
            ocr_dets = list(u_det)
            tracked_u = [i for i in u_track if pool[i].state == TRACKED]
            other_u = [i for i in u_track if pool[i].state != TRACKED]
            u_t1, u_d1 = self._ocr([pool[i] for i in tracked_u], dets, ocr_dets, activated, refind)
            remaining = [ocr_dets[j] for j in u_d1]
            u_t2, u_d2 = self._ocr([pool[i] for i in other_u], dets, remaining, activated, refind)
            u_track = [tracked_u[i] for i in u_t1] + [other_u[i] for i in u_t2]
            u_det = [ocr_dets[u_d1[j]] for j in u_d2]

        for i in u_track:  # no ByteTrack second pass: unmatched Tracked tracks are lost
            if pool[i].state == TRACKED:
                pool[i].state = LOST
                lost.append(pool[i])

        # Unconfirmed tracks against the leftover detections.
        left = dets.subset(u_det)
        if unconfirmed:
            matches, u_unconfirmed, u_new = linear_assignment(self._dists(unconfirmed, left), thresh=0.7)
            self._apply(matches, unconfirmed, left, activated, refind)
            for i in u_unconfirmed:
                unconfirmed[i].state = REMOVED
                removed.append(unconfirmed[i])
        else:
            u_new = range(len(left))

        for i in u_new:
            if left.score[i] >= self.new_track_thresh:
                activated.append(self._activate(left, i))

        for t in self._lost:
            if self.frame_id - t.frame_id > self.max_frames_lost:
                t.state = REMOVED
                removed.append(t)

        self._merge(activated, refind, lost, removed)
        return self._output()

    # ------------------------------------------------------------------
    # Association
    # ------------------------------------------------------------------

    def _dists(self, tracks: list[_Track], dets: _Detections) -> np.ndarray:
        """OCSORT.get_dists: IoU distance, score fusion, plus inertia x direction cost."""
        cost = 1 - self._iou(self._xyxy(tracks).astype(np.float32), dets.xyxy)
        if self.fuse_score and cost.size:
            cost = 1 - (1 - cost) * dets.score[None].repeat(cost.shape[0], axis=0)
        return cost + self.inertia * self._direction_cost(tracks, dets)

    def _direction_cost(self, tracks: list[_Track], dets: _Detections) -> np.ndarray:
        """OCSORT._velocity_direction_cost, all tracks at once.

        The batched matmul rounds each row exactly like Ultralytics' per-track
        ``directions[valid] @ velocity`` (checked bit-for-bit, and by tests/test_ocsort.py).
        """
        cost = np.zeros((len(tracks), len(dets)), dtype=np.float32)
        rows = [i for i, t in enumerate(tracks) if t.velocity is not None and t.last_obs[0] >= 0]
        if cost.size == 0 or not rows:
            return cost
        obs = np.stack([tracks[i].last_obs for i in rows])
        centers = np.stack([(obs[:, 0] + obs[:, 2]) / 2, (obs[:, 1] + obs[:, 3]) / 2], axis=1)
        directions = dets.center[None] - centers[:, None]
        norms = np.sqrt(np.add.reduce(directions * directions, axis=2))  # np.linalg.norm(..., axis=1)
        valid = norms > 1e-6
        directions[valid] /= norms[valid][:, None]
        velocity = np.stack([tracks[i].velocity for i in rows])
        dots = np.clip(np.matmul(directions, velocity[:, :, None])[..., 0], -1.0, 1.0)
        cost[rows] = np.where(valid, np.arccos(dots) / np.pi, 0)
        return cost

    def _ocr(self, tracks, dets, det_rows, activated, refind) -> tuple[list[int], list[int]]:
        """OCSORT._ocr_associate: IoU against last observations; revives lost tracks with ORU."""
        if not tracks or not det_rows:
            return list(range(len(tracks))), list(range(len(det_rows)))
        sub = dets.subset(det_rows)
        boxes = [t.last_obs if t.last_obs[0] >= 0 else xyxy for t, xyxy in zip(tracks, self._xyxy(tracks))]
        cost = 1 - self._iou(np.asarray(boxes, dtype=np.float32), sub.xyxy)
        if self.fuse_score:
            cost = 1 - (1 - cost) * sub.score[None].repeat(cost.shape[0], axis=0)
        matches, u_t, u_d = linear_assignment(cost, thresh=self.match_thresh)
        for it, idet in matches:
            if tracks[it].state != TRACKED:
                self._oru(tracks[it], sub.xyxy[idet])
        self._apply(matches, tracks, sub, activated, refind)
        return list(u_t), list(u_d)

    def _apply(self, matches, tracks: list[_Track], dets: _Detections, activated, refind) -> None:
        """Update matched Tracked tracks, re-activate matched lost ones (one batched Kalman update)."""
        if len(matches) == 0:
            return
        matched = [tracks[it] for it, _ in matches]
        rows = [idet for _, idet in matches]
        for t, i in zip(matched, rows):
            self._record(t, dets.xyxy[i].copy())
        self._kalman_update(matched, dets.xyah[rows])
        for t, i in zip(matched, rows):
            (activated if t.state == TRACKED else refind).append(t)
            t.state = TRACKED
            t.is_activated = True
            t.frame_id = self.frame_id
            t.score, t.cls, t.idx = dets.score[i], dets.cls[i], dets.idx[i]
            t.saved_mean, t.saved_cov = t.mean.copy(), t.cov.copy()
        self._set_velocities(matched)

    # ------------------------------------------------------------------
    # Track lifecycle
    # ------------------------------------------------------------------

    def _activate(self, dets: _Detections, i: int) -> _Track:
        """STrack.activate + OCSortTrack.activate for detection row ``i``."""
        t = _Track()
        t.track_id = self._next_id
        self._next_id += 1
        t.mean, t.cov = self._kf.initiate(dets.xyah[i])
        t.state = TRACKED
        t.is_activated = self.frame_id == 1
        t.frame_id = t.start_frame = self.frame_id
        t.score, t.cls, t.idx = dets.score[i], dets.cls[i], dets.idx[i]
        t.obs = deque(maxlen=self.delta_t + 2)
        t.velocity = None
        self._record(t, self._xyxy([t])[0].astype(np.float32))
        t.saved_mean, t.saved_cov = t.mean.copy(), t.cov.copy()
        return t

    def _record(self, t: _Track, box: np.ndarray) -> None:
        """OCSortTrack._record_observation: keep the last ``delta_t + 2`` observations."""
        t.last_obs = box
        t.obs.append((self.frame_id, box))

    def _set_velocities(self, tracks: list[_Track]) -> None:
        """OCSortTrack._compute_velocity for each track, centres batched.

        Direction: newest observation minus the newest one at least ``delta_t``
        frames older (else the oldest kept). ``sqrt(v.dot(v))`` is how
        ``np.linalg.norm`` computes a vector norm, so rounding is unchanged.
        """
        moving, cur, prev = [], [], []
        for t in tracks:
            if len(t.obs) < 2:
                t.velocity = None
                continue
            frame, box = t.obs[-1]
            moving.append(t)
            cur.append(box)
            prev.append(next((b for f, b in reversed(t.obs) if f < frame - self.delta_t + 1), t.obs[0][1]))
        if not moving:
            return
        cur, prev = np.stack(cur), np.stack(prev)
        direction = np.stack([(cur[:, 0] + cur[:, 2]) / 2, (cur[:, 1] + cur[:, 3]) / 2], axis=1) - np.stack(
            [(prev[:, 0] + prev[:, 2]) / 2, (prev[:, 1] + prev[:, 3]) / 2], axis=1
        )
        for t, d in zip(moving, direction):
            norm = np.sqrt(d.dot(d))
            t.velocity = np.zeros(2, dtype=np.float32) if norm < 1e-6 else (d / norm).astype(np.float32)

    def _oru(self, t: _Track, new_xyxy: np.ndarray) -> None:
        """OCSortTrack.apply_oru: replay the occlusion gap on interpolated boxes."""
        last_frame, last_obs = t.obs[-1]
        gap = self.frame_id - last_frame
        if gap <= 1:
            return
        t.mean, t.cov = t.saved_mean.copy(), t.saved_cov.copy()
        for k in range(1, gap):
            alpha = k / gap
            box = (1 - alpha) * last_obs + alpha * new_xyxy
            xyah = box.copy()  # xyxy2ltwh, then tlwh_to_xyah
            xyah[2] = box[2] - box[0]
            xyah[3] = box[3] - box[1]
            xyah[:2] += xyah[2:] / 2
            xyah[2] /= xyah[3]
            t.mean, t.cov = self._kf.predict(t.mean, t.cov)
            t.mean, t.cov = self._kf.update(t.mean, t.cov, xyah)
        t.mean, t.cov = self._kf.predict(t.mean, t.cov)

    def _merge(self, activated, refind, lost, removed) -> None:
        """stracks.merge_track_pools, including its one-frame removal delay."""
        tracked = [t for t in self._tracked if t.state == TRACKED]
        tracked = _joint(_joint(tracked, activated), refind)
        lost_pool = _sub(self._lost, tracked) + lost
        removed_ids = set(self._removed_ids)  # this frame's removals are not in it yet
        lost_pool = [t for t in lost_pool if t.track_id not in removed_ids]
        self._tracked, self._lost = self._remove_duplicates(tracked, lost_pool)
        self._removed_ids.extend(t.track_id for t in removed)
        del self._removed_ids[:-_REMOVED_BUFFER]

    def _remove_duplicates(self, a: list[_Track], b: list[_Track]) -> tuple[list[_Track], list[_Track]]:
        """stracks.remove_duplicate_stracks (IoU > 0.85): drop the shorter-lived copy."""
        if not a or not b:
            return a, b
        dist = 1 - self._iou(self._xyxy(a).astype(np.float32), self._xyxy(b).astype(np.float32))
        drop_a, drop_b = set(), set()
        for p, q in zip(*np.where(dist < 0.15)):
            if a[p].frame_id - a[p].start_frame > b[q].frame_id - b[q].start_frame:
                drop_b.add(q)
            else:
                drop_a.add(p)
        return [t for i, t in enumerate(a) if i not in drop_a], [t for i, t in enumerate(b) if i not in drop_b]

    def _output(self) -> np.ndarray:
        """``[x1, y1, x2, y2, track_id, score, cls, idx]`` for confirmed Tracked tracks."""
        live = [t for t in self._tracked if t.is_activated]
        if not live:
            return np.zeros((0, 8), dtype=np.float32)
        out = np.empty((len(live), 8), dtype=np.float32)
        out[:, :4] = self._xyxy(live)
        out[:, 4] = [t.track_id for t in live]
        out[:, 5] = [t.score for t in live]
        out[:, 6] = [t.cls for t in live]
        out[:, 7] = [t.idx for t in live]
        return out

    # ------------------------------------------------------------------
    # Batched numerics
    # ------------------------------------------------------------------

    def _multi_predict(self, tracks: list[_Track]) -> None:
        """STrack.multi_predict: lost tracks predict with zero height velocity."""
        if not tracks:
            return
        mean = np.asarray([t.mean for t in tracks])
        cov = np.asarray([t.cov for t in tracks])
        mean[[i for i, t in enumerate(tracks) if t.state != TRACKED], 7] = 0
        mean, cov = self._kf.multi_predict(mean, cov)
        for t, m, c in zip(tracks, mean, cov):
            t.mean, t.cov = m, c

    def _kalman_update(self, tracks: list[_Track], xyah: np.ndarray) -> None:
        """KalmanFilterXYAH.update for many tracks (batched matmuls round like its per-track dots)."""
        mean = np.asarray([t.mean for t in tracks])
        cov = np.asarray([t.cov for t in tracks])
        std = self._kf._std_weight_position * mean[:, 3]
        noise = np.zeros((len(tracks), 4, 4))
        noise[:, [0, 1, 2, 3], [0, 1, 2, 3]] = np.square(np.stack([std, std, np.full_like(std, 1e-1), std], 1))
        projected = cov[:, :4, :4] + noise
        gain = np.linalg.solve(projected, cov[:, :, :4].transpose(0, 2, 1)).transpose(0, 2, 1)
        gain_t = gain.transpose(0, 2, 1)
        mean = mean + np.matmul((xyah - mean[:, :4])[:, None, :], gain_t)[:, 0]
        cov = cov - np.matmul(gain, np.matmul(projected, gain_t))  # np.linalg.multi_dot order
        for t, m, c in zip(tracks, mean, cov):
            t.mean, t.cov = m, c

    @staticmethod
    def _xyxy(tracks: list[_Track]) -> np.ndarray:
        """STrack.xyxy (float64) from each track's Kalman mean."""
        box = np.asarray([t.mean[:4] for t in tracks], dtype=np.float64).reshape(-1, 4)
        box[:, 2] *= box[:, 3]
        box[:, :2] -= box[:, 2:] / 2
        box[:, 2:] += box[:, :2]
        return box

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """matching.iou_distance's IoU (float32)."""
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)), dtype=np.float32)
        return bbox_ioa(np.ascontiguousarray(a), np.ascontiguousarray(b), iou=True)


def _joint(a: list[_Track], b: list[_Track]) -> list[_Track]:
    """stracks.joint_stracks: a, then b's tracks whose ID is not in a."""
    ids = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in ids]


def _sub(a: list[_Track], b: list[_Track]) -> list[_Track]:
    """stracks.sub_stracks: a without b's IDs."""
    ids = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids]
