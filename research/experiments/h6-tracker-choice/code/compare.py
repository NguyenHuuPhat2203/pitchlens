"""H6b: OC-SORT vs ByteTrack vs FastTrack — IDF1/MOTA/IDsw on SportsMOT + speed."""

import src  # noqa: F401  # project-local caches

import json
import sys
import time
from pathlib import Path
from statistics import median

import cv2
import motmetrics as mm
import numpy as np

from src.backbone.gpu_utils import fp16_model_path
from src.serve.pipeline.tracker import Tracker

OUT = Path(sys.argv[1])
SEQ = Path(".cache/sportsmot/dataset/train/v_gQNyhv8y0QY_c013")
MODEL = fp16_model_path("models/player_detector.onnx")

images = sorted((SEQ / "img1").glob("*.jpg"))
gt = np.loadtxt(SEQ / "gt" / "gt.txt", delimiter=",")
gt_by_frame = {int(f): gt[gt[:, 0] == f] for f in np.unique(gt[:, 0])}
frames = [cv2.imread(str(p)) for p in images]

cap = cv2.VideoCapture("data/sample.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
sample = [cap.read()[1] for _ in range(300)]
cap.release()


def iou_distance(a: np.ndarray, b: np.ndarray, min_iou: float = 0.5) -> np.ndarray:
    """1 - IoU for xywh boxes, NaN below ``min_iou`` (motmetrics convention;
    its own iou_matrix uses np.asfarray, removed in NumPy 2)."""
    if len(a) == 0 or len(b) == 0:
        return np.empty((len(a), len(b)))
    a1, a2 = a[:, None, :2], a[:, None, :2] + a[:, None, 2:]
    b1, b2 = b[None, :, :2], b[None, :, :2] + b[None, :, 2:]
    wh = np.clip(np.minimum(a2, b2) - np.maximum(a1, b1), 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    iou = inter / union
    return np.where(iou >= min_iou, 1 - iou, np.nan)


def run(cfg: str) -> dict:
    tracker = Tracker(MODEL, cfg, device="0", half=True, conf=0.3, tensorrt=True)
    acc = mm.MOTAccumulator(auto_id=True)
    times = []
    for i, img in enumerate(frames, start=1):
        t = time.perf_counter()
        tracks = tracker.track(img)
        times.append((time.perf_counter() - t) * 1000)
        g = gt_by_frame.get(i, np.zeros((0, 6)))
        hyp = np.array([[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in (t.bbox for t in tracks)])
        dist = iou_distance(g[:, 2:6], hyp.reshape(-1, 4))
        acc.update(g[:, 1].astype(int).tolist(), [t.track_id for t in tracks], dist)
    summary = mm.metrics.create().compute(
        acc, metrics=["idf1", "mota", "num_switches", "num_false_positives", "num_misses"]
    )
    tracker.reset()
    sample_times = []
    for img in sample:
        t = time.perf_counter()
        tracker.track(img)
        sample_times.append((time.perf_counter() - t) * 1000)
    row = summary.iloc[0]
    return {
        "tracker": cfg,
        "idf1": round(float(row["idf1"]) * 100, 2),
        "mota": round(float(row["mota"]) * 100, 2),
        "id_switches": int(row["num_switches"]),
        "fp": int(row["num_false_positives"]),
        "fn": int(row["num_misses"]),
        "sportsmot_track_ms": round(median(times[10:]), 2),
        "sample_track_ms": round(median(sample_times[10:]), 2),
    }


results = []
for cfg in ["ocsort.yaml", "bytetrack.yaml", "fasttrack.yaml"] * 2:
    r = run(cfg)
    results.append(r)
    print(json.dumps(r), flush=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=2))
