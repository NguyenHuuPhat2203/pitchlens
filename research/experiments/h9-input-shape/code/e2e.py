"""H9 step 4: end to end 960x960 (A) vs 544x960 (B), interleaved; then where A and B detections differ."""

import src  # noqa: F401  # project-local caches

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.tools import mot  # noqa: E402
from research.tools.bench import compare, interleave  # noqa: E402
from src.backbone.gpu_utils import fp16_model_path  # noqa: E402
from src.serve.pipeline.tracker import Tracker  # noqa: E402

OUT = Path(sys.argv[1])
B_MODEL = "models/player_detector_544x960.onnx"


def use_b(cfg) -> None:
    cfg.model.player_detector = B_MODEL


interleave(OUT, {"A": None, "B": None}, "ABAB", configure={"B": use_b})
guard = {"B_vs_A": compare(OUT / "run1_A", OUT / "run2_B"), "A_vs_A": compare(OUT / "run1_A", OUT / "run3_A")}
print("guard", json.dumps(guard), flush=True)
(OUT / "guard.json").write_text(json.dumps(guard, indent=2))

# EXPLORATORY: detections vs SportsMOT ground truth, split by vertical position.
gt = np.loadtxt(mot.SEQ / "gt" / "gt.txt", delimiter=",")
images = mot.frames()
bands = [(0, 100), (100, 250), (250, 450), (450, 720)]  # box bottom y (pixels)
report = {}
for name, path in (("960x960", "models/player_detector.onnx"), ("544x960", B_MODEL)):
    tracker = Tracker(fp16_model_path(path), "ocsort.yaml", "0", True, 0.3, True)
    stats = {f"{lo}-{hi}": {"det": 0, "fp": 0, "gt": 0, "fn": 0} for lo, hi in bands}
    for i, img in enumerate(images, start=1):
        r = tracker._model.predict(img, device="0", quantize=16, classes=[0], conf=0.3, verbose=False)[0]
        det = r.boxes.xyxy.cpu().numpy()
        g = gt[gt[:, 0] == i][:, 2:6]
        g = np.c_[g[:, :2], g[:, :2] + g[:, 2:]]
        dist = mot._iou_distance(np.c_[det[:, :2], det[:, 2:] - det[:, :2]], np.c_[g[:, :2], g[:, 2:] - g[:, :2]])
        det_hit = ~np.isnan(dist).all(axis=1) if dist.size else np.zeros(len(det), bool)
        gt_hit = ~np.isnan(dist).all(axis=0) if dist.size else np.zeros(len(g), bool)
        for lo, hi in bands:
            k = f"{lo}-{hi}"
            in_det, in_gt = (det[:, 3] >= lo) & (det[:, 3] < hi), (g[:, 3] >= lo) & (g[:, 3] < hi)
            stats[k]["det"] += int(in_det.sum())
            stats[k]["fp"] += int((in_det & ~det_hit).sum())
            stats[k]["gt"] += int(in_gt.sum())
            stats[k]["fn"] += int((in_gt & ~gt_hit).sum())
    report[name] = stats
    print(name, json.dumps(stats), flush=True)
    del tracker
(OUT / "fp_by_band.json").write_text(json.dumps(report, indent=2))
