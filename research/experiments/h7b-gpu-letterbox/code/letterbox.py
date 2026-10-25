"""H7b: GPU letterbox — input parity, then interleaved S/G end-to-end runs."""

import src  # noqa: F401  # project-local caches

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.tools.bench import compare, interleave  # noqa: E402
from src.backbone.ball_detector import BallDetector  # noqa: E402
from src.backbone.gpu_utils import enable_gpu_letterbox, fp16_model_path  # noqa: E402
from src.serve.pipeline.tracker import Tracker  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

# 1. Input parity on frames 100-299.
cap = cv2.VideoCapture("data/sample.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
frames = [cap.read()[1] for _ in range(200)]
cap.release()
parity = {}
for name, yolo in (
    ("player", Tracker(fp16_model_path("models/player_detector.onnx"), "ocsort.yaml", "0", True, 0.3, True)._model),
    ("ball", BallDetector(fp16_model_path("models/ball_detector.onnx"), "0", True, 0.3, True)._model),
):
    cpu = yolo.predictor.preprocess
    enable_gpu_letterbox(yolo)
    gpu = yolo.predictor.preprocess
    diff, worst, n = 0, 0.0, 0
    for f in frames:
        a, b = cpu([f]).float(), gpu([f]).float()
        assert a.shape == b.shape
        d = (a - b).abs() * 255
        diff += int((d > 0.5).sum())
        worst = max(worst, float(d.max()))
        n += d.numel()
    parity[name] = {"input_shape": list(a.shape), "values_differing": round(diff / n, 6), "max_abs_grey": round(worst, 3)}
    del yolo
torch.cuda.empty_cache()
print("parity", json.dumps(parity), flush=True)
(OUT / "parity.json").write_text(json.dumps(parity, indent=2))


# 2. End to end, interleaved.
def gpu_letterbox(pipeline) -> None:
    enable_gpu_letterbox(pipeline._tracker._model)
    enable_gpu_letterbox(pipeline._ball_detector._model)


interleave(OUT, {"S": None, "G": gpu_letterbox}, "SGSG")
guard = {"G_vs_S": compare(OUT / "run1_S", OUT / "run2_G"), "S_vs_S": compare(OUT / "run1_S", OUT / "run3_S")}
print("guard", json.dumps(guard), flush=True)
(OUT / "guard.json").write_text(json.dumps(guard, indent=2))
