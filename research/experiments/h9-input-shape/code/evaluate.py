"""H9 step 3: per input shape — TRT forward time on sample.mp4 and SportsMOT identity metrics."""

import src  # noqa: F401  # project-local caches

import gc
import json
import sys
import time
from pathlib import Path
from statistics import median

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.tools import mot  # noqa: E402
from src.backbone.gpu_utils import fp16_model_path  # noqa: E402
from src.serve.pipeline.tracker import Tracker  # noqa: E402

OUT = Path(sys.argv[1])
VARIANTS = {
    "960x960": "models/player_detector.onnx",
    "544x960": "models/player_detector_544x960.onnx",
    "768x768": "models/player_detector_768x768.onnx",
    "448x768": "models/player_detector_448x768.onnx",
    "384x640": "models/player_detector_384x640.onnx",
}

cap = cv2.VideoCapture("data/sample.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
sample = [cap.read()[1] for _ in range(200)]
cap.release()
sportsmot = mot.frames()

results = {}
for name, path in VARIANTS.items():
    t0 = time.perf_counter()
    tracker = Tracker(fp16_model_path(path), "ocsort.yaml", "0", True, 0.3, True)
    build_s = round(time.perf_counter() - t0, 1)
    predictor = tracker._model.predictor
    provider = predictor.model.backend.session.get_providers()[0]
    forward = []
    for i, frame in enumerate(sample):
        im = predictor.preprocess([frame])
        torch.cuda.synchronize()
        t = time.perf_counter()
        predictor.model(im)
        forward.append((time.perf_counter() - t) * 1000)
    tracker.reset()
    metrics = mot.evaluate(tracker.track, sportsmot)
    results[name] = {"provider": provider, "setup_s": build_s, "input": list(im.shape[2:]),
                     "forward_ms": round(median(forward[10:]), 2), **metrics}
    print(name, json.dumps(results[name]), flush=True)
    del tracker, predictor
    gc.collect()
    torch.cuda.empty_cache()

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "variants.json").write_text(json.dumps(results, indent=2))
