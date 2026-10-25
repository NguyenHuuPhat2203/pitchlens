"""H6/H7: cProfile Tracker.track with the product TRT config."""

import src  # noqa: F401  # project-local caches

import cProfile
import json
import pstats
import sys
from pathlib import Path

import cv2

from src.backbone.gpu_utils import fp16_model_path
from src.serve.pipeline.tracker import Tracker

OUT = Path(sys.argv[1])
START, N, WARM = 100, 300, 10

cap = cv2.VideoCapture("data/sample.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, START)
frames = [cap.read()[1] for _ in range(N)]
cap.release()

tracker = Tracker(fp16_model_path("models/player_detector.onnx"), "ocsort.yaml",
                  device="0", half=True, conf=0.3, tensorrt=True)
provider = tracker._model.predictor.model.backend.session.get_providers()[0]
for f in frames[:WARM]:
    tracker.track(f)

counts = []
prof = cProfile.Profile()
prof.enable()
for f in frames[WARM:]:
    counts.append(len(tracker.track(f)))
prof.disable()

calls = N - WARM
stats = pstats.Stats(prof)
rows = []
for (file, line, func), (cc, nc, tt, ct, _) in stats.stats.items():
    rows.append({"func": f"{Path(file).name}:{line}:{func}", "ncalls": nc,
                 "cum_ms_per_frame": round(ct * 1000 / calls, 3),
                 "self_ms_per_frame": round(tt * 1000 / calls, 3)})
rows.sort(key=lambda r: -r["cum_ms_per_frame"])
OUT.write_text(json.dumps({"provider": provider, "frames": calls,
                           "tracks_per_frame": round(sum(counts) / calls, 1),
                           "top_cumulative": rows[:60]}, indent=2))
print(provider, "tracks/frame", round(sum(counts) / calls, 1))
for r in rows[:45]:
    print(f'{r["cum_ms_per_frame"]:8.3f} {r["self_ms_per_frame"]:8.3f} {r["ncalls"]:>7} {r["func"]}')
