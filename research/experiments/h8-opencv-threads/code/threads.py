"""H8: current product (S, cv2 single-threaded by Ultralytics) vs cv2 default threads (M)."""

import src  # noqa: F401  # project-local caches

import json
import sys
import time
from pathlib import Path

import cv2

from src.config import load_config
from src.serve.pipeline.pipeline import Pipeline

OUT = Path(sys.argv[1])


def run(variant: str, tag: str) -> dict:
    cfg = load_config("config/default.yaml")
    cfg.pipeline.display = False
    base = OUT.parent / tag
    cfg.pipeline.output_json = str(base / "frames.json")
    cfg.timeline.output_json = str(base / "timeline.json")
    cfg.timeline.benchmark_json = str(base / "benchmark.json")
    p = Pipeline(cfg)
    cv2.setNumThreads(-1 if variant == "M" else 0)
    threads = cv2.getNumThreads()
    t = time.perf_counter()
    n = p.run(cfg.video.source).frame_count
    wall = time.perf_counter() - t
    bench = json.loads(Path(cfg.timeline.benchmark_json).read_text())
    return {"variant": variant, "tag": tag, "cv2_threads": threads, "frames": n,
            "ms_per_frame": round(wall * 1000 / n, 2),
            "player_track_ms": bench["player_track"]["median_ms"],
            "ball_detect_ms": bench["ball_detect"]["median_ms"]}


results = []
for i, v in enumerate("SMSM"):
    r = run(v, f"run{i + 1}_{v}")
    results.append(r)
    print(json.dumps(r), flush=True)
OUT.write_text(json.dumps(results, indent=2))
