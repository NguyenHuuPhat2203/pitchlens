"""H4: serial (S) vs ball-detection-in-parallel (P) pipeline, interleaved S/P/S/P."""

import src  # noqa: F401  # project-local caches

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.config import load_config
from src.serve.pipeline.pipeline import Pipeline

OUT = Path(sys.argv[1])


class ParallelPipeline(Pipeline):
    """Starts the ball detector on its own thread before player tracking."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._detect = self._ball_detector.detect_balls
        self._future = None

    def _process_frame_image(self, image, frame_index, timestamp, render=True):
        if frame_index % 2 == 0:
            self._future = self._pool.submit(self._detect, image)
        return super()._process_frame_image(image, frame_index, timestamp, render)

    def _update_ball(self, image, frame_index):
        # Base class calls detect_balls on even frames; hand it the future's result.
        self._ball_detector.detect_balls = lambda _img: self._future.result()
        try:
            return super()._update_ball(image, frame_index)
        finally:
            self._ball_detector.detect_balls = self._detect


def run(variant: str, tag: str) -> dict:
    cfg = load_config("config/default.yaml")
    cfg.pipeline.display = False
    base = OUT.parent / tag
    cfg.pipeline.output_json = str(base / "frames.json")
    cfg.timeline.output_json = str(base / "timeline.json")
    cfg.timeline.benchmark_json = str(base / "benchmark.json")
    p = (ParallelPipeline if variant == "P" else Pipeline)(cfg)
    t = time.perf_counter()
    n = p.run(cfg.video.source).frame_count
    wall = time.perf_counter() - t
    bench = json.loads(Path(cfg.timeline.benchmark_json).read_text())
    return {"variant": variant, "tag": tag, "frames": n,
            "ms_per_frame": round(wall * 1000 / n, 2),
            "player_track_ms": bench["player_track"]["median_ms"],
            "ball_detect_ms": bench["ball_detect"]["median_ms"]}


results = []
for i, v in enumerate("SPSP"):
    r = run(v, f"run{i + 1}_{v}")
    results.append(r)
    print(json.dumps(r), flush=True)
OUT.write_text(json.dumps(results, indent=2))
