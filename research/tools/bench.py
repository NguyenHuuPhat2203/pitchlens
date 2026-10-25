"""Shared harness: time Pipeline.run on sample.mp4, interleave variants, compare outputs."""

import src  # noqa: F401  # project-local caches

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.config import load_config
from src.serve.pipeline.pipeline import Pipeline


def run_pipeline(
    out_dir: Path,
    tag: str,
    setup: Callable[[Pipeline], None] | None = None,
    configure: Callable | None = None,
) -> dict:
    """One JSON-only run of the default config.

    ``configure(cfg)`` edits the config before the pipeline is built; ``setup(pipeline)`` edits it after.
    """
    cfg = load_config("config/default.yaml")
    if configure is not None:
        configure(cfg)
    cfg.pipeline.display = False
    base = out_dir / tag
    cfg.pipeline.output_json = str(base / "frames.json")
    cfg.timeline.output_json = str(base / "timeline.json")
    cfg.timeline.benchmark_json = str(base / "benchmark.json")
    pipeline = Pipeline(cfg)
    if setup is not None:
        setup(pipeline)
    t = time.perf_counter()
    n = pipeline.run(cfg.video.source).frame_count
    wall = time.perf_counter() - t
    bench = json.loads(Path(cfg.timeline.benchmark_json).read_text())
    return {
        "tag": tag,
        "frames": n,
        "ms_per_frame": round(wall * 1000 / n, 2),
        **{f"{k}_ms": bench[k]["median_ms"] for k in ("player_track", "ball_detect") if k in bench},
    }


def interleave(
    out_dir: Path,
    variants: dict[str, Callable | None],
    order: str,
    configure: dict[str, Callable] | None = None,
) -> list[dict]:
    """Run variants in ``order`` (e.g. "SGSG"), print and save each result."""
    results = []
    for i, key in enumerate(order):
        tag = f"run{i + 1}_{key}"
        r = {"variant": key, **run_pipeline(out_dir, tag, variants[key], (configure or {}).get(key))}
        results.append(r)
        print(json.dumps(r), flush=True)
    (out_dir / "runs.json").write_text(json.dumps(results, indent=2))
    means = {k: round(float(np.mean([r["ms_per_frame"] for r in results if r["variant"] == k])), 2) for k in variants}
    print("mean ms/frame", means, flush=True)
    return results


def _frames(run_dir: Path) -> list[dict]:
    return [item["result"] for item in json.loads((run_dir / "frames.json").read_text())]


def _iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def compare(base: Path, test: Path) -> dict:
    """Guard metrics of run ``test`` against run ``base`` (same video)."""
    fb, ft = _frames(base), _frames(test)
    matched = total = 0
    ious = []
    for a, b in zip(fb, ft):
        boxes, used = [p["bbox"] for p in b["players"]], set()
        for p in a["players"]:
            total += 1
            best, j = max(((_iou(p["bbox"], q), k) for k, q in enumerate(boxes) if k not in used), default=(0.0, -1))
            if best >= 0.5:
                matched += 1
                used.add(j)
                ious.append(best)

    def in_play(frames):
        return round(sum((f["ball"] or {}).get("state") == "in_play" for f in frames) / len(frames), 4)

    return {
        "frames": [len(fb), len(ft)],
        "identical": fb == ft,
        "player_recall": round(matched / max(total, 1), 4),
        "player_mean_iou": round(float(np.mean(ious)) if ious else 0.0, 4),
        "ball_in_play": [in_play(fb), in_play(ft)],
        "shots": [[e["frame_number"] for f in fr for e in f["events"]] for fr in (fb, ft)],
    }
