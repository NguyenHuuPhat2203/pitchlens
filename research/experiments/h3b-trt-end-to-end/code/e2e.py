"""H3b: end-to-end ms/frame with CUDA EP (B) vs TensorRT EP (T), interleaved."""

import src  # noqa: F401  # project-local caches

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt_libs

os.add_dll_directory(os.path.dirname(tensorrt_libs.__file__))
import onnxruntime as ort  # noqa: E402

from src.config import load_config  # noqa: E402
from src.serve.pipeline.pipeline import Pipeline  # noqa: E402

OUT = Path(sys.argv[1])
CACHE = str(Path(".cache/trt").resolve())


def swap_to_trt(yolo) -> None:
    """Replace a loaded Ultralytics ONNX backend's session with a TRT EP session."""
    be = yolo.predictor.model.backend
    path = be.session._model_path
    s = ort.InferenceSession(path, providers=[
        ("TensorrtExecutionProvider", {
            "trt_fp16_enable": True, "trt_engine_cache_enable": True,
            "trt_engine_cache_path": CACHE, "trt_timing_cache_enable": True,
            "trt_timing_cache_path": CACHE,
        }),
        "CUDAExecutionProvider",
    ])
    assert s.get_providers()[0] == "TensorrtExecutionProvider"
    io = s.io_binding()
    for out, tensor in zip(s.get_outputs(), be.bindings):
        io.bind_output(name=out.name, device_type="cuda", device_id=0,
                       element_type=np.float16 if "float16" in out.type else np.float32,
                       shape=tuple(tensor.shape), buffer_ptr=tensor.data_ptr())
    be.session, be.io = s, io


def run(variant: str, tag: str) -> dict:
    cfg = load_config("config/default.yaml")
    cfg.pipeline.display = False
    cfg.pipeline.output_video = None
    base = OUT.parent / tag
    cfg.pipeline.output_json = str(base / "frames.json")
    cfg.timeline.output_json = str(base / "timeline.json")
    cfg.timeline.benchmark_json = str(base / "benchmark.json")
    p = Pipeline(cfg)
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    p._tracker.track(dummy)
    p._ball_detector.detect_balls(dummy)
    if variant == "T":
        swap_to_trt(p._tracker._model)
        swap_to_trt(p._ball_detector._model)
    t = time.perf_counter()
    n = p.run(cfg.video.source).frame_count
    wall = time.perf_counter() - t
    bench = json.loads(Path(cfg.timeline.benchmark_json).read_text())
    return {"variant": variant, "tag": tag, "frames": n,
            "ms_per_frame": round(wall * 1000 / n, 2),
            "player_track_ms": bench["player_track"]["median_ms"],
            "ball_detect_ms": bench["ball_detect"]["median_ms"]}


results = []
for i, v in enumerate("BTBT"):
    r = run(v, f"run{i + 1}_{v}")
    results.append(r)
    print(json.dumps(r), flush=True)
OUT.write_text(json.dumps(results, indent=2))
