"""H3: ORT TensorRT EP vs CUDA EP — latency and detection parity."""

import src  # noqa: F401  # project-local caches

import json
import os
import sys
import time
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import tensorrt_libs
from ultralytics.data.augment import LetterBox

os.add_dll_directory(os.path.dirname(tensorrt_libs.__file__))
import onnxruntime as ort  # noqa: E402

OUT = Path(sys.argv[1])
START, N, WARM, CONF = 100, 200, 10, 0.3
CACHE = Path(".cache/trt").resolve()
CACHE.mkdir(parents=True, exist_ok=True)


def frames():
    cap = cv2.VideoCapture("data/sample.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, START)
    out = [cap.read()[1] for _ in range(N)]
    cap.release()
    return out


def prep(img, size):
    x = LetterBox((size, size), auto=False)(image=img)[..., ::-1].transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x, dtype=np.float32) / 255.0


def session(path, trt):
    providers = ["CUDAExecutionProvider"]
    if trt:
        providers.insert(0, ("TensorrtExecutionProvider", {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(CACHE),
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": str(CACHE),
        }))
    s = ort.InferenceSession(path, providers=providers)
    want = "TensorrtExecutionProvider" if trt else "CUDAExecutionProvider"
    assert s.get_providers()[0] == want, s.get_providers()
    return s


def run_all(sess, inputs):
    name = sess.get_inputs()[0].name
    fp16 = "float16" in sess.get_inputs()[0].type
    times, outs = [], []
    t0 = time.perf_counter()
    sess.run(None, {name: inputs[0].astype(np.float16) if fp16 else inputs[0]})  # engine build
    build_s = time.perf_counter() - t0
    for i, x in enumerate(inputs):
        x = x.astype(np.float16) if fp16 else x
        t = time.perf_counter()
        out = sess.run(None, {name: x})[0][0].astype(np.float32)
        if i >= WARM:
            times.append((time.perf_counter() - t) * 1000)
        outs.append(out[out[:, 4] >= CONF])
    return round(median(times), 2), round(build_s, 1), outs


def iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])  # noqa: E731
    return inter / (area(a) + area(b) - inter + 1e-9)


def parity(ref, test):
    matched, total, ious = 0, 0, []
    for r, t in zip(ref, test):
        total += len(r)
        used = set()
        for box in r:
            if len(t) == 0:
                break
            scores = iou(box, t)
            scores[list(used)] = -1
            j = int(scores.argmax())
            if scores[j] >= 0.5:
                matched += 1
                used.add(j)
                ious.append(scores[j])
    return {"ref_boxes": total, "recall": round(matched / max(total, 1), 4),
            "mean_iou": round(float(np.mean(ious)) if ious else 0.0, 4)}


imgs = frames()
result = {}
for model, size in (("player_detector", 960), ("ball_detector", 1280)):
    inputs = [prep(im, size) for im in imgs]
    base_ms, _, base_out = run_all(session(f"models/{model}.fp16.onnx", trt=False), inputs)
    entry = {"cuda_fp16_ms": base_ms}
    for variant in ("onnx", "fp16.onnx"):
        ms, build, out = run_all(session(f"models/{model}.{variant}", trt=True), inputs)
        entry[f"trt_from_{variant}"] = {"ms": ms, "first_run_s": build,
                                        "speedup": round(base_ms / ms, 2), **parity(base_out, out)}
    result[model] = entry
    print(model, json.dumps(entry), flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2))
