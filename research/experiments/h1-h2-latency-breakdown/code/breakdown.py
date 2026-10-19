"""H1/H2: split detector latency into preprocess / ORT run / Ultralytics / tracking."""

import src  # noqa: F401  # project-local caches

import json
import sys
import time
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

from src.backbone.ball_detector import BallDetector
from src.serve.pipeline.tracker import Tracker

OUT = Path(sys.argv[1])
START, N, WARM = 100, 200, 10


def frames():
    cap = cv2.VideoCapture("data/sample.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, START)
    out = [cap.read()[1] for _ in range(N)]
    cap.release()
    return out


def timed(fn):
    t = time.perf_counter()
    fn()
    return (time.perf_counter() - t) * 1000


def prep(img, size):
    x = LetterBox((size, size), auto=False)(image=img)
    x = x[..., ::-1].transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x, dtype=np.float16) / np.float16(255)


def stages(path, size, imgs, end_to_end):
    sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider"])
    assert sess.get_providers()[0] == "CUDAExecutionProvider"
    name = sess.get_inputs()[0].name
    io = sess.io_binding()
    res = {k: [] for k in ("preprocess", "run_numpy", "run_iobind", *end_to_end)}
    for i, img in enumerate(imgs):
        sample = {}
        x = None

        def do_prep():
            nonlocal x
            x = prep(img, size)

        sample["preprocess"] = timed(do_prep)
        sample["run_numpy"] = timed(lambda: sess.run(None, {name: x}))
        dev = ort.OrtValue.ortvalue_from_numpy(x, "cuda", 0)
        io.bind_ortvalue_input(name, dev)
        io.bind_output(sess.get_outputs()[0].name, "cuda")

        def run_bound():
            sess.run_with_iobinding(io)
            io.copy_outputs_to_cpu()

        sample["run_iobind"] = timed(run_bound)
        for k, fn in end_to_end.items():
            sample[k] = timed(lambda: fn(img))
        if i >= WARM:
            for k, v in sample.items():
                res[k].append(v)
    return {k: round(median(v), 2) for k, v in res.items()}


imgs = frames()
player_path, ball_path = "models/player_detector.fp16.onnx", "models/ball_detector.fp16.onnx"
yolo = YOLO(player_path, task="detect")
tracker = Tracker(player_path, "ocsort.yaml", device="0", half=True, conf=0.3)
ball = BallDetector(ball_path, device="0", half=True, conf=0.3)
predict_kw = dict(device="0", quantize=16, classes=[0], conf=0.3, imgsz=960, verbose=False)

result = {
    "player": stages(player_path, 960, imgs, {
        "predict": lambda im: yolo.predict(im, **predict_kw),
        "track": tracker.track,
    }),
    "ball": stages(ball_path, 1280, imgs, {"predict": ball.detect_balls}),
}
OUT.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
