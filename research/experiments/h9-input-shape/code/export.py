"""H9 step 1-2: is models/player_detector.onnx = yolo26s.pt at 960? Then export the variants."""

import src  # noqa: F401  # project-local caches

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
VARIANTS = [(960, 960), (544, 960), (768, 768), (448, 768), (384, 640)]
EXPORT_ARGS = dict(format="onnx", batch=1, dynamic=False, simplify=True, nms=False)  # = current ONNX metadata


def export(h: int, w: int) -> Path:
    dst = Path(f"models/player_detector_{h}x{w}.onnx")
    if not dst.exists():
        shutil.move(YOLO("yolo26s.pt").export(imgsz=(h, w), **EXPORT_ARGS), dst)
    return dst


def detections(path: Path, frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = LetterBox(size, auto=False)(image=frame)[..., ::-1].transpose(2, 0, 1)[None]
    out = sess.run(None, {sess.get_inputs()[0].name: np.ascontiguousarray(x, dtype=np.float32) / 255})[0][0]
    out = out[out[:, 4] >= 0.25]
    return out[np.lexsort((out[:, 0], -out[:, 4]))]


ckpt = torch.load("yolo26s.pt", map_location="cpu", weights_only=False)
report = {"pt": {k: str(ckpt.get(k)) for k in ("date", "version", "license")}}

cap = cv2.VideoCapture("data/sample.mp4")
frames = []
for i in range(0, 1500, 75):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    frames.append(cap.read()[1])
cap.release()
ref, new = Path("models/player_detector.onnx"), export(960, 960)
diffs, counts = [], []
for f in frames:
    a, b = detections(ref, f, (960, 960)), detections(new, f, (960, 960))
    counts.append((len(a), len(b)))
    if len(a) == len(b) and len(a):
        diffs.append(float(np.abs(a[:, :5] - b[:, :5]).max()))
report["provenance"] = {
    "frames": len(frames),
    "same_count_frames": sum(x == y for x, y in counts),
    "max_abs_diff_box_conf": max(diffs) if diffs else None,
}
print(json.dumps(report), flush=True)
for h, w in VARIANTS[1:]:
    print("exported", export(h, w), flush=True)
(OUT / "provenance.json").write_text(json.dumps(report, indent=2))
