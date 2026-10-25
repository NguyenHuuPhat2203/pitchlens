"""H6d: batched OC-SORT v2 + clip — real-data replay, tracker cost, end-to-end S/V (identity required)."""

import src  # noqa: F401  # project-local caches

import gc
import json
import sys
import time
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.oc_sort import OCSORT
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.tools import mot  # noqa: E402
from research.tools.bench import compare, interleave  # noqa: E402
from src.backbone.gpu_utils import enable_gpu_letterbox, enable_tensorrt, fp16_model_path, safe_predict  # noqa: E402
from src.model.track import PlayerTrack  # noqa: E402
from src.serve.pipeline.ocsort import OCSort  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


class PortTracker:
    """predict + OCSort: the candidate for src/serve/pipeline/tracker.py."""

    def __init__(self, model_name, device="0", half=False, conf=0.25, tensorrt=False):
        self._model = YOLO(model_name)
        self._device, self._half, self._conf = device, half, conf
        self._ocsort = OCSort()
        if device != "cpu":
            self._detect(np.zeros((64, 64, 3), dtype=np.uint8))
            if tensorrt:
                enable_tensorrt(self._model, device)
            enable_gpu_letterbox(self._model)

    def reset(self) -> None:
        self._ocsort.reset()

    def _detect(self, frame) -> np.ndarray:
        r = safe_predict(self._model.predict, frame, device=self._device, quantize=16 if self._half else None,
                         classes=[0], conf=self._conf, imgsz=960, verbose=False)[0]
        return r.boxes.data.cpu().numpy()

    def track(self, frame) -> list[PlayerTrack]:
        d = self._detect(frame)
        rows = self._ocsort.update(d[:, :4], d[:, 4], d[:, 5])
        h, w = frame.shape[:2]  # Results.update clips tracker boxes to the image
        rows[:, [0, 2]] = rows[:, [0, 2]].clip(0, w)
        rows[:, [1, 3]] = rows[:, [1, 3]].clip(0, h)
        return [PlayerTrack(track_id=int(r[4]), bbox=(float(r[0]), float(r[1]), float(r[2]), float(r[3])),
                            class_id=int(r[6]), confidence=float(r[5])) for r in rows]


def video_frames(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    out = []
    while (frame := cap.read()[1]) is not None:
        out.append(frame)
    cap.release()
    return out


# 1. Record the product detector's output on both sequences.
player = fp16_model_path("models/player_detector.onnx")
detector = PortTracker(player, "0", True, 0.3, True)
recorded = {
    "sportsmot": [detector._detect(f) for f in mot.frames()],
    "sample": [detector._detect(f) for f in video_frames("data/sample.mp4")],
}
dtype = str(recorded["sample"][0].dtype)
del detector
gc.collect()
torch.cuda.empty_cache()

# 2. Replay both trackers on the recorded detections.
replay = {"input_dtype": dtype}
cfg = IterableSimpleNamespace(**YAML.load(check_yaml("ocsort.yaml")))
for name, dets in recorded.items():
    ref, port = OCSORT(cfg), OCSort()
    t_ref, t_port, mismatch = [], [], []
    for i, d in enumerate(dets):
        t = time.perf_counter()
        expected = np.asarray(ref.update(Boxes(d, (720, 1280)), None), dtype=np.float32).reshape(-1, 8)
        t_ref.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        got = port.update(d[:, :4], d[:, 4], d[:, 5])
        t_port.append((time.perf_counter() - t) * 1000)
        if not np.array_equal(got, expected):
            mismatch.append(i)
    replay[name] = {"frames": len(dets), "mismatched_frames": len(mismatch), "first_mismatch": mismatch[:5],
                    "ultralytics_update_ms": round(median(t_ref), 3), "port_update_ms": round(median(t_port), 3)}
    print(name, json.dumps(replay[name]), flush=True)
(OUT / "replay.json").write_text(json.dumps(replay, indent=2))


# 3. End to end, interleaved.
def port_tracker(pipeline) -> None:
    del pipeline._tracker
    gc.collect()
    torch.cuda.empty_cache()
    pipeline._tracker = PortTracker(player, "0", True, 0.3, True)


interleave(OUT, {"S": None, "V": port_tracker}, "SVSV")
guard = {"V_vs_S": compare(OUT / "run1_S", OUT / "run2_V"), "S_vs_S": compare(OUT / "run1_S", OUT / "run3_S")}
print("guard", json.dumps(guard), flush=True)
(OUT / "guard.json").write_text(json.dumps(guard, indent=2))
