"""Player tracker: YOLO person detection + OC-SORT association."""

import numpy as np
from ultralytics import YOLO

from src.backbone.gpu_utils import enable_gpu_letterbox, enable_tensorrt, safe_predict
from src.model.track import PlayerTrack
from src.serve.pipeline.ocsort import OCSort


class Tracker:
    """Detects persons with YOLO and tracks them with OC-SORT.

    Returns the same tracks as Ultralytics' ``model.track(tracker="ocsort.yaml")``.
    :class:`OCSort` is an exact, batched port of that tracker, and boxes are
    clipped to the frame as Ultralytics' ``Results.update`` does. It is
    4.5 ms/frame faster (research/experiments/h6d-ocsort-ship).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "0",
        half: bool = False,
        conf: float = 0.25,
        tensorrt: bool = False,
    ) -> None:
        """Load the YOLO model.

        Args:
            model_name: YOLO model filename, e.g. ``"yolo26s.pt"``.
            device: PyTorch device string.
            half: Run FP16 inference.
            conf: Minimum detection confidence (``model.detection_conf``).
            tensorrt: Run an ONNX model on the TensorRT EP when available.
        """
        self._model = YOLO(model_name)
        self._device = device
        self._half = half
        self._conf = conf
        self._ocsort = OCSort()
        if device != "cpu":
            self._detect(np.zeros((64, 64, 3), dtype=np.uint8))  # builds the predictor
            if tensorrt:
                enable_tensorrt(self._model, device)
            enable_gpu_letterbox(self._model)

    def reset(self) -> None:
        """Forget all tracks and restart IDs (call between videos)."""
        self._ocsort.reset()

    def track(self, frame: np.ndarray) -> list[PlayerTrack]:
        """Detect and track persons in one frame.

        A new player is reported from its second frame on (OC-SORT
        confirmation), except on a video's first frame.

        Args:
            frame: BGR image array (height, width, 3).

        Returns:
            Confirmed tracks with persistent IDs. Boxes are the Kalman-filtered
            state, clipped to the frame.
        """
        det = self._detect(frame)
        rows = self._ocsort.update(det[:, :4], det[:, 4], det[:, 5])
        h, w = frame.shape[:2]
        rows[:, [0, 2]] = rows[:, [0, 2]].clip(0, w)
        rows[:, [1, 3]] = rows[:, [1, 3]].clip(0, h)
        return [
            PlayerTrack(
                track_id=int(r[4]),
                bbox=(float(r[0]), float(r[1]), float(r[2]), float(r[3])),
                class_id=int(r[6]),
                confidence=float(r[5]),
            )
            for r in rows
        ]

    def _detect(self, frame: np.ndarray) -> np.ndarray:
        """Person detections as (N, 6) rows ``[x1, y1, x2, y2, conf, cls]``."""
        results = safe_predict(
            self._model.predict,
            frame,
            device=self._device,
            quantize=16 if self._half else None,
            classes=[0],  # person only
            conf=self._conf,
            imgsz=960,  # for .pt models; exported ONNX models keep their own input size
            verbose=False,
        )
        return results[0].boxes.data.cpu().numpy()
