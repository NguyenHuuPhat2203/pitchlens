"""Ball detector — finds the football in a frame using YOLO."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from src.model.detection import Detection
from src.serve.pipeline.gpu_utils import safe_predict


class BallDetector:
    """Detects the football using a YOLO model.

    Custom fine-tuned models use class index 0 for the ball.
    """

    _BALL_CLASS = 0

    def __init__(self, model_name: str, device: str = "0", half: bool = False) -> None:
        """Load the YOLO model.

        Args:
            model_name: YOLO model filename.
            device: PyTorch device string.
            half: Run FP16 inference.
        """
        self._model = YOLO(model_name)
        self._device = device
        self._half = half

    def detect_ball(self, frame: np.ndarray) -> Detection | None:
        """Run inference and return the highest-confidence ball detection.

        Args:
            frame: BGR image array (height, width, 3).

        Returns:
            A Detection if a ball is found, or None.
        """
        results = safe_predict(
            self._model,
            frame,
            device=self._device,
            quantize=16 if self._half else None,
            classes=[self._BALL_CLASS],
            verbose=False,
        )
        best: Detection | None = None
        if results is None:
            return None
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for bbox, conf in zip(boxes, confs):
                det = Detection(
                    bbox=tuple(float(v) for v in bbox),
                    class_id=self._BALL_CLASS,
                    confidence=float(conf),
                )
                if best is None or det.confidence > best.confidence:
                    best = det
        return best
