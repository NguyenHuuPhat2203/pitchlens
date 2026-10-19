"""Ball detector — finds the football in a frame using YOLO."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from src.backbone.gpu_utils import safe_predict
from src.model.detection import Detection


class BallDetector:
    """Detects the football using a YOLO model.

    Custom fine-tuned models use class index 0 for the ball.
    """

    _BALL_CLASS = 0

    def __init__(
        self,
        model_name: str,
        device: str = "0",
        half: bool = False,
        conf: float = 0.25,
    ) -> None:
        """Load the YOLO model.

        Args:
            model_name: YOLO model filename.
            device: PyTorch device string.
            half: Run FP16 inference.
            conf: Minimum detection confidence (``model.detection_conf``).
        """
        self._model = YOLO(model_name)
        self._device = device
        self._half = half
        self._conf = conf

    def detect_balls(self, frame: np.ndarray) -> list[Detection]:
        """Run inference and return every ball candidate above ``conf``.

        The tracker picks the candidate consistent with its trajectory.

        Args:
            frame: BGR image array (height, width, 3).

        Returns:
            Ball Detections, possibly empty.
        """
        results = safe_predict(
            self._model,
            frame,
            device=self._device,
            quantize=16 if self._half else None,
            classes=[self._BALL_CLASS],
            conf=self._conf,
            verbose=False,
        )
        candidates: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for bbox, conf in zip(boxes, confs):
                candidates.append(
                    Detection(
                        bbox=tuple(float(v) for v in bbox),
                        class_id=self._BALL_CLASS,
                        confidence=float(conf),
                    )
                )
        return candidates
