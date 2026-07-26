"""YOLO object detector — wraps ultralytics.YOLO for player detection."""

import numpy as np
from ultralytics import YOLO

from src.model.detection import Detection


class Detector:
    """Runs YOLO inference on a frame and returns Detection objects.

    Filters to the COCO ``person`` class (class_id=0) by default.
    The model is loaded once at construction and reused across frames.
    """

    _PERSON_CLASS = 0

    def __init__(self, model_name: str, device: str = "0") -> None:
        """Load the YOLO model.

        Args:
            model_name: Model filename, e.g. ``"yolo26s.pt"``.
            device: PyTorch device string — ``"0"`` for GPU 0,
                ``"cpu"`` for CPU.
        """
        self._model = YOLO(model_name)
        self._device = device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a single BGR frame.

        Args:
            frame: BGR image array (height, width, 3).

        Returns:
            List of Detection objects for persons found in the frame.
        """
        results = self._model(
            frame,
            device=self._device,
            classes=[self._PERSON_CLASS],
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for bbox, conf, cls_id in zip(boxes, confs, classes):
                detections.append(
                    Detection(
                        bbox=tuple(float(v) for v in bbox),
                        class_id=int(cls_id),
                        confidence=float(conf),
                    )
                )
        return detections
