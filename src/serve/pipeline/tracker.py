"""Multi-object tracker — wraps Ultralytics built-in OC-SORT."""

import numpy as np
from ultralytics import YOLO

from src.backbone.gpu_utils import safe_predict
from src.model.track import PlayerTrack


class Tracker:
    """Runs YOLO detection + OC-SORT tracking on a frame.

    Uses ``model.track()`` which couples detection and tracking for
    performance.  The separate Detector from ticket 02 is kept for
    detection-only use cases (team classifier sampling).
    """

    def __init__(
        self,
        model_name: str,
        tracker_config: str,
        device: str = "0",
        half: bool = False,
        conf: float = 0.25,
    ) -> None:
        """Load the YOLO model with tracking configuration.

        Args:
            model_name: YOLO model filename, e.g. ``"yolo26s.pt"``.
            tracker_config: Tracker YAML filename (ocsort.yaml, etc.).
            device: PyTorch device string.
            half: Run FP16 inference.
            conf: Minimum detection confidence (``model.detection_conf``).
        """
        self._model = YOLO(model_name)
        self._tracker_config = tracker_config
        self._device = device
        self._half = half
        self._conf = conf

    def reset(self) -> None:
        """Forget all tracks and restart IDs (call between videos)."""
        predictor = self._model.predictor
        for tracker in getattr(predictor, "trackers", ()):
            tracker.reset()

    def track(self, frame: np.ndarray) -> list[PlayerTrack]:
        """Detect and track persons in a single frame.

        Detections the tracker has not (yet) assigned an ID to are
        dropped, so every returned track has ``track_id >= 0``.

        Args:
            frame: BGR image array (height, width, 3).

        Returns:
            List of PlayerTrack objects with persistent track IDs.
        """
        results = safe_predict(
            self._model.track,
            frame,
            device=self._device,
            quantize=16 if self._half else None,
            tracker=self._tracker_config,
            classes=[0],  # person only
            conf=self._conf,
            imgsz=960,  # higher resolution for small players
            verbose=False,
            persist=True,  # keep IDs across track() calls
        )
        tracks: list[PlayerTrack] = []
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            ids = result.boxes.id.cpu().numpy().astype(int)

            for bbox, conf, cls_id, track_id in zip(boxes, confs, classes, ids):
                tracks.append(
                    PlayerTrack(
                        track_id=int(track_id),
                        bbox=tuple(float(v) for v in bbox),
                        class_id=int(cls_id),
                        confidence=float(conf),
                    )
                )
        return tracks
