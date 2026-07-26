"""Multi-object tracker — wraps Ultralytics built-in OC-SORT."""

import numpy as np
from ultralytics import YOLO

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
    ) -> None:
        """Load the YOLO model with tracking configuration.

        Args:
            model_name: YOLO model filename, e.g. ``"yolo26s.pt"``.
            tracker_config: Tracker YAML filename (ocsort.yaml, etc.).
            device: PyTorch device string.
        """
        self._model = YOLO(model_name)
        self._tracker_config = tracker_config
        self._device = device

    def track(self, frame: np.ndarray) -> list[PlayerTrack]:
        """Detect and track persons in a single frame.

        Args:
            frame: BGR image array (height, width, 3).

        Returns:
            List of PlayerTrack objects with persistent track IDs.
        """
        results = self._model.track(
            frame,
            device=self._device,
            tracker=self._tracker_config,
            classes=[0],  # person only
            verbose=False,
            persist=True,  # keep IDs across track() calls
        )
        tracks: list[PlayerTrack] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            ids = (
                result.boxes.id.cpu().numpy().astype(int)
                if result.boxes.id is not None
                else [-1] * len(boxes)
            )

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
