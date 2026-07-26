"""Writes pipeline output — annotated video and JSON data."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.model.frame import ProcessedFrame
from src.view.serializer import Serializer


class OutputWriter:
    """Saves annotated video frames and per-frame JSON data.

    The JSON output follows the RULES.md envelope convention.
    """

    def __init__(self) -> None:
        """Create a serializer for JSON output."""
        self._serializer = Serializer()

    def write_video(
        self, frames: list[np.ndarray], output_path: str, fps: float
    ) -> None:
        """Write annotated frames to an .mp4 file.

        Args:
            frames: Annotated BGR images in display order.
            output_path: Destination file path.
            fps: Frame rate for the output video.
        """
        if not frames:
            return

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            writer.write(frame)
        writer.release()

    def write_json(
        self, processed_frames: list[ProcessedFrame], output_path: str
    ) -> None:
        """Write processed frame data to a JSON file with envelope.

        Args:
            processed_frames: ProcessedFrame objects from the pipeline.
            output_path: Destination file path.
        """
        doc = self._serializer.serialize_all(processed_frames)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)
