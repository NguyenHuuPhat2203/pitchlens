"""Writes pipeline output — annotated video and JSON data."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2

from src.model.frame import ProcessedFrame
from src.model.tactical_event import TacticalEvent
from src.serve.artifacts.serializer import Serializer


class OutputWriter:
    """Saves annotated video frames and per-frame JSON data.

    The JSON output follows the RULES.md envelope convention.
    """

    def __init__(self) -> None:
        """Create a serializer for JSON output."""
        self._serializer = Serializer()

    @staticmethod
    def open_video(
        output_path: str, fps: float, size: tuple[int, int]
    ) -> cv2.VideoWriter:
        """Open an .mp4 writer; the caller writes frames and releases it.

        Args:
            output_path: Destination file path.
            fps: Frame rate for the output video.
            size: (height, width) of the frames.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        height, width = size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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

    def write_timeline(
        self, events: list[TacticalEvent], output_path: str
    ) -> None:
        """Write ordered tactical events as a JSON array."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump([asdict(event) for event in events], fh, indent=2)

    def write_benchmark(self, report: dict, output_path: str) -> None:
        """Write stage timing data as JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
