"""Tests for OutputWriter — video and JSON output."""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.model.frame import ProcessedFrame
from src.model.track import PlayerTrack
from src.serve.artifacts.output_writer import OutputWriter


def _make_processed_frames(count: int = 3) -> list[ProcessedFrame]:
    """Return synthetic ProcessedFrame objects for testing."""
    frames = []
    for i in range(count):
        player = PlayerTrack(
            track_id=1,
            bbox=(10.0, 20.0, 50.0, 80.0),
            class_id=0,
            confidence=0.95,
        )
        frames.append(
            ProcessedFrame(
                frame_number=i,
                timestamp=i / 25.0,
                players=[player],
                ball=None,
                events=[],
            )
        )
    return frames


class TestOutputWriter:
    """Unit tests for OutputWriter."""

    def test_write_video_creates_playable_file(self) -> None:
        """write_video creates a valid .mp4 with the correct frame count."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = str(Path(tmp) / "output.mp4")
            frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(5)]
            writer = OutputWriter()
            writer.write_video(frames, output_path, fps=25.0)

            assert Path(output_path).exists()
            cap = cv2.VideoCapture(output_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            assert frame_count == 5

    def test_write_json_uses_item_schema(self) -> None:
        """write_json outputs array of {status, result} per ML rules §09."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = str(Path(tmp) / "output.json")
            processed = _make_processed_frames(3)
            writer = OutputWriter()
            writer.write_json(processed, output_path)

            with open(output_path, encoding="utf-8") as fh:
                doc = json.load(fh)

            assert isinstance(doc, list)
            assert len(doc) == 3
            assert doc[0]["status"] == "success"
            assert "result" in doc[0]

    def test_json_frame_has_player_data(self) -> None:
        """Each frame result in the JSON contains player track data."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = str(Path(tmp) / "output.json")
            processed = _make_processed_frames(1)
            writer = OutputWriter()
            writer.write_json(processed, output_path)

            with open(output_path, encoding="utf-8") as fh:
                doc = json.load(fh)

            frame_data = doc[0]["result"]
            assert frame_data["frame_number"] == 0
            assert len(frame_data["players"]) == 1
            assert frame_data["players"][0]["track_id"] == 1
