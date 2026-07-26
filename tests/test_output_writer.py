"""Tests for OutputWriter — video and JSON output."""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.model.frame import ProcessedFrame
from src.model.track import PlayerTrack
from src.repository.output_writer import OutputWriter


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

    def test_write_json_uses_envelope_format(self) -> None:
        """write_json wraps data in {success, data, meta} per RULES.md."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = str(Path(tmp) / "output.json")
            processed = _make_processed_frames(3)
            writer = OutputWriter()
            writer.write_json(processed, output_path)

            with open(output_path, encoding="utf-8") as fh:
                doc = json.load(fh)

            assert doc["success"] is True
            assert "data" in doc
            assert "meta" in doc
            assert len(doc["data"]["frames"]) == 3

    def test_json_frame_has_player_data(self) -> None:
        """Each frame in the JSON contains player track data."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = str(Path(tmp) / "output.json")
            processed = _make_processed_frames(1)
            writer = OutputWriter()
            writer.write_json(processed, output_path)

            with open(output_path, encoding="utf-8") as fh:
                doc = json.load(fh)

            frame_data = doc["data"]["frames"][0]
            assert frame_data["frame_number"] == 0
            assert len(frame_data["players"]) == 1
            assert frame_data["players"][0]["track_id"] == 1
