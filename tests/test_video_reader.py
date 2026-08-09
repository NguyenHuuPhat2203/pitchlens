"""Tests for VideoReader — frame iteration and metadata correctness."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.serve.pipeline.video_reader import VideoReader


def _make_test_video(path: str, frame_count: int = 30, fps: float = 25.0) -> None:
    """Write a synthetic black video to *path* with the given frame count."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (640, 480))
    for i in range(frame_count):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame, str(i), (300, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
        )
        writer.write(frame)
    writer.release()


class TestVideoReader:
    """Unit tests for VideoReader."""

    def test_yields_correct_frame_count(self) -> None:
        """VideoReader yields exactly the number of frames written."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test.mp4")
            _make_test_video(path, frame_count=30)
            reader = VideoReader(path)
            frames = list(reader)
            assert len(frames) == 30

    def test_frame_indices_are_sequential(self) -> None:
        """Frame indices start at 0 and increase by 1."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test.mp4")
            _make_test_video(path, frame_count=10)
            reader = VideoReader(path)
            indices = [f.index for f in reader]
            assert indices == list(range(10))

    def test_frame_images_are_non_empty(self) -> None:
        """Every frame has a valid BGR image array."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test.mp4")
            _make_test_video(path, frame_count=5)
            reader = VideoReader(path)
            for frame in reader:
                assert isinstance(frame.image, np.ndarray)
                assert frame.image.shape == (480, 640, 3)
                assert frame.image.dtype == np.uint8

    def test_timestamp_increases_monotonically(self) -> None:
        """Timestamps grow at 1/fps intervals."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test.mp4")
            _make_test_video(path, frame_count=10, fps=25.0)
            reader = VideoReader(path)
            timestamps = [f.timestamp for f in reader]
            # Allow small floating-point drift.
            for i in range(1, len(timestamps)):
                assert timestamps[i] > timestamps[i - 1]
            assert timestamps[0] == pytest.approx(0.0, abs=0.01)
            assert timestamps[-1] == pytest.approx(9 / 25.0, abs=0.01)

    def test_raises_for_missing_file(self) -> None:
        """VideoReader raises FileNotFoundError for a nonexistent path."""
        with pytest.raises(FileNotFoundError):
            VideoReader("nonexistent_video_12345.mp4")
