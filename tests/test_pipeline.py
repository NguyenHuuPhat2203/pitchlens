"""Integration tests for the pipeline — end-to-end frame processing."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.controller.pipeline import Pipeline


def _make_test_video(path: str, frame_count: int = 5) -> None:
    """Write a short synthetic video for pipeline testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 25.0, (320, 240))
    for _ in range(frame_count):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _make_config(video_path: str) -> dict:
    """Return minimal config for testing."""
    return yaml.safe_load(f"""
model:
  player_detector: "yolo26s.pt"
  detection_conf: 0.3
  device: "cpu"

pipeline:
  display: false
  cpu_workers: 2

team_classifier:
  enabled: false

calibration:
  enabled: false
    """)


class TestPipeline:
    """End-to-end pipeline tests."""

    def test_processes_all_frames(self) -> None:
        """Pipeline processes every frame without deadlocking."""
        with tempfile.TemporaryDirectory() as tmp:
            video_path = str(Path(tmp) / "test.mp4")
            _make_test_video(video_path, frame_count=5)
            config = _make_config(video_path)
            pipeline = Pipeline(config)
            processed = pipeline.run(video_path)
            assert len(processed) == 5, (
                f"Expected 5 frames, got {len(processed)}"
            )

    def test_output_frames_have_correct_shape(self) -> None:
        """Every output frame is a valid BGR image array."""
        with tempfile.TemporaryDirectory() as tmp:
            video_path = str(Path(tmp) / "test.mp4")
            _make_test_video(video_path, frame_count=3)
            config = _make_config(video_path)
            pipeline = Pipeline(config)
            processed = pipeline.run(video_path)
            for frame in processed:
                assert isinstance(frame, np.ndarray)
                assert frame.shape == (240, 320, 3)
