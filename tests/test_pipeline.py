"""Integration tests for the pipeline — end-to-end frame processing."""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.model.track import PlayerTrack
from src.serve.pipeline.pipeline import Pipeline


def _make_test_video(path: str, frame_count: int = 5) -> None:
    """Write a short synthetic video for pipeline testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 25.0, (320, 240))
    for _ in range(frame_count):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _make_config(video_path: str, output_json=None, timeline_json=None):
    """Return minimal AppConfig for testing."""
    from src.config import (
        AppConfig,
        CalibrationConfig,
        ModelConfig,
        PipelineConfig,
        TeamClassifierConfig,
        VideoConfig,
    )

    return AppConfig(
        video=VideoConfig(source=video_path),
        model=ModelConfig(player_detector="yolo26s.pt", device="cpu"),
        pipeline=PipelineConfig(
            display=False, cpu_workers=2, output_json=output_json
        ),
        team_classifier=TeamClassifierConfig(enabled=False),
        calibration=CalibrationConfig(enabled=False),
        timeline={"output_json": timeline_json} if timeline_json else {},
    )


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



def test_json_only_batch_skips_renderer(tmp_path, monkeypatch) -> None:
    video_path = str(tmp_path / "test.mp4")
    frame_json = str(tmp_path / "frames.json")
    timeline_json = str(tmp_path / "timeline.json")
    _make_test_video(video_path, frame_count=1)
    pipeline = Pipeline(
        _make_config(video_path, frame_json, timeline_json)
    )
    pipeline._tracker.track = lambda image: [
        PlayerTrack(1, (10.0, 20.0, 50.0, 80.0), 0, 0.9)
    ]
    monkeypatch.setattr(
        pipeline._renderer,
        "render_tracks",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("renderer should be skipped")
        ),
    )

    frames = pipeline.run(video_path)

    assert len(frames) == 1
    assert Path(frame_json).exists()
    assert Path(timeline_json).exists()