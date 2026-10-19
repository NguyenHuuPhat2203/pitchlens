"""Integration tests for the pipeline — end-to-end frame processing."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.model.detection import Detection
from src.model.track import PlayerTrack
from src.serve.pipeline.pipeline import Pipeline


def _make_test_video(path: str, frame_count: int = 5, fps: float = 25.0) -> None:
    """Write a short synthetic video for pipeline testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (320, 240))
    for _ in range(frame_count):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()


def _make_config(video_path: str, output_json=None, timeline_json=None, **pipeline):
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
            **{"display": False, "output_json": output_json, **pipeline}
        ),
        team_classifier=TeamClassifierConfig(enabled=False),
        calibration=CalibrationConfig(enabled=False),
        ball_detector={"enabled": False},
        timeline={"output_json": timeline_json} if timeline_json else {},
    )


def _frames(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [item["result"] for item in json.load(fh)]


def _walking_player(frame_index: int) -> list[PlayerTrack]:
    """One player whose feet move 2 px right per frame."""
    x = 50.0 + 2.0 * frame_index
    return [PlayerTrack(1, (x, 100.0, x + 20.0, 160.0), 0, 0.9)]


class _FrameCounter:
    """Stub tracker.track that knows which frame it is called for."""

    def __init__(self, make_tracks) -> None:
        self.calls = 0
        self._make = make_tracks

    def __call__(self, image):
        tracks = self._make(self.calls)
        self.calls += 1
        return tracks


def test_output_video_contains_every_frame(tmp_path) -> None:
    video_path = str(tmp_path / "test.mp4")
    out_path = str(tmp_path / "out.mp4")
    _make_test_video(video_path, frame_count=5)

    result = Pipeline(_make_config(video_path, output_video=out_path)).run(video_path)

    assert result.frame_count == 5
    cap = cv2.VideoCapture(out_path)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    assert (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (320, 240)
    cap.release()


def test_json_only_batch_skips_renderer(tmp_path, monkeypatch) -> None:
    video_path = str(tmp_path / "test.mp4")
    frame_json = str(tmp_path / "frames.json")
    timeline_json = str(tmp_path / "timeline.json")
    _make_test_video(video_path, frame_count=4)
    pipeline = Pipeline(_make_config(video_path, frame_json, timeline_json))
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

    result = pipeline.run(video_path)

    assert result.frame_count == 4
    assert len(_frames(frame_json)) == 4
    assert Path(timeline_json).exists()


def test_calibration_prompt_keeps_frame_zero(tmp_path, monkeypatch) -> None:
    """Skipping calibration with Q must not drop frame 0 or crash."""
    video_path = str(tmp_path / "test.mp4")
    frame_json = str(tmp_path / "frames.json")
    _make_test_video(video_path, frame_count=3)
    config = _make_config(video_path, frame_json, display=True)
    config.calibration.enabled = True
    for name in ("imshow", "setMouseCallback", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, lambda *a, **k: None)
    # Q at the calibration prompt (100 ms polling); no key during playback.
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("q") if delay == 100 else -1)

    result = Pipeline(config).run(video_path)

    assert result.frame_count == 3
    assert [f["frame_number"] for f in _frames(frame_json)] == [0, 1, 2]


def test_headless_run_never_opens_calibration_window(tmp_path, monkeypatch) -> None:
    video_path = str(tmp_path / "test.mp4")
    _make_test_video(video_path, frame_count=2)
    config = _make_config(video_path)
    config.calibration.enabled = True
    monkeypatch.setattr(
        cv2, "imshow", lambda *a: (_ for _ in ()).throw(AssertionError("window opened"))
    )

    assert Pipeline(config).run(video_path).frame_count == 2


def test_timestamps_and_video_follow_source_fps(tmp_path) -> None:
    video_path = str(tmp_path / "test.mp4")
    frame_json = str(tmp_path / "frames.json")
    out_path = str(tmp_path / "out.mp4")
    _make_test_video(video_path, frame_count=3, fps=30.0)

    Pipeline(_make_config(video_path, frame_json, output_video=out_path)).run(video_path)

    stamps = [f["timestamp"] for f in _frames(frame_json)]
    assert stamps == pytest.approx([0.0, 1 / 30, 2 / 30])
    cap = cv2.VideoCapture(out_path)
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(30.0)
    cap.release()


def test_speed_uses_elapsed_time_of_whole_window(tmp_path, monkeypatch) -> None:
    """10 px = 1 m; feet move 2 px/frame at 25 fps -> 5 m/s = 18 km/h."""
    video_path = str(tmp_path / "test.mp4")
    out_path = str(tmp_path / "out.mp4")
    frame_json = str(tmp_path / "frames.json")
    _make_test_video(video_path, frame_count=20)
    config = _make_config(video_path, frame_json, output_video=out_path)
    config.calibration.enabled = True
    pipeline = Pipeline(config)
    pts = np.array([[0, 0], [1050, 0], [1050, 680], [0, 680]], dtype=np.float32)
    pipeline._calibrator.fit(pts, pts / 10)
    pipeline._tracker.track = _FrameCounter(_walking_player)
    labels_seen: list[str] = []
    render = pipeline._renderer.render_tracks

    def spy(image, detections, labels):
        labels_seen.extend(labels)
        return render(image, detections, labels)

    monkeypatch.setattr(pipeline._renderer, "render_tracks", spy)

    pipeline.run(video_path)

    speeds = [f["players"][0]["speed_kmh"] for f in _frames(frame_json)[1:]]
    assert speeds == pytest.approx([18.0] * len(speeds), abs=0.2)
    assert labels_seen[-1] == "#1 18km/h"


def test_second_run_starts_fresh(tmp_path) -> None:
    video_path = str(tmp_path / "test.mp4")
    frame_json = str(tmp_path / "frames.json")
    _make_test_video(video_path, frame_count=3)
    pipeline = Pipeline(_make_config(video_path, frame_json))

    for _ in range(2):
        pipeline._tracker.track = _FrameCounter(_walking_player)
        pipeline.run(video_path)
        first = _frames(frame_json)[0]["players"][0]
        assert first["direction_deg"] is None


def test_frames_without_players_still_track_the_ball(tmp_path) -> None:
    video_path = str(tmp_path / "test.mp4")
    frame_json = str(tmp_path / "frames.json")
    _make_test_video(video_path, frame_count=4)
    config = _make_config(video_path, frame_json)
    config.ball_detector.enabled = True
    pipeline = Pipeline(config)
    pipeline._tracker.track = lambda image: []
    pipeline._ball_detector.detect_balls = lambda image: [
        Detection(bbox=(100.0, 100.0, 110.0, 110.0), class_id=0, confidence=0.9)
    ]

    pipeline.run(video_path)

    balls = [f["ball"] for f in _frames(frame_json)]
    assert all(b is not None and b["state"] == "in_play" for b in balls)
