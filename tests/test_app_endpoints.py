"""Tests for HTTP endpoints - upload, stream, progress, analyze."""

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serve.adapt import app as app_mod
from src.serve.adapt.app import create_app


class StubPipeline:
    def __init__(self, config):
        pass

    def reset(self, fps):
        pass

    def process_frame(self, image, frame_index, timestamp=None):
        return image, None


class CrashPipeline(StubPipeline):
    def process_frame(self, image, frame_index, timestamp=None):
        raise RuntimeError("boom")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_mod, "Pipeline", StubPipeline)
    app_mod._sessions.clear()
    return TestClient(create_app())


@pytest.fixture
def video_bytes(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(5):
        w.write(np.full((48, 64, 3), i * 40, np.uint8))
    w.release()
    return path.read_bytes()


def _upload(client, data, name="clip.avi"):
    return client.post("/v1/upload", files={"file": (name, data, "video/avi")})


def test_upload_endpoint(client, video_bytes):
    resp = _upload(client, video_bytes)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["result"]["total_frames"] == 5


def test_stream_unknown_session_returns_404(client):
    assert client.get("/v1/stream/nonexistent").status_code == 404


def test_progress_unknown_session_returns_404(client):
    assert client.get("/v1/progress/nonexistent").status_code == 404


def test_upload_traversal_filename_stays_in_temp_dir(client, video_bytes):
    sid = _upload(client, video_bytes, "../../evil.exe").json()["result"]["session_id"]
    path = Path(app_mod._sessions[sid].video_path)
    assert path.name == "input.mp4"
    assert path.parent.name.startswith("football_")


def test_upload_garbage_returns_400_and_no_leftover_dir(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod.tempfile, "tempdir", str(tmp_path / "tmp"))
    os.makedirs(tmp_path / "tmp")
    resp = _upload(client, b"not a video")
    assert resp.status_code == 400
    assert list((tmp_path / "tmp").iterdir()) == []


def test_upload_oversize_returns_413_and_cleans(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod.tempfile, "tempdir", str(tmp_path / "tmp"))
    os.makedirs(tmp_path / "tmp")
    monkeypatch.setattr(app_mod, "MAX_UPLOAD_BYTES", 10)
    assert _upload(client, b"x" * 100).status_code == 413
    assert list((tmp_path / "tmp").iterdir()) == []


def test_stream_completes_cleans_dir_and_progress_reports_done(client, video_bytes):
    sid = _upload(client, video_bytes).json()["result"]["session_id"]
    tmp_dir = Path(app_mod._sessions[sid].video_path).parent
    resp = client.get(f"/v1/stream/{sid}")
    assert resp.content.count(b"image/jpeg") >= 5
    assert not tmp_dir.exists()
    result = client.get(f"/v1/progress/{sid}").json()["result"]
    assert result["done"] is True and result["processed"] == 5 and result["error"] is None


def test_stream_terminates_after_pipeline_crash(client, video_bytes, monkeypatch):
    monkeypatch.setattr(app_mod, "Pipeline", CrashPipeline)
    sid = _upload(client, video_bytes).json()["result"]["session_id"]
    client.get(f"/v1/stream/{sid}")  # would hang before the fix
    result = client.get(f"/v1/progress/{sid}").json()["result"]
    assert result["done"] is True
    assert "boom" in result["error"]


def test_stream_returns_429_when_busy(client, video_bytes):
    sid = _upload(client, video_bytes).json()["result"]["session_id"]
    assert app_mod._slots.acquire(blocking=False)
    try:
        assert client.get(f"/v1/stream/{sid}").status_code == 429
    finally:
        app_mod._slots.release()


def test_expired_unclaimed_session_purged_with_dir(client, video_bytes, monkeypatch):
    sid = _upload(client, video_bytes).json()["result"]["session_id"]
    tmp_dir = Path(app_mod._sessions[sid].video_path).parent
    monkeypatch.setattr(app_mod, "SESSION_TTL_S", -1)
    _upload(client, video_bytes)
    assert sid not in app_mod._sessions
    assert not tmp_dir.exists()


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "https://evil.com/watch?v=1", "--exec=calc", "ftp://youtube.com/x",
])
def test_analyze_rejects_bad_url(client, url):
    assert client.post("/v1/analyze", json={"url": url}).status_code == 400


def test_analyze_downloads_with_double_dash_and_cleans_on_failure(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod.tempfile, "tempdir", str(tmp_path))
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(app_mod.subprocess, "run", fake_run)
    resp = client.post("/v1/analyze", json={"url": "https://youtu.be/abc"})
    assert resp.status_code == 400
    assert seen["cmd"][-2:] == ["--", "https://youtu.be/abc"]
    assert list(tmp_path.iterdir()) == []


def test_analyze_missing_ytdlp_returns_503(client, monkeypatch):
    def fake_run(cmd, **kw):
        raise FileNotFoundError

    monkeypatch.setattr(app_mod.subprocess, "run", fake_run)
    assert client.post("/v1/analyze", json={"url": "https://youtu.be/abc"}).status_code == 503
