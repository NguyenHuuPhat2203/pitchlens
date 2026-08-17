"""Tests for HTTP endpoints — upload, stream, progress."""

import pytest
from fastapi.testclient import TestClient

from src.serve.adapt.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def test_upload_endpoint(client):
    resp = client.post(
        "/v1/upload",
        files={"file": ("clip.mp4", b"fakevideo", "video/mp4")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert "session_id" in body["result"]


def test_stream_unknown_session_returns_404(client):
    resp = client.get("/v1/stream/nonexistent")
    assert resp.status_code == 404


def test_progress_unknown_session_returns_404(client):
    resp = client.get("/v1/progress/nonexistent")
    assert resp.status_code == 404
