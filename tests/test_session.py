"""Tests for Session — single-client session state."""

from src.serve.adapt.producer import FrameProducer
from src.serve.adapt.session import Session


def test_claim_first_client_succeeds():
    s = Session(
        session_id="abc",
        video_path="x.mp4",
        total_frames=10,
        producer=FrameProducer(),
    )
    assert s.claim() is True
    assert s.active_client is True


def test_claim_second_client_rejected():
    s = Session(
        session_id="abc",
        video_path="x.mp4",
        total_frames=10,
        producer=FrameProducer(),
    )
    assert s.claim() is True
    assert s.claim() is False


def test_release_resets_client():
    s = Session(
        session_id="abc",
        video_path="x.mp4",
        total_frames=10,
        producer=FrameProducer(),
    )
    s.claim()
    s.release()
    assert s.active_client is False
