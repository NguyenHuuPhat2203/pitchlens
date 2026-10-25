"""Tests for MJPEG stream helpers."""

from src.serve.adapt.producer import FrameProducer
from src.serve.adapt.session import Session
from src.serve.adapt.streaming import format_frame, mjpeg_generator


def test_format_frame_has_boundary_and_length():
    jpeg = b"\xff\xd8fakejpeg"
    out = format_frame(jpeg)
    assert out.startswith(b"--frame\r\n")
    assert b"Content-Length: " + str(len(jpeg)).encode() in out
    assert b"Content-Type: image/jpeg\r\n" in out
    assert out.endswith(jpeg + b"\r\n")


def test_generator_yields_frames_until_done():
    session = Session(
        session_id="a",
        video_path="x",
        total_frames=2,
        producer=FrameProducer(maxsize=10),
    )
    session.producer.put(b"f1")
    session.producer.put(b"f2")
    session.producer.mark_done()

    gen = mjpeg_generator(session)
    chunks = list(gen)
    joined = b"".join(chunks)
    assert b"f1" in joined
    assert b"f2" in joined


def test_generator_drains_frames_put_after_done_checked():
    session = Session(
        session_id="a", video_path="x", total_frames=1,
        producer=FrameProducer(maxsize=10),
    )
    real_get = session.producer.get
    calls = []

    def racy_get(timeout=0.5):
        calls.append(timeout)
        if len(calls) == 1:  # empty at first poll, then pipeline finishes
            session.producer.put(b"late")
            session.producer.mark_done()
            return None
        return real_get(timeout)

    session.producer.get = racy_get
    assert b"late" in b"".join(mjpeg_generator(session))
