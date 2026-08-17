"""MJPEG stream helpers."""

from __future__ import annotations

from collections.abc import Iterator

from src.serve.adapt.session import Session

BOUNDARY = b"--frame\r\n"
CONTENT_TYPE = b"Content-Type: image/jpeg\r\n\r\n"


def format_frame(jpeg: bytes) -> bytes:
    """Wrap a JPEG in a multipart MJPEG part."""
    return (
        BOUNDARY
        + b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
        + CONTENT_TYPE
        + jpeg
        + b"\r\n"
    )


def mjpeg_generator(session: Session) -> Iterator[bytes]:
    """Yield MJPEG chunks until done or the producer is stopped."""
    yield format_frame(b"")
    while True:
        jpeg = session.producer.get(timeout=0.5)
        if jpeg is None:
            if session.producer.done.is_set() or session.producer.stop_flag.is_set():
                break
            continue
        yield format_frame(jpeg)
