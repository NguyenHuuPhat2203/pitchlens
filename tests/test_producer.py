"""Tests for FrameProducer — thread-safe frame buffer."""

import time

from src.serve.adapt.producer import FrameProducer


def test_put_and_get_roundtrip():
    p = FrameProducer(maxsize=10)
    p.put(b"frame1")
    assert p.get(timeout=0.1) == b"frame1"


def test_put_blocks_when_full_then_times_out():
    p = FrameProducer(maxsize=2)
    p.put(b"a")
    p.put(b"b")
    start = time.monotonic()
    p.put(b"c", timeout=0.2)
    elapsed = time.monotonic() - start
    assert elapsed >= 0.15


def test_get_times_out_when_empty():
    p = FrameProducer(maxsize=2)
    start = time.monotonic()
    assert p.get(timeout=0.2) is None
    assert time.monotonic() - start >= 0.15


def test_stop_flag_set():
    p = FrameProducer()
    p.stop()
    assert p.stop_flag.is_set()


def test_done_flag_set():
    p = FrameProducer()
    p.mark_done()
    assert p.done.is_set()


def test_processed_increments():
    p = FrameProducer()
    p.put(b"a")
    p.put(b"b")
    assert p.processed == 2
