"""Tests for FrameProducer — thread-safe frame buffer."""

import threading
import time

from src.serve.adapt.producer import FrameProducer


def test_put_and_get_roundtrip():
    p = FrameProducer(maxsize=10)
    p.put(b"frame1")
    assert p.get(timeout=0.1) == b"frame1"


def test_put_blocks_when_full_until_consumer_drains():
    p = FrameProducer(maxsize=2)
    p.put(b"a")
    p.put(b"b")
    threading.Timer(0.2, p.get).start()
    start = time.monotonic()
    assert p.put(b"c") is True
    assert time.monotonic() - start >= 0.15
    assert p.processed == 3
    assert [p.get(0.1), p.get(0.1)] == [b"b", b"c"]


def test_put_returns_false_when_stopped_while_full():
    p = FrameProducer(maxsize=1)
    p.put(b"a")
    threading.Timer(0.1, p.stop).start()
    assert p.put(b"b") is False
    assert p.processed == 1


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
