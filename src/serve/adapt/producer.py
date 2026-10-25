"""Thread-safe frame buffer between pipeline thread and stream generator."""

from __future__ import annotations

import queue
import threading


class FrameProducer:
    """Bounded frame queue with stop/done flags.

    The pipeline thread calls ``put``; the MJPEG generator calls
    ``get``.  ``put`` blocks (never drops) until space or stop.
    """

    def __init__(self, maxsize: int = 100) -> None:
        """Create the producer.

        Args:
            maxsize: Maximum number of frames buffered in memory.
        """
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.stop_flag = threading.Event()
        self.done = threading.Event()
        self.processed = 0

    def put(self, jpeg: bytes) -> bool:
        """Enqueue a frame, blocking while full.  False if stopped first."""
        while not self.stop_flag.is_set():
            try:
                self._queue.put(jpeg, timeout=0.1)
            except queue.Full:
                continue
            self.processed += 1
            return True
        return False

    def get(self, timeout: float = 0.5) -> bytes | None:
        """Dequeue a frame.  Returns None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Signal the pipeline to halt at the next frame boundary."""
        self.stop_flag.set()

    def mark_done(self) -> None:
        """Signal that the video has been fully processed."""
        self.done.set()
