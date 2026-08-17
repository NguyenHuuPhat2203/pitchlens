"""Session object — one per analysis request."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from src.serve.adapt.producer import FrameProducer


@dataclass
class Session:
    """In-memory state for one video analysis session.

    One client per session.  ``claim`` atomically reserves the session
    for a client; ``release`` frees it.
    """

    session_id: str
    video_path: str
    total_frames: int
    producer: FrameProducer
    thread: threading.Thread | None = None
    active_client: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def claim(self) -> bool:
        """Atomically reserve the session.  False if already taken."""
        with self._lock:
            if self.active_client:
                return False
            self.active_client = True
            return True

    def release(self) -> None:
        """Free the session for another client."""
        with self._lock:
            self.active_client = False
