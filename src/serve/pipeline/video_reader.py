"""Reads video files frame by frame without buffering everything into RAM."""

from collections.abc import Iterator
from pathlib import Path

import cv2

from src.model.frame import Frame


class VideoReader:
    """Yields Frame objects from a video file.

    Opens the file lazily on first iteration and releases it when the
    iterator is exhausted or garbage-collected.  Never loads all frames
    into memory at once.

    Usage:
        reader = VideoReader("data/match.mp4")
        for frame in reader:
            print(frame.index, frame.timestamp)
    """

    def __init__(self, path: str) -> None:
        """Store the video path and probe its frame rate.

        Args:
            path: Absolute or relative path to a video file.
        """
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video file not found: {self._path}")
        cap = cv2.VideoCapture(str(self._path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        #: Frames per second; 25 when the container reports none.
        self.fps = fps if fps > 0 else 25.0

    def __iter__(self) -> Iterator[Frame]:
        """Yield one Frame per video frame until the stream ends."""
        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self._path}")

        try:
            index = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                yield Frame(index=index, timestamp=index / self.fps, image=image)
                index += 1
        finally:
            cap.release()
