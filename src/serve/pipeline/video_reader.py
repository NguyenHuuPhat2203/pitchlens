"""Reads video files frame by frame without buffering everything into RAM."""

from pathlib import Path
from typing import Iterator

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
        """Store the video path; the file is opened lazily.

        Args:
            path: Absolute or relative path to a video file.
        """
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video file not found: {self._path}")

    def __iter__(self) -> Iterator[Frame]:
        """Yield one Frame per video frame until the stream ends."""
        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self._path}")

        try:
            index = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25.0  # sensible default for broadcast footage

            while True:
                ok, image = cap.read()
                if not ok:
                    break
                timestamp = index / fps
                yield Frame(index=index, timestamp=timestamp, image=image)
                index += 1
        finally:
            cap.release()
