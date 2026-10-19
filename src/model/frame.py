"""A single video frame with metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.model.track import PlayerTrack


@dataclass
class Frame:
    """One frame extracted from a video source.

    Attributes:
        index: Zero-based frame number in the video sequence.
        timestamp: Elapsed seconds from video start.
        image: BGR image array (height, width, 3).
    """

    index: int
    timestamp: float
    image: np.ndarray


@dataclass
class ProcessedFrame:
    """The output of processing one video frame.

    Aggregates all tracking results and events for a single frame,
    ready for serialization or downstream consumption.

    Attributes:
        frame_number: Zero-based frame index.
        timestamp: Elapsed seconds from video start.
        players: List of tracked players visible in this frame.
        ball: Tracked ball position, or None if not detected.
        events: Events that occurred on this frame (e.g. shot).
    """

    frame_number: int
    timestamp: float
    players: list[PlayerTrack] = field(default_factory=list)
    ball: object | None = None
    events: list = field(default_factory=list)


@dataclass
class RunResult:
    """Outcome of one ``Pipeline.run()`` call.

    Annotated frames are displayed or encoded as they are produced and
    never returned, so long videos do not accumulate in memory.

    Attributes:
        frame_count: Frames processed.
    """

    frame_count: int
