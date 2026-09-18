"""Consumer-facing tactical timeline events."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TacticalEvent:
    """One action event in an analyzed football match."""

    frame_start: int
    frame_end: int
    type: str
    actor_track_id: int | None
    team: str | None
    confidence: float
    pitch_position: tuple[float, float] | None
    source: str
