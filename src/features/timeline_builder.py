"""Conservative tactical events derived from completed frame facts."""

from __future__ import annotations

import math

from src.model.event import ShotEvent
from src.model.frame import ProcessedFrame
from src.model.tactical_event import TacticalEvent
from src.model.track import PlayerTrack


class TimelineBuilder:
    """Build possession transitions and shot events without a new model."""

    def __init__(
        self, max_ball_player_distance_px: float, min_possession_frames: int
    ) -> None:
        self._max_distance = max_ball_player_distance_px
        self._min_frames = min_possession_frames

    def build(self, frames: list[ProcessedFrame]) -> list[TacticalEvent]:
        owners = [self._nearest_owner(frame) for frame in frames]
        confirmed: list[tuple[int, str] | None] = [None] * len(frames)
        events: list[TacticalEvent] = []
        current: tuple[int, str] | None = None
        candidate: tuple[int, str] | None = None
        candidate_start = 0
        candidate_count = 0

        for index, owner in enumerate(owners):
            if owner is None:
                candidate = None
                candidate_count = 0
                confirmed[index] = current
                continue

            if owner == candidate:
                candidate_count += 1
            else:
                candidate = owner
                candidate_start = index
                candidate_count = 1

            if candidate_count >= self._min_frames and candidate != current:
                if current is not None:
                    events.append(
                        TacticalEvent(
                            frame_start=candidate_start,
                            frame_end=index,
                            type="pass",
                            actor_track_id=current[0],
                            team=current[1],
                            confidence=0.7,
                            pitch_position=None,
                            source="heuristic",
                        )
                    )
                current = candidate
            confirmed[index] = current

        for index, frame in enumerate(frames):
            for event in frame.events:
                if not isinstance(event, ShotEvent):
                    continue
                owner = confirmed[index] or owners[index]
                events.append(
                    TacticalEvent(
                        frame_start=event.frame_number,
                        frame_end=event.frame_number,
                        type="shot",
                        actor_track_id=owner[0] if owner else None,
                        team=owner[1] if owner else None,
                        confidence=0.8,
                        pitch_position=None,
                        source="heuristic",
                    )
                )

        return sorted(events, key=lambda event: (event.frame_start, event.frame_end))

    def _nearest_owner(
        self, frame: ProcessedFrame
    ) -> tuple[int, str] | None:
        if frame.ball is None or frame.ball.state != "in_play":
            return None
        bx, by = frame.ball.centroid
        nearest: tuple[float, PlayerTrack] | None = None
        for player in frame.players:
            if player.team is None:
                continue
            px = (player.bbox[0] + player.bbox[2]) / 2.0
            py = (player.bbox[1] + player.bbox[3]) / 2.0
            distance = math.hypot(bx - px, by - py)
            if distance <= self._max_distance and (
                nearest is None or distance < nearest[0]
            ):
                nearest = (distance, player)
        if nearest is None:
            return None
        player = nearest[1]
        return player.track_id, player.team
