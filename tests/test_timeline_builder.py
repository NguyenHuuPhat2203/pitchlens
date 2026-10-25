"""Tests for tactical timeline contracts and heuristic events."""

from src.config import AppConfig, VideoConfig
from src.features.timeline_builder import TimelineBuilder
from src.model.event import ShotEvent
from src.model.frame import ProcessedFrame
from src.model.tactical_event import TacticalEvent
from src.model.track import BallTrack, PlayerTrack


def test_tactical_event_keeps_explicit_evidence() -> None:
    event = TacticalEvent(
        frame_start=10,
        frame_end=12,
        type="pass",
        actor_track_id=4,
        team="A",
        confidence=0.75,
        pitch_position=None,
        source="heuristic",
    )
    assert event.actor_track_id == 4
    assert event.source == "heuristic"


def test_timeline_config_accepts_thresholds() -> None:
    config = AppConfig(
        video=VideoConfig(),
        timeline={
            "enabled": True,
            "max_ball_player_distance_px": 80.0,
            "min_possession_frames": 3,
            "output_json": "timeline.json",
            "benchmark_json": "benchmark.json",
        },
    )
    assert config.timeline.min_possession_frames == 3


def test_timeline_config_rejects_non_positive_thresholds() -> None:
    import pytest

    with pytest.raises(ValueError):
        AppConfig(video=VideoConfig(), timeline={"max_ball_player_distance_px": 0})
    with pytest.raises(ValueError):
        AppConfig(video=VideoConfig(), timeline={"min_possession_frames": 0})


def _frame(index: int, owner: int, owner_team: str, event=None) -> ProcessedFrame:
    players = [
        PlayerTrack(owner, (95.0, 95.0, 105.0, 105.0), 0, 0.9, owner_team),
        PlayerTrack(
            2 if owner == 1 else 1,
            (195.0, 95.0, 205.0, 105.0),
            0,
            0.9,
            "B" if owner_team == "A" else "A",
        ),
    ]
    ball = BallTrack(0, (95.0, 95.0, 105.0, 105.0), (100.0, 100.0), (0.0, 0.0), "in_play")
    return ProcessedFrame(index, index / 25.0, players, ball, [event] if event else [])


def test_emits_turnover_when_possession_changes_teams() -> None:
    frames = [
        _frame(0, 1, "A"),
        _frame(1, 1, "A"),
        _frame(2, 1, "A"),
        _frame(3, 2, "B"),
        _frame(4, 1, "A"),
        _frame(5, 2, "B"),
        _frame(6, 2, "B"),
        _frame(7, 2, "B"),
    ]
    timeline = TimelineBuilder(80.0, 3).build(frames)
    assert [(event.type, event.actor_track_id, event.team) for event in timeline] == [
        ("turnover", 1, "A"),
    ]


def test_emits_pass_when_possession_changes_within_team() -> None:
    frames = [_frame(i, 1, "A") for i in range(3)] + [
        _frame(i, 3, "A") for i in range(3, 6)
    ]
    timeline = TimelineBuilder(80.0, 3).build(frames)
    assert [(event.type, event.actor_track_id, event.team) for event in timeline] == [
        ("pass", 1, "A"),
    ]


def test_referee_and_negative_track_ids_never_own_possession() -> None:
    frames = []
    for index in range(4):
        frame = _frame(index, 1, "A")
        # Referee and an untracked (-1) player sit exactly on the ball.
        frame.players[0] = PlayerTrack(5, (99.0, 99.0, 101.0, 101.0), 0, 0.9, "referee")
        frame.players.append(PlayerTrack(-1, (99.0, 99.0, 101.0, 101.0), 0, 0.9, "B"))
        frames.append(frame)
    assert TimelineBuilder(80.0, 3).build(frames) == []


def test_does_not_invent_event_when_ball_is_unowned() -> None:
    frames = [_frame(index, 1, "A") for index in range(3)]
    frames[1].ball = BallTrack(
        0, (0.0, 0.0, 10.0, 10.0), (5.0, 5.0), (0.0, 0.0), "in_play"
    )
    assert TimelineBuilder(20.0, 3).build(frames) == []


def test_converts_existing_shot_to_tactical_event() -> None:
    frames = [_frame(0, 1, "A", ShotEvent(0, None, 42.0))]
    timeline = TimelineBuilder(80.0, 3).build(frames)
    assert timeline[0].type == "shot"
    assert timeline[0].actor_track_id == 1
