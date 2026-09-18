"""Tests for reviewable tactical timeline candidates."""

import json

from src.model.tactical_event import TacticalEvent
from src.training.timeline_export import export_candidates


def test_export_candidate_writes_reviewer_label_template(tmp_path) -> None:
    event = TacticalEvent(10, 12, "pass", 4, "A", 0.7, None, "heuristic")
    path = export_candidates(
        [event], "matches/sample.mp4", tmp_path / "candidates.jsonl"
    )
    candidate = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert candidate["event"] == "pass"
    assert candidate["clip"] == {
        "video_path": "matches/sample.mp4", "frame_start": 10, "frame_end": 12,
    }
    assert candidate["review"] == {
        "type": None, "actor_track_id": None, "team": None,
        "frame_start": None, "frame_end": None,
    }


def test_export_candidate_includes_unknown_false_positive(tmp_path) -> None:
    event = TacticalEvent(20, 20, "unknown", None, None, 0.0, None, "heuristic")
    path = export_candidates([event], "sample.mp4", tmp_path / "candidates.jsonl")
    candidate = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert candidate["event"] == "unknown"
