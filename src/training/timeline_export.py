"""Export heuristic events as reviewable training candidates."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from src.model.tactical_event import TacticalEvent

_REVIEW_TEMPLATE = {
    "type": None,
    "actor_track_id": None,
    "team": None,
    "frame_start": None,
    "frame_end": None,
}


def export_candidates(
    events: Iterable[TacticalEvent],
    video_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Write one reviewable JSON object per heuristic event."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    video = str(video_path)
    with destination.open("w", encoding="utf-8") as fh:
        for event in events:
            event_data = asdict(event)
            record = {
                "candidate_id": (
                    f"{event.frame_start}-{event.frame_end}-{event.type}"
                ),
                "clip": {
                    "video_path": video,
                    "frame_start": event.frame_start,
                    "frame_end": event.frame_end,
                },
                "heuristic": event_data,
                "event": event.type,
                "review": dict(_REVIEW_TEMPLATE),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return destination
