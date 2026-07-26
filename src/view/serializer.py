"""Serializes ProcessedFrame objects to JSON-safe dicts with envelope."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from src.model.frame import ProcessedFrame


class Serializer:
    """Converts pipeline output to the JSON envelope format.

    Follows RULES.md: every response is wrapped in
    ``{"success": true, "data": {...}, "meta": {...}}``.
    """

    def serialize(self, frame: ProcessedFrame) -> dict:
        """Convert a single ProcessedFrame to a JSON-safe dict.

        Args:
            frame: A ProcessedFrame from the pipeline.

        Returns:
            Dict ready for ``json.dump``.
        """
        return {
            "frame_number": frame.frame_number,
            "timestamp": frame.timestamp,
            "players": [
                {
                    "track_id": p.track_id,
                    "bbox": list(p.bbox),
                    "class_id": p.class_id,
                    "confidence": p.confidence,
                    "team": p.team,
                    "speed_kmh": p.speed_kmh,
                    "direction_deg": p.direction_deg,
                }
                for p in frame.players
            ],
            "ball": None,
            "events": [],
        }

    def serialize_all(self, frames: list[ProcessedFrame]) -> dict:
        """Serialize a list of ProcessedFrames into an envelope.

        Args:
            frames: All processed frames from a pipeline run.

        Returns:
            Envelope dict with ``success``, ``data``, and ``meta`` keys.
        """
        return {
            "success": True,
            "data": {
                "frames": [self.serialize(f) for f in frames],
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": str(uuid.uuid4()),
                "total_frames": len(frames),
            },
        }
