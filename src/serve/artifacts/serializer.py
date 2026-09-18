"""Serializes ProcessedFrame objects to JSON-safe dicts"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass

from src.model.frame import ProcessedFrame


class Serializer:
    """Converts pipeline output to the ML API item schema.

    Follows the item schema: every item is
    ``{"status": "success", "result": {...}}``.
    """

    def serialize(self, frame: ProcessedFrame) -> dict:
        """Convert a single ProcessedFrame to a JSON-safe dict.

        Args:
            frame: A ProcessedFrame from the pipeline.

        Returns:
            Dict ready for ``json.dump``.
        """
        ball = (
            asdict(frame.ball)
            if frame.ball is not None and is_dataclass(frame.ball)
            else frame.ball
        )
        events = [
            asdict(event) if is_dataclass(event) else event
            for event in frame.events
        ]
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
            "ball": ball,
            "events": events,
        }

    def serialize_item(self, frame: ProcessedFrame) -> dict:
        """Wrap a single frame in the ML API item schema.

        Args:
            frame: A ProcessedFrame from the pipeline.

        Returns:
            ``{"status": "success", "result": {...}}``.
        """
        return {
            "status": "success",
            "result": self.serialize(frame),
        }

    def serialize_all(self, frames: list[ProcessedFrame]) -> list[dict]:
        """Serialize frames as an array of items

        Batch responses are arrays preserving input order.
        Each item has its own ``status`` field.

        Args:
            frames: All processed frames from a pipeline run.

        Returns:
            List of ``{"status": "success", "result": {...}}`` dicts.
        """
        return [self.serialize_item(f) for f in frames]
