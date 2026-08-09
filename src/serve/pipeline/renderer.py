"""Renders annotations onto frames using Supervision."""

import numpy as np
import supervision as sv


class Renderer:
    """Draws tracking annotations — boxes, IDs, and trail paths."""

    def __init__(self) -> None:
        """Create annotators with sensible football defaults."""
        self._box_annotator = sv.BoxAnnotator(
            thickness=2,
            color=sv.Color(220, 40, 40),
        )
        self._label_annotator = sv.LabelAnnotator(
            text_color=sv.Color(255, 255, 255),
            color=sv.Color(220, 40, 40),
            text_padding=4,
            text_position=sv.Position.TOP_LEFT,
        )
        self._trace_annotator = sv.TraceAnnotator(
            thickness=2,
            trace_length=30,
        )

    def render_tracks(
        self,
        frame: np.ndarray,
        tracks: sv.Detections,
        labels: list[str] | None = None,
    ) -> np.ndarray:
        """Draw bounding boxes, track IDs, and movement trails.

        Args:
            frame: BGR image array (height, width, 3).
            tracks: Supervision Detections with ``tracker_id`` field.
            labels: Optional custom labels to show instead of track IDs.

        Returns:
            A new annotated BGR image array.
        """
        annotated = self._trace_annotator.annotate(
            scene=frame.copy(), detections=tracks
        )
        annotated = self._box_annotator.annotate(
            scene=annotated, detections=tracks
        )
        if labels is None:
            labels = [f"#{tid}" for tid in tracks.tracker_id.astype(int)]
        annotated = self._label_annotator.annotate(
            scene=annotated, detections=tracks, labels=labels
        )
        return annotated
