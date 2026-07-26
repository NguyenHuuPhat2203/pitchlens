"""Pipeline controller — wires services together and runs frame by frame."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import supervision as sv

from src.model.frame import ProcessedFrame
from src.repository.output_writer import OutputWriter
from src.repository.video_reader import VideoReader
from src.service.tracker import Tracker
from src.view.renderer import Renderer


class Pipeline:
    """Orchestrates the football tracking pipeline.

    Reads frames, runs YOLO detection + OC-SORT tracking,
    displays annotated frames, and saves output video + JSON.

    Uses a producer-consumer pattern so frame N+1 is read from disk
    while frame N is being processed on the GPU.
    """

    def __init__(self, config: dict) -> None:
        """Wire up services from configuration.

        Args:
            config: Parsed YAML configuration dictionary.
        """
        model_cfg = config["model"]
        pipeline_cfg = config.get("pipeline", {})
        self._display = pipeline_cfg.get("display", True)
        self._cpu_workers = pipeline_cfg.get("cpu_workers", 2)
        self._output_video = pipeline_cfg.get("output_video")
        self._output_json = pipeline_cfg.get("output_json")
        self._tracker = Tracker(
            model_name=model_cfg["player_detector"],
            tracker_config="ocsort.yaml",
            device=str(model_cfg.get("device", 0)),
        )
        self._renderer = Renderer()
        self._writer = OutputWriter()

    def run(self, source_path: str) -> list[np.ndarray]:
        """Run the pipeline on every frame, return annotated frames.

        Saves output video and JSON if configured in the pipeline
        section of the config file.

        Args:
            source_path: Path to the input video file.

        Returns:
            List of annotated BGR frames in display order.
        """
        reader = VideoReader(source_path)
        frame_iter = iter(reader)
        annotated_frames: dict[int, np.ndarray] = {}
        processed_frames: list[ProcessedFrame] = []
        total_frames = 0

        with (
            ThreadPoolExecutor(max_workers=1) as io_pool,
            ThreadPoolExecutor(max_workers=self._cpu_workers) as cpu_pool,
        ):
            read_future = io_pool.submit(self._read, frame_iter)

            while read_future is not None:
                frame_index, frame_image = read_future.result()
                if frame_image is None:
                    break

                total_frames += 1
                read_future = io_pool.submit(self._read, frame_iter)

                annotated, processed = self._process_frame_image(
                    frame_image, frame_index
                )
                annotated_frames[frame_index] = annotated
                processed_frames.append(processed)

                if self._display:
                    cv2.imshow("Football Tracker", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        cv2.destroyAllWindows()

        # Persist outputs if paths are configured.
        ordered_annotated = [annotated_frames[i] for i in range(total_frames)]
        if self._output_video:
            self._writer.write_video(
                ordered_annotated, self._output_video, fps=25.0,
            )
        if self._output_json:
            self._writer.write_json(processed_frames, self._output_json)

        return ordered_annotated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read(frame_iter):
        """Read the next frame.  Runs in the I/O thread."""
        try:
            frame = next(frame_iter)
            return frame.index, frame.image
        except StopIteration:
            return -1, None

    def _process_frame_image(self, image, frame_index):
        """Track players on a frame and return annotated + data."""
        tracks = self._tracker.track(image)
        if not tracks:
            processed = ProcessedFrame(
                frame_number=frame_index,
                timestamp=0.0,
            )
            return image.copy(), processed

        sv_detections = sv.Detections(
            xyxy=np.array([t.bbox for t in tracks]),
            confidence=np.array([t.confidence for t in tracks]),
            class_id=np.array([t.class_id for t in tracks]),
            tracker_id=np.array([t.track_id for t in tracks]),
        )
        annotated = self._renderer.render_tracks(image, sv_detections)
        processed = ProcessedFrame(
            frame_number=frame_index,
            timestamp=0.0,
            players=list(tracks),
        )
        return annotated, processed
