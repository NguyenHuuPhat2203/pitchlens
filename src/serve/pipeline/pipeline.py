"""Pipeline controller — wires services together and runs frame by frame."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import supervision as sv

from src.backbone.ball_detector import BallDetector
from src.backbone.gpu_utils import execution_summary, fp16_model_path
from src.features.team_classifier import TeamClassifier
from src.features.timeline_builder import TimelineBuilder
from src.logger import get_logger
from src.model.frame import ProcessedFrame, RunResult
from src.serve.artifacts.benchmark import StageTimer
from src.serve.artifacts.output_writer import OutputWriter
from src.serve.pipeline.auto_calibrator import CalibrationProcess
from src.serve.pipeline.ball_tracker import BallTracker
from src.serve.pipeline.renderer import Renderer
from src.serve.pipeline.shot_detector import ShotDetector
from src.serve.pipeline.synloc_calibrator import SynLocCalibrator
from src.serve.pipeline.tracker import Tracker
from src.serve.pipeline.video_reader import VideoReader

_log = get_logger(__name__)

_DEFAULT_FPS = 25.0  # until reset(fps) or run() supplies the video's rate


class Pipeline:
    """Orchestrates the football tracking pipeline.

    Reads frames, runs YOLO detection + OC-SORT tracking, optionally
    displays and encodes annotated frames, and saves JSON outputs.

    Frame N+1 is decoded on an I/O thread while frame N is processed.
    Pitch calibration runs per frame in its own process, started as soon as the
    frame is decoded: found and tracked from the painted lines
    (``AutoCalibrator``), or a fixed SynLoc camera model.
    """

    def __init__(self, config) -> None:
        """Wire up services from configuration.

        Args:
            config: AppConfig from src.config (Pydantic model).
        """
        model_cfg = config.model
        pipeline_cfg = config.pipeline
        self._display = pipeline_cfg.display
        self._output_video = pipeline_cfg.output_video
        self._output_json = pipeline_cfg.output_json

        device = model_cfg.device
        half = model_cfg.half

        def model_path(name: str) -> str:
            return fp16_model_path(name) if half else name

        self._tracker = Tracker(
            model_name=model_path(model_cfg.player_detector),
            device=device,
            half=half,
            conf=model_cfg.detection_conf,
            tensorrt=model_cfg.tensorrt,
        )
        self._renderer = Renderer()
        self._writer = OutputWriter()

        self._team_enabled = config.team_classifier.enabled
        self._team_fit_frames = config.team_classifier.fit_frames

        self._ball_cfg = config.ball_detector
        self._shot_cfg = config.shot_detector
        self._ball_enabled = self._ball_cfg.enabled
        self._ball_detector = BallDetector(
            model_name=model_path(self._ball_cfg.model),
            device=device,
            half=half,
            conf=model_cfg.detection_conf,
            tensorrt=model_cfg.tensorrt,
        ) if self._ball_enabled else None
        self._ball_pool = ThreadPoolExecutor(max_workers=1) if self._ball_enabled else None

        calib_cfg = config.calibration
        self._calibrator: CalibrationProcess | SynLocCalibrator | None = None
        if calib_cfg.enabled:
            if calib_cfg.camera_dir:
                self._calibrator = SynLocCalibrator()
                self._calibrator.load_camera(calib_cfg.camera_dir)
            else:
                self._calibrator = CalibrationProcess()
        self._speed_window = calib_cfg.speed_window

        timeline_cfg = config.timeline
        self._timeline_enabled = timeline_cfg.enabled
        self._timeline_builder = TimelineBuilder(
            timeline_cfg.max_ball_player_distance_px,
            timeline_cfg.min_possession_frames,
        )
        self._benchmark_output = timeline_cfg.benchmark_json
        self._timeline_output = timeline_cfg.output_json
        self.reset(_DEFAULT_FPS)
        _log.info("pipeline_device", **execution_summary(device, half))

    def reset(self, fps: float) -> None:
        """Forget everything learned from a previous video.

        ``run()`` calls this itself; ``process_frame()`` callers call it
        before each new video with that video's frame rate.
        """
        self._fps = fps
        self._tracker.reset()
        self._team_clf = TeamClassifier() if self._team_enabled else None
        self._team_crops: list[np.ndarray] = []
        self._ball_tracker = BallTracker(
            max_lost_frames=self._ball_cfg.max_lost_frames,
            gate_px=self._ball_cfg.gate_px,
        ) if self._ball_enabled else None
        self._shot_detector = ShotDetector(
            angle_threshold=self._shot_cfg.angle_threshold,
            accel_threshold=self._shot_cfg.accel_threshold,
        ) if self._ball_enabled else None
        self._active_shot: tuple[int, float | None] | None = None
        # Per track: (timestamp, foot pixel, foot on the pitch or None) samples,
        # newest last. The pitch position is fixed when seen: the camera may move.
        self._position_history: dict[
            int, list[tuple[float, tuple[float, float], tuple[float, float] | None]]
        ] = {}
        self._last_direction: dict[int, float] = {}
        self._stage_timer = StageTimer()
        if isinstance(self._calibrator, CalibrationProcess):
            self._calibrator.reset()

    def run(self, source_path: str) -> RunResult:
        """Run the pipeline on every frame of a video.

        Annotated frames are shown and/or encoded as they are produced;
        none are kept in memory.
        """
        reader = VideoReader(source_path)
        self.reset(reader.fps)
        frame_iter = iter(reader)

        render = self._display or self._output_video is not None
        processed_frames: list[ProcessedFrame] = []
        total_frames = 0
        video = None

        try:
            with ThreadPoolExecutor(max_workers=1) as io_pool:
                def read():
                    return io_pool.submit(
                        self._stage_timer.measure, "decode", self._next_frame, frame_iter
                    )

                read_future = read()
                while (item := read_future.result())[0] is not None:
                    frame, calib_future = item
                    read_future = read()
                    total_frames += 1
                    annotated, processed = self._process_frame_image(
                        frame.image, frame.index, frame.timestamp, render=render,
                        calib_future=calib_future,
                    )
                    processed_frames.append(processed)

                    if self._output_video:
                        if video is None:
                            video = self._writer.open_video(
                                self._output_video, self._fps, annotated.shape[:2]
                            )
                        self._stage_timer.measure("encode", video.write, annotated)
                    if self._display:
                        cv2.imshow("Football Tracker", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        finally:
            if video is not None:
                video.release()
            if self._display:
                cv2.destroyAllWindows()

        if self._output_json:
            self._stage_timer.measure(
                "json_write", self._writer.write_json,
                processed_frames, self._output_json
            )
        if self._timeline_enabled and self._timeline_output:
            events = self._stage_timer.measure(
                "timeline_build",
                self._timeline_builder.build,
                processed_frames,
            )
            self._stage_timer.measure(
                "timeline_write",
                self._writer.write_timeline,
                events,
                self._timeline_output,
            )
        if self._benchmark_output:
            self._writer.write_benchmark(
                self._stage_timer.report(), self._benchmark_output
            )

        return RunResult(frame_count=total_frames)

    def process_frame(
        self, image: np.ndarray, frame_index: int, timestamp: float | None = None
    ):
        """Public entry point for external callers (e.g. web adapter).

        Args:
            image: BGR image array.
            frame_index: Zero-based frame number.
            timestamp: Seconds from video start; ``frame_index / fps`` if
                omitted (fps from the last ``reset()``/``run()``, else 25).

        Returns:
            (annotated_image, processed_frame) tuple.
        """
        if timestamp is None:
            timestamp = frame_index / self._fps
        return self._process_frame_image(image, frame_index, timestamp, render=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_frame(self, frame_iter):
        """Decode the next frame and start calibrating it: both overlap the current frame."""
        frame = next(frame_iter, None)
        calib_future = None
        if frame is not None and isinstance(self._calibrator, CalibrationProcess):
            calib_future = self._calibrator.submit(frame.image, frame.index)
        return frame, calib_future

    def _process_frame_image(
        self, image, frame_index, timestamp, render=True, calib_future=None
    ):
        """Track players and ball, classify teams, optionally render."""
        ball_future = None
        if self._ball_enabled and frame_index % 2 == 0:
            # Overlap the ball model with player tracking: -3 ms/frame with
            # identical output (research/experiments/h4-parallel-ball).
            ball_future = self._ball_pool.submit(
                self._stage_timer.measure,
                "ball_detect", self._ball_detector.detect_balls, image,
            )
        if calib_future is None and isinstance(self._calibrator, CalibrationProcess):
            calib_future = self._calibrator.submit(image, frame_index)
        tracks = self._stage_timer.measure(
            "player_track", self._tracker.track, image
        )
        self._classify_teams(image, tracks, frame_index)
        to_pitch = self._frame_transform(calib_future)
        arrows = self._update_motion(tracks, timestamp, to_pitch)
        ball_track, events = self._update_ball(ball_future, frame_index, to_pitch)

        annotated = image
        if render:
            if tracks:
                labels = [
                    f"#{t.track_id}"
                    + (f" {t.team}" if t.team else "")
                    + (f" {t.speed_kmh:.0f}km/h" if t.speed_kmh is not None else "")
                    for t in tracks
                ]
                sv_detections = sv.Detections(
                    xyxy=np.array([t.bbox for t in tracks]),
                    confidence=np.array([t.confidence for t in tracks]),
                    class_id=np.array([t.class_id for t in tracks]),
                    tracker_id=np.array([t.track_id for t in tracks]),
                )
                annotated = self._stage_timer.measure(
                    "render", self._renderer.render_tracks,
                    image, sv_detections, labels
                )  # returns a copy
            else:
                annotated = image.copy()
            for origin, direction in arrows:
                self._draw_arrow(annotated, origin, direction)
            self._draw_ball(annotated, ball_track, frame_index)

        processed = ProcessedFrame(
            frame_number=frame_index,
            timestamp=timestamp,
            players=list(tracks),
            ball=ball_track,
            events=events,
        )
        return annotated, processed

    def _classify_teams(self, image, tracks, frame_index) -> None:
        """Collect torso crops, fit once enough exist, then label tracks."""
        if not self._team_enabled:
            return
        clf = self._team_clf
        if not clf.is_fitted:
            self._team_crops.extend(
                c for t in tracks if (c := self._crop_torso(image, t.bbox)) is not None
            )
            if frame_index + 1 < self._team_fit_frames:
                return
            try:
                clf.fit(self._team_crops)
            except ValueError:
                return  # too few crops yet; keep collecting
            self._team_crops = []
        for t in tracks:
            crop = self._crop_torso(image, t.bbox)
            t.team = clf.predict(crop) if crop is not None else None

    def _frame_transform(self, calib_future):
        """This frame's pixels -> pitch metres, or None when the pitch is not registered."""
        if self._calibrator is None:
            return None
        if calib_future is None:  # fixed SynLoc camera model
            return self._calibrator.transform
        H = self._stage_timer.measure("calibrate_wait", calib_future.result)
        if H is None:
            return None
        return lambda pts: cv2.perspectiveTransform(
            np.asarray(pts, np.float32).reshape(-1, 1, 2), H
        ).reshape(-1, 2)

    def _update_motion(self, tracks, timestamp, to_pitch):
        """Set direction and speed per track; return arrows to draw."""
        arrows = []
        for t in tracks:
            # Foot point: the homography maps the ground plane.
            foot = ((t.bbox[0] + t.bbox[2]) / 2, t.bbox[3])
            ground = tuple(to_pitch([foot])[0]) if to_pitch is not None else None
            hist = self._position_history.setdefault(t.track_id, [])
            hist.append((timestamp, foot, ground))
            if len(hist) > self._speed_window:
                hist.pop(0)
            if len(hist) < 2:
                continue

            t0, start, start_ground = hist[0]
            dx, dy = foot[0] - start[0], foot[1] - start[1]
            if np.hypot(dx, dy) >= 10:
                self._last_direction[t.track_id] = float(np.degrees(np.arctan2(dy, dx)))
            direction = self._last_direction.get(t.track_id, 0.0)
            t.direction_deg = round(direction, 1)
            arrows.append((foot, direction))

            if ground is not None and start_ground is not None and timestamp > t0:
                t.speed_kmh = round(_kmh(start_ground, ground, timestamp - t0), 1)
        return arrows

    def _update_ball(self, ball_future, frame_index, to_pitch):
        """Track the ball from this frame's detections (if any), detect shots."""
        if not self._ball_enabled:
            return None, []
        candidates = ball_future.result() if ball_future is not None else []
        ball_track = self._ball_tracker.update(candidates)
        shot = self._shot_detector.update(ball_track, frame_index, self._fps)
        if shot is None:
            return ball_track, []
        if to_pitch is not None:
            # Velocity is px/frame: one frame of travel, projected to metres.
            cx, cy = ball_track.centroid
            vx, vy = ball_track.velocity
            start, end = to_pitch([(cx, cy), (cx + vx, cy + vy)])
            shot.ball_speed_kmh = round(_kmh(start, end, 1 / self._fps), 1)
        shot.timestamp = frame_index / self._fps
        self._active_shot = (frame_index, shot.ball_speed_kmh)
        return ball_track, [shot]

    def _draw_ball(self, annotated, ball_track, frame_index) -> None:
        """Draw the ball and a one-second SHOT! banner after a shot."""
        if ball_track is not None and ball_track.state == "in_play":
            cx, cy = int(ball_track.centroid[0]), int(ball_track.centroid[1])
            cv2.circle(annotated, (cx, cy), 6, (0, 255, 0), -1)
            cv2.circle(annotated, (cx, cy), 9, (0, 255, 0), 2)
        if self._active_shot is None:
            return
        shot_frame, shot_speed = self._active_shot
        if frame_index - shot_frame >= self._fps:
            self._active_shot = None
            return
        text = "SHOT!" if shot_speed is None else f"SHOT! {shot_speed} km/h"
        cv2.putText(
            annotated, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 3
        )

    @staticmethod
    def _crop_torso(image, bbox):
        """Extract the jersey region from a player bounding box."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, x2 = max(0, x1), min(image.shape[1], x2)
        y1, y2 = max(0, y1), min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        # Middle third — the jersey area, avoiding head and legs.
        h = y2 - y1
        jersey_y1 = y1 + h // 3
        jersey_y2 = y1 + 2 * h // 3
        if jersey_y2 <= jersey_y1:
            return None
        return image[jersey_y1:jersey_y2, x1:x2]

    @staticmethod
    def _draw_arrow(image, origin, direction_deg, length=30, color=(0, 255, 0)):
        """Draw a direction arrow on the image."""
        rad = np.radians(direction_deg)
        dx = int(length * np.cos(rad))
        dy = int(length * np.sin(rad))
        ox, oy = int(origin[0]), int(origin[1])
        cv2.arrowedLine(image, (ox, oy), (ox + dx, oy + dy), color, 2)


def _kmh(start, end, dt: float) -> float:
    """Ground speed between two pitch positions (metres) dt seconds apart."""
    return math.hypot(end[0] - start[0], end[1] - start[1]) / dt * 3.6
