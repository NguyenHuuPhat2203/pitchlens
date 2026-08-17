"""Pipeline controller — wires services together and runs frame by frame."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import supervision as sv

from src.model.frame import ProcessedFrame
from src.serve.artifacts.output_writer import OutputWriter
from src.serve.pipeline.video_reader import VideoReader
from src.backbone.ball_detector import BallDetector
from src.serve.pipeline.ball_tracker import BallTracker
from src.serve.pipeline.synloc_calibrator import SynLocCalibrator
from src.serve.pipeline.shot_detector import ShotDetector
from src.serve.pipeline.speed_estimator import SpeedEstimator
from src.features.team_classifier import TeamClassifier
from src.serve.pipeline.tracker import Tracker
from src.serve.pipeline.renderer import Renderer


class Pipeline:
    """Orchestrates the football tracking pipeline.

    Reads frames, runs YOLO detection + OC-SORT tracking,
    displays annotated frames, and saves output video + JSON.

    Uses a producer-consumer pattern so frame N+1 is read from disk
    while frame N is being processed on the GPU.
    """

    def __init__(self, config) -> None:
        """Wire up services from configuration.

        Args:
            config: AppConfig from src.config (Pydantic model).
        """
        model_cfg = config.model
        pipeline_cfg = config.pipeline
        team_cfg = config.team_classifier
        self._display = pipeline_cfg.display
        self._cpu_workers = pipeline_cfg.cpu_workers
        self._output_video = pipeline_cfg.output_video
        self._output_json = pipeline_cfg.output_json

        device = model_cfg.device

        self._tracker = Tracker(
            model_name=model_cfg.player_detector,
            tracker_config="ocsort.yaml",
            device=device,
        )
        self._renderer = Renderer()
        self._writer = OutputWriter()

        # Team classification (fitted lazily during the first N frames).
        self._team_enabled = team_cfg.enabled
        self._team_clf = TeamClassifier() if self._team_enabled else None
        self._team_crops: list[np.ndarray] = []
        self._team_fitted = False
        self._team_fit_frames = team_cfg.fit_frames

        # Ball detection + tracking + shot detection.
        ball_cfg = config.ball_detector
        shot_cfg = config.shot_detector
        self._ball_enabled = ball_cfg.enabled
        self._ball_detector = BallDetector(
            model_name=ball_cfg.model,
            device=device,
        ) if self._ball_enabled else None
        self._ball_tracker = BallTracker(
            max_lost_frames=ball_cfg.max_lost_frames,
        ) if self._ball_enabled else None
        self._shot_detector = ShotDetector(
            angle_threshold=shot_cfg.angle_threshold,
            accel_threshold=shot_cfg.accel_threshold,
        ) if self._ball_enabled else None
        self._active_shot: tuple[int, float] | None = None

        # Pitch calibration + speed estimation.
        calib_cfg = config.calibration
        self._calib_enabled = calib_cfg.enabled
        self._calibrator = SynLocCalibrator() if self._calib_enabled else None
        if self._calibrator and calib_cfg.camera_dir:
            self._calibrator.load_camera(calib_cfg.camera_dir)
        self._speed_estimator = SpeedEstimator(fps=25.0)
        self._position_history: dict[int, list[tuple[float, float]]] = {}
        self._speed_window = calib_cfg.speed_window
        self._last_direction: dict[int, float] = {}  # frames to smooth over

    def run(self, source_path: str) -> list[np.ndarray]:
        """Run the pipeline on every frame, return annotated frames.

        If calibration is enabled, collects 4 pitch-landmark clicks
        on the first frame before processing begins.

        Args:
            source_path: Path to the input video file.

        Returns:
            List of annotated BGR frames in display order.
        """
        reader = VideoReader(source_path)
        frame_iter = iter(reader)

        # --- Calibration click phase (before main loop) ---
        if self._calib_enabled and not self._calibrator.has_camera:
            self._collect_calibration_clicks(frame_iter)

        annotated_frames: dict[int, np.ndarray] = {}
        processed_frames: list[ProcessedFrame] = []
        total_frames = 0

        with ThreadPoolExecutor(max_workers=1) as io_pool:
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

    def process_frame(self, image: np.ndarray, frame_index: int):
        """Public entry point for external callers (e.g. web adapter).

        Args:
            image: BGR image array.
            frame_index: Zero-based frame number.

        Returns:
            (annotated_image, processed_frame) tuple.
        """
        return self._process_frame_image(image, frame_index)

    @staticmethod
    def _read(frame_iter):
        """Read the next frame.  Runs in the I/O thread."""
        try:
            frame = next(frame_iter)
            return frame.index, frame.image
        except StopIteration:
            return -1, None

    def _process_frame_image(self, image, frame_index):
        """Track players, classify teams, and render annotations."""
        tracks = self._tracker.track(image)
        if not tracks:
            processed = ProcessedFrame(
                frame_number=frame_index,
                timestamp=0.0,
            )
            return image.copy(), processed

        # Team classification — collect crops during fit phase,
        # predict labels after fitting.
        labels: list[str] = []
        if self._team_enabled:
            if not self._team_fitted and frame_index < self._team_fit_frames:
                for t in tracks:
                    crop = self._crop_torso(image, t.bbox)
                    if crop is not None:
                        self._team_crops.append(crop)
            elif not self._team_fitted:
                if len(self._team_crops) >= 3:
                    self._team_clf.fit(self._team_crops)
                self._team_fitted = True

            if self._team_fitted and self._team_clf._is_fitted:
                for t in tracks:
                    crop = self._crop_torso(image, t.bbox)
                    t.team = (
                        self._team_clf.predict(crop) if crop is not None
                        else None
                    )

        for t in tracks:
            team_str = f" {t.team}" if t.team else ""
            speed_str = f" {t.speed_kmh:.0f}km/h" if t.speed_kmh else ""
            labels.append(f"#{t.track_id}{team_str}{speed_str}")

        sv_detections = sv.Detections(
            xyxy=np.array([t.bbox for t in tracks]),
            confidence=np.array([t.confidence for t in tracks]),
            class_id=np.array([t.class_id for t in tracks]),
            tracker_id=np.array([t.track_id for t in tracks]),
        )
        annotated = self._renderer.render_tracks(image, sv_detections, labels)

        # Direction arrows (always on, pixel-space) + speed (homography only).
        for t in tracks:
            cx = (t.bbox[0] + t.bbox[2]) / 2
            cy = (t.bbox[1] + t.bbox[3]) / 2
            curr_pos = (cx, cy)

            hist = self._position_history.setdefault(t.track_id, [])
            hist.append(curr_pos)
            if len(hist) > self._speed_window:
                hist.pop(0)

            if len(hist) >= 2:
                prev = hist[0]
                dx = curr_pos[0] - prev[0]
                dy = curr_pos[1] - prev[1]
                dist = float(np.hypot(dx, dy))

                if dist >= 10:
                    direction = float(np.degrees(np.arctan2(dy, dx)))
                    self._last_direction[t.track_id] = direction
                else:
                    direction = self._last_direction.get(t.track_id, 0.0)

                t.direction_deg = round(direction, 1)
                self._draw_arrow(annotated, curr_pos, direction)

                if self._calib_enabled and self._calibrator.is_calibrated:
                    speed, _ = self._speed_estimator.estimate(
                        prev, curr_pos, self._calibrator.homography,
                    )
                    t.speed_kmh = round(speed, 1)

        # Ball detection + tracking + shot detection.
        ball_track = None
        events = []
        if self._ball_enabled:
            ball_det = self._ball_detector.detect_ball(image)
            ball_track = self._ball_tracker.update(ball_det)
            shot = self._shot_detector.update(ball_track)
            if shot is not None:
                self._active_shot = (frame_index, shot.ball_speed_kmh)
                events.append(shot)

        # Draw ball on the frame.
        if ball_track is not None and ball_track.state == "in_play":
            cx, cy = int(ball_track.centroid[0]), int(ball_track.centroid[1])
            cv2.circle(annotated, (cx, cy), 6, (0, 255, 0), -1)
            cv2.circle(annotated, (cx, cy), 9, (0, 255, 0), 2)

        # Flash "SHOT!" overlay.
        if self._active_shot is not None:
            shot_frame, shot_speed = self._active_shot
            if frame_index - shot_frame < 25:  # ~1 second at 25 fps
                cv2.putText(
                    annotated,
                    f"SHOT! {shot_speed} km/h",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 100, 255),
                    3,
                )
            else:
                self._active_shot = None

        processed = ProcessedFrame(
            frame_number=frame_index,
            timestamp=0.0,
            players=list(tracks),
            ball=ball_track,
            events=events,
        )
        return annotated, processed

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

    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------

    def _collect_calibration_clicks(self, frame_iter):
        """Show the first frame and collect 4 pitch-landmark clicks."""
        try:
            first = next(frame_iter)
        except StopIteration:
            return

        clicks: list[tuple[int, int]] = []
        display = first.image.copy()

        def _on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicks.append((x, y))

        cv2.imshow("Football Tracker", display)
        cv2.setMouseCallback("Football Tracker", _on_click)

        while len(clicks) < 4:
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                break
            display = first.image.copy()
            # Instructions overlay.
            cv2.putText(
                display,
                "Click 4 corners: TL, TR, BR, BL. Q=skip",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            )
            for i, (cx, cy) in enumerate(clicks):
                cv2.circle(display, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(
                    display, str(i + 1), (cx + 8, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
            cv2.imshow("Football Tracker", display)

        cv2.setMouseCallback("Football Tracker", lambda *a: None)

        if len(clicks) >= 4:
            image_pts = np.array(clicks[:4], dtype=np.float32)
            pitch_pts = np.array(
                [[0, 0], [105, 0], [105, 68], [0, 68]], dtype=np.float32,
            )
            self._calibrator.fit(image_pts, pitch_pts)

    @staticmethod
    def _draw_arrow(image, origin, direction_deg, length=30, color=(0, 255, 0)):
        """Draw a direction arrow on the image."""
        rad = np.radians(direction_deg)
        dx = int(length * np.cos(rad))
        dy = int(length * np.sin(rad))
        ox, oy = int(origin[0]), int(origin[1])
        cv2.arrowedLine(image, (ox, oy), (ox + dx, oy + dy), color, 2)
