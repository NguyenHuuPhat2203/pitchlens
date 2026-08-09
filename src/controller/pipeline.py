"""Pipeline controller — wires services together and runs frame by frame."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import supervision as sv
import torch

from src.model.frame import ProcessedFrame
from src.repository.output_writer import OutputWriter
from src.repository.video_reader import VideoReader
from src.service.ball_detector import BallDetector
from src.service.ball_tracker import BallTracker
from src.service.calibrator import Calibrator
from src.service.shot_detector import ShotDetector
from src.service.speed_estimator import SpeedEstimator
from src.service.team_classifier import TeamClassifier
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
        team_cfg = config.get("team_classifier", {})
        self._display = pipeline_cfg.get("display", True)
        self._cpu_workers = pipeline_cfg.get("cpu_workers", 2)
        self._output_video = pipeline_cfg.get("output_video")
        self._output_json = pipeline_cfg.get("output_json")

        device = self._resolve_device(model_cfg.get("device", "auto"))

        self._tracker = Tracker(
            model_name=model_cfg["player_detector"],
            tracker_config="ocsort.yaml",
            device=device,
        )
        self._renderer = Renderer()
        self._writer = OutputWriter()

        # Team classification (fitted lazily during the first N frames).
        self._team_enabled = team_cfg.get("enabled", True)
        self._team_clf = TeamClassifier() if self._team_enabled else None
        self._team_crops: list[np.ndarray] = []
        self._team_fitted = False
        self._team_fit_frames = team_cfg.get("fit_frames", 30)

        # Ball detection + tracking + shot detection.
        ball_cfg = config.get("ball_detector", {})
        shot_cfg = config.get("shot_detector", {})
        self._ball_enabled = ball_cfg.get("enabled", True)
        self._ball_detector = BallDetector(
            model_name=ball_cfg.get("model", model_cfg["player_detector"]),
            device=device,
        ) if self._ball_enabled else None
        self._ball_tracker = BallTracker(
            max_lost_frames=ball_cfg.get("max_lost_frames", 15),
        ) if self._ball_enabled else None
        self._shot_detector = ShotDetector(
            angle_threshold=shot_cfg.get("angle_threshold", 30.0),
            accel_threshold=shot_cfg.get("accel_threshold", 50.0),
        ) if self._ball_enabled else None
        self._active_shot: tuple[int, float] | None = None  # (frame, speed)

        # Pitch calibration + speed estimation.
        calib_cfg = config.get("calibration", {})
        self._calib_enabled = calib_cfg.get("enabled", True)
        self._calibrator = Calibrator() if self._calib_enabled else None
        self._speed_estimator = SpeedEstimator(fps=25.0)
        self._position_history: dict[int, list[tuple[float, float]]] = {}
        self._speed_window = calib_cfg.get("speed_window", 5)  # frames to smooth over

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
        if self._calib_enabled:
            self._collect_calibration_clicks(frame_iter)

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

        # Speed estimation via homography (smoothed over N frames).
        if self._calib_enabled and self._calibrator.is_calibrated:
            for t in tracks:
                cx = (t.bbox[0] + t.bbox[2]) / 2
                cy = (t.bbox[1] + t.bbox[3]) / 2
                curr_pos = (cx, cy)

                hist = self._position_history.setdefault(t.track_id, [])
                hist.append(curr_pos)
                if len(hist) > self._speed_window:
                    hist.pop(0)

                if len(hist) >= 2:
                    prev = hist[0]  # position N frames ago
                    speed, direction = self._speed_estimator.estimate(
                        prev, curr_pos, self._calibrator.homography,
                    )
                    t.speed_kmh = round(speed, 1)
                    t.direction_deg = round(direction, 1)
                    self._draw_arrow(annotated, curr_pos, direction)

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
        """Extract the torso region from a player bounding box."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, x2 = max(0, x1), min(image.shape[1], x2)
        y1, y2 = max(0, y1), min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        torso_y1 = y1 + (y2 - y1) // 2  # lower half = torso
        if torso_y1 >= y2:
            return None
        return image[torso_y1:y2, x1:x2]

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

    @staticmethod
    def _resolve_device(device_cfg):
        """Resolve 'auto' to CUDA if available, otherwise CPU."""
        if device_cfg == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        return str(device_cfg)
