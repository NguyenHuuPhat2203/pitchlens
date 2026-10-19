"""Pydantic config models — validated at startup."""

from __future__ import annotations

from pathlib import Path

import torch
from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)


class ModelConfig(BaseModel):
    """YOLO model settings."""

    player_detector: str = "yolo26s.pt"
    detection_conf: NonNegativeFloat = Field(default=0.3, le=1.0)
    device: str = "auto"
    half: bool | None = None  # None = FP16 when device is not CPU

    @field_validator("device", mode="after")
    @classmethod
    def _resolve_device(cls, v: str) -> str:
        if v == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        return v

    @model_validator(mode="after")
    def _resolve_half(self) -> ModelConfig:
        if self.half is None:
            self.half = self.device != "cpu"
        return self


class PipelineConfig(BaseModel):
    """Pipeline execution settings."""

    display: bool = True
    output_video: str | None = None
    output_json: str | None = None
    cpu_workers: PositiveInt = 2


class TeamClassifierConfig(BaseModel):
    """HSV team classifier settings."""

    enabled: bool = True
    fit_frames: PositiveInt = 50


class BallDetectorConfig(BaseModel):
    """Ball detection and tracking settings."""

    enabled: bool = True
    model: str = "yolo26s.pt"
    max_lost_frames: PositiveInt = 15


class ShotDetectorConfig(BaseModel):
    """Shot event detection thresholds."""

    angle_threshold: NonNegativeFloat = 30.0
    accel_threshold: NonNegativeFloat = 50.0


class CalibrationConfig(BaseModel):
    """Pitch calibration and speed estimation settings."""

    enabled: bool = True
    speed_window: PositiveInt = 12
    camera_dir: str | None = None


class TimelineConfig(BaseModel):
    """Offline tactical timeline settings."""

    enabled: bool = True
    max_ball_player_distance_px: PositiveFloat = 80.0
    min_possession_frames: PositiveInt = 3
    output_json: str | None = None
    benchmark_json: str | None = None


class AppConfig(BaseModel):
    """Root configuration — mirrors config/default.yaml."""

    video: VideoConfig
    model: ModelConfig = ModelConfig()
    pipeline: PipelineConfig = PipelineConfig()
    team_classifier: TeamClassifierConfig = TeamClassifierConfig()
    ball_detector: BallDetectorConfig = BallDetectorConfig()
    shot_detector: ShotDetectorConfig = ShotDetectorConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    timeline: TimelineConfig = TimelineConfig()


class VideoConfig(BaseModel):
    """Input video source."""

    source: str = "data/sample.mp4"


def load_config(path: str | Path) -> AppConfig:
    """Load and validate YAML config into a Pydantic model.

    Args:
        path: Path to a YAML config file.

    Returns:
        A validated AppConfig instance.

    Raises:
        ValidationError: If the YAML is missing required fields or
            values are out of range.
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AppConfig(**raw)
