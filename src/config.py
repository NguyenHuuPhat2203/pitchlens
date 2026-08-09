"""Pydantic config models — validated at startup per ML rules §03."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from pydantic import BaseModel, Field, NonNegativeFloat, PositiveInt
from pydantic import field_validator


class ModelConfig(BaseModel):
    """YOLO model settings."""

    player_detector: str = "yolo26s.pt"
    detection_conf: NonNegativeFloat = Field(default=0.3, le=1.0)
    device: str = "auto"

    @field_validator("device", mode="after")
    @classmethod
    def _resolve_device(cls, v: str) -> str:
        if v == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        return v


class PipelineConfig(BaseModel):
    """Pipeline execution settings."""

    display: bool = True
    output_video: Optional[str] = None
    output_json: Optional[str] = None
    cpu_workers: PositiveInt = 2


class TeamClassifierConfig(BaseModel):
    """HSV team classifier settings."""

    enabled: bool = True
    fit_frames: PositiveInt = 30


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
    speed_window: PositiveInt = 5


class AppConfig(BaseModel):
    """Root configuration — mirrors config/default.yaml."""

    video: VideoConfig
    model: ModelConfig = ModelConfig()
    pipeline: PipelineConfig = PipelineConfig()
    team_classifier: TeamClassifierConfig = TeamClassifierConfig()
    ball_detector: BallDetectorConfig = BallDetectorConfig()
    shot_detector: ShotDetectorConfig = ShotDetectorConfig()
    calibration: CalibrationConfig = CalibrationConfig()


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
