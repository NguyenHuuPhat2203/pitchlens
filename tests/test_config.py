"""Tests for model precision resolution."""

from src.config import ModelConfig


def test_half_defaults_off_on_cpu() -> None:
    assert ModelConfig(device="cpu").half is False


def test_half_defaults_on_for_cuda_device() -> None:
    assert ModelConfig(device="0").half is True


def test_explicit_half_wins() -> None:
    assert ModelConfig(device="0", half=False).half is False


def test_omitted_device_resolves_like_auto() -> None:
    cfg = ModelConfig()
    assert cfg.device == ModelConfig(device="auto").device
    assert cfg.device in ("0", "cpu")
    assert cfg.half is (cfg.device != "cpu")
