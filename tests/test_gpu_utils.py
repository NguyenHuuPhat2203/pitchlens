"""Tests for FP16 model conversion, TensorRT switching, and device diagnostics."""

import importlib.util
import os
import sys

import cv2
import onnx
import pytest
import torch
from onnx import TensorProto, helper
from onnxruntime.capi import onnxruntime_pybind11_state as ort_state

from src.backbone.gpu_utils import (
    enable_gpu_letterbox,
    enable_tensorrt,
    execution_summary,
    fp16_model_path,
    safe_predict,
)

PLAYER_ONNX = "models/player_detector.fp16.onnx"
needs_cuda_model = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.path.exists(PLAYER_ONNX)),
    reason="needs CUDA and the exported player model",
)


def _sample_frame():
    cap = cv2.VideoCapture("data/sample.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
    ok, frame = cap.read()
    cap.release()
    assert ok
    return frame


def _boxes(yolo):
    r = yolo.predict(_sample_frame(), device="0", classes=[0], conf=0.3, verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy()


@needs_cuda_model
def test_enable_tensorrt_keeps_cuda_when_runtime_missing(monkeypatch) -> None:
    from ultralytics import YOLO

    monkeypatch.setitem(sys.modules, "tensorrt_libs", None)  # import now fails
    yolo = YOLO(PLAYER_ONNX, task="detect")

    assert enable_tensorrt(yolo, "0") is False
    assert len(_boxes(yolo)) > 0
    session = yolo.predictor.model.backend.session
    assert session.get_providers()[0] == "CUDAExecutionProvider"


@needs_cuda_model
@pytest.mark.skipif(
    importlib.util.find_spec("tensorrt_libs") is None, reason="TensorRT runtime not installed"
)
def test_enable_tensorrt_switches_provider_with_same_detections() -> None:
    from ultralytics import YOLO

    ref = _boxes(YOLO(PLAYER_ONNX, task="detect"))
    trt = YOLO(PLAYER_ONNX, task="detect")
    _boxes(trt)  # first real call loads the model with the production arguments

    assert enable_tensorrt(trt, "0") is True
    got = _boxes(trt)
    assert trt.predictor.model.backend.session.get_providers()[0] == "TensorrtExecutionProvider"
    assert len(got) == pytest.approx(len(ref), abs=1)
    for box in ref:
        assert abs(got - box).max(axis=1).min() < 3.0  # every reference box found within 3 px


def test_enable_tensorrt_is_a_noop_on_cpu() -> None:
    from ultralytics import YOLO

    assert enable_tensorrt(YOLO("yolo26s.pt"), "cpu") is False


@needs_cuda_model
def test_gpu_letterbox_matches_opencv_letterbox() -> None:
    from ultralytics import YOLO

    yolo = YOLO(PLAYER_ONNX, task="detect")
    _boxes(yolo)  # builds the predictor
    opencv = yolo.predictor.preprocess
    assert enable_gpu_letterbox(yolo) is True
    gpu = yolo.predictor.preprocess

    frame = _sample_frame()  # 1280x720: resize to 960x540, then pad
    a, b = opencv([frame]), gpu([frame])
    assert a.shape == b.shape == (1, 3, 960, 960)
    assert float((a.float() - b.float()).abs().max()) * 255 < 1.5  # at most one grey level
    padded_only = cv2.resize(frame, (960, 540))  # no resize needed: identical input
    assert torch.equal(opencv([padded_only]), gpu([padded_only]))


def _tiny_onnx(path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Relu", ["x"], ["y"])
    model = helper.make_model(helper.make_graph([node], "g", [x], [y]))
    onnx.save(model, str(path))


def test_fp16_model_path_converts_input_to_float16(tmp_path) -> None:
    src = tmp_path / "m.onnx"
    _tiny_onnx(src)
    out = fp16_model_path(str(src))
    assert out.endswith("m.fp16.onnx")
    model = onnx.load(out)
    assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT16


def test_fp16_model_path_reuses_cache_until_source_changes(tmp_path) -> None:
    src = tmp_path / "m.onnx"
    _tiny_onnx(src)
    out = fp16_model_path(str(src))
    src_mtime = os.path.getmtime(src)

    # A cache newer than the source is reused as-is.
    with open(out, "wb") as fh:
        fh.write(b"stale")
    os.utime(out, (src_mtime + 10, src_mtime + 10))
    with open(fp16_model_path(str(src)), "rb") as fh:
        assert fh.read() == b"stale"

    # A cache older than the source is rebuilt.
    os.utime(out, (src_mtime - 10, src_mtime - 10))
    model = onnx.load(fp16_model_path(str(src)))
    assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT16


def test_fp16_model_path_keeps_pt_unchanged() -> None:
    assert fp16_model_path("yolo26s.pt") == "yolo26s.pt"


def test_execution_summary_reports_device_half_and_providers() -> None:
    summary = execution_summary("cpu", half=False)
    assert summary["device"] == "cpu"
    assert summary["half"] is False
    assert isinstance(summary["cuda_available"], bool)
    assert "CPUExecutionProvider" in summary["onnx_providers"]
    assert isinstance(summary["onnx_cuda"], bool)


@pytest.mark.parametrize(
    "error", [torch.cuda.OutOfMemoryError("oom"), ort_state.RuntimeException("oom")]
)
def test_safe_predict_retries_once_after_oom(error) -> None:
    calls = []

    def model(x, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise error
        return "ok"

    assert safe_predict(model, 1, device="0") == "ok"
    assert calls == [{"device": "0"}, {"device": "0"}]


def test_safe_predict_propagates_persistent_failure() -> None:
    def model(x):
        raise torch.cuda.OutOfMemoryError("oom")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        safe_predict(model, 1)
