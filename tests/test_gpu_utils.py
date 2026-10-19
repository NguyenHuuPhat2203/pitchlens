"""Tests for FP16 model conversion and device diagnostics."""

import os

import onnx
import pytest
import torch
from onnx import TensorProto, helper
from onnxruntime.capi import onnxruntime_pybind11_state as ort_state

from src.backbone.gpu_utils import execution_summary, fp16_model_path, safe_predict


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
