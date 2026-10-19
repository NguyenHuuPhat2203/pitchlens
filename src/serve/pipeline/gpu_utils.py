"""GPU inference helpers — OOM handling, FP16 models, device diagnostics."""

from __future__ import annotations

import gc
from pathlib import Path

import onnxruntime
import torch


def safe_predict(model, *args, **kwargs):
    """Call model with CUDA OOM recovery.

    On OOM: clear cache, retry on CPU.  Returns the model output
    (or None if both GPU and CPU fail).

    Args:
        model: A YOLO model instance (or any callable returning results).
        *args: Positional args forwarded to the model.
        **kwargs: Keyword args forwarded to the model.

    Returns:
        Model results, or None on unrecoverable error.
    """
    try:
        return model(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        _free_gpu_memory()
        try:
            # Retry on CPU.
            kwargs_cpu = {**kwargs, "device": "cpu"}
            return model(*args, **kwargs_cpu)
        except Exception:
            return None


def _free_gpu_memory() -> None:
    """Release available GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fp16_model_path(model_path: str) -> str:
    """Return a cached FP16 copy of an ONNX model; other formats pass through.

    Ultralytics reads ONNX precision from the graph input type, so FP16
    needs an FP16 file. The copy sits beside the source as ``*.fp16.onnx``
    and is rebuilt when the source is newer.
    """
    src = Path(model_path)
    if src.suffix != ".onnx":
        return model_path
    dst = src.with_name(f"{src.stem}.fp16.onnx")
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16

        model = convert_float_to_float16(onnx.load(str(src)), keep_io_types=False)
        onnx.save(model, str(dst))
    return str(dst)


def _onnx_cuda_usable() -> bool:
    """Return True only if a session really runs on CUDAExecutionProvider.

    ``get_available_providers()`` lists CUDA even when its DLLs (cuBLAS,
    cuDNN) fail to load; ONNX Runtime then falls back to CPU silently.
    """
    if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        return False
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "probe", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CUDAExecutionProvider"]
    )
    return "CUDAExecutionProvider" in session.get_providers()


def execution_summary(device: str, half: bool) -> dict:
    """Report the device, precision, and whether ONNX inference can use CUDA."""
    return {
        "device": device,
        "half": half,
        "cuda_available": torch.cuda.is_available(),
        "onnx_providers": onnxruntime.get_available_providers(),
        "onnx_cuda": _onnx_cuda_usable(),
    }
