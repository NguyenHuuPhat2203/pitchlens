"""GPU inference helpers — OOM handling, FP16 models, device diagnostics."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from onnxruntime.capi import onnxruntime_pybind11_state as _ort_state

from src.logger import get_logger

_log = get_logger(__name__)

# CUDA OOM surfaces as a torch error for .pt models and as an
# onnxruntime Fail/RuntimeException for .onnx models.
_OOM_ERRORS = (
    torch.cuda.OutOfMemoryError,
    _ort_state.Fail,
    _ort_state.RuntimeException,
)


def safe_predict(model, *args, **kwargs):
    """Call model; on an OOM error free GPU memory and retry once.

    The retry uses the same device and arguments (an already-loaded
    Ultralytics model cannot be moved by a ``device`` kwarg).  A second
    failure propagates to the caller.

    Args:
        model: A YOLO model instance (or any callable returning results).
        *args: Positional args forwarded to the model.
        **kwargs: Keyword args forwarded to the model.

    Returns:
        Model results.
    """
    try:
        return model(*args, **kwargs)
    except _OOM_ERRORS:
        _log.warning("inference_oom_retry")
        _free_gpu_memory()
        return model(*args, **kwargs)


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


def enable_tensorrt(yolo, device: str) -> bool:
    """Run a YOLO ONNX model on ONNX Runtime's TensorRT EP instead of CUDA.

    Ultralytics always opens ONNX models on the CUDA EP, so this swaps the
    loaded session for a TensorRT one on the same file, bound to the same
    output buffers.  TensorRT FP16 runs the detectors ~1.8x faster with
    matching detections (see ``research/experiments/h3-tensorrt``).

    Call it after one real ``predict``/``track`` call made with the same
    ``device``/``quantize``/``tracker`` arguments as later calls: Ultralytics
    rebuilds its predictor (dropping this session) when those change.

    Engines are cached in ``.cache/trt``; the first build of a model takes
    several minutes.  Returns False and leaves the CUDA session in place
    when the device is CPU, the model is not ONNX on CUDA, or the TensorRT
    runtime (``tensorrt-cu12-libs`` 10.x) is missing or fails.
    """
    if device == "cpu":
        return False
    try:
        import tensorrt_libs
    except ImportError:
        _log.info("tensorrt_unavailable", reason="tensorrt-cu12-libs not installed")
        return False
    os.add_dll_directory(os.path.dirname(tensorrt_libs.__file__))

    if yolo.predictor is None:
        raise ValueError("enable_tensorrt() needs one real predict/track call first")
    backend = yolo.predictor.model.backend
    if not getattr(backend, "use_io_binding", False):
        return False  # not an ONNX model on the CUDA EP

    from src import CACHE_DIR

    cache = str(CACHE_DIR / "trt")
    os.makedirs(cache, exist_ok=True)
    path = backend.session._model_path
    _log.info("tensorrt_session_start", model=path, note="first build of an engine takes minutes")
    try:
        session = onnxruntime.InferenceSession(path, providers=[
            ("TensorrtExecutionProvider", {
                "device_id": int(device),
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cache,
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": cache,
            }),
            ("CUDAExecutionProvider", {"device_id": int(device)}),
        ])
    except (_ort_state.Fail, _ort_state.RuntimeException) as e:
        _log.warning("tensorrt_failed", model=path, error=str(e))
        return False
    if session.get_providers()[0] != "TensorrtExecutionProvider":
        _log.warning("tensorrt_failed", model=path, error="provider did not load")
        return False

    io = session.io_binding()
    for output, buffer in zip(session.get_outputs(), backend.bindings):
        io.bind_output(
            name=output.name,
            device_type="cuda",
            device_id=int(device),
            element_type=np.float16 if "float16" in output.type else np.float32,
            shape=tuple(buffer.shape),
            buffer_ptr=buffer.data_ptr(),
        )
    backend.session, backend.io = session, io
    return True


def enable_gpu_letterbox(yolo) -> bool:
    """Letterbox frames on the GPU instead of with OpenCV on the CPU.

    Ultralytics resizes and pads each frame with ``cv2`` before uploading it
    (~3 ms per 1280x720 frame on the main thread). This uploads the raw frame
    and resizes/pads with torch instead, using ``LetterBox``'s own geometry, so
    detections map back to the frame exactly as before. Bilinear rounding can
    differ from OpenCV's fixed-point resize by one grey level on some pixels.
    Without a resize (the 1280 ball model) the input is bit-identical.

    Call after one real ``predict`` call (the predictor must exist); returns
    False on CPU, where the OpenCV path is kept.
    """
    from ultralytics.data.augment import LetterBox

    predictor = yolo.predictor
    if predictor is None:
        raise ValueError("enable_gpu_letterbox() needs one real predict call first")
    if predictor.device.type != "cuda":
        return False
    cpu_preprocess = predictor.preprocess

    def preprocess(im):
        if isinstance(im, torch.Tensor) or len(im) != 1 or im[0].ndim != 3 or im[0].shape[2] != 3:
            return cpu_preprocess(im)
        model, frame = predictor.model, im[0]
        auto = predictor.args.rect and not predictor.scale_fill and (  # as BasePredictor.pre_transform
            model.format == "pt" or (getattr(model, "dynamic", False) and model.format != "imx")
        )
        box = LetterBox(predictor.imgsz, auto=auto, scale_fill=predictor.scale_fill, stride=model.stride)
        p = box.get_params({"img": frame})
        x = torch.from_numpy(frame).to(predictor.device).permute(2, 0, 1).unsqueeze(0)
        w, h = p["new_unpad"]
        if (h, w) != frame.shape[:2]:
            x = torch.nn.functional.interpolate(x.float(), size=(h, w), mode="bilinear", align_corners=False)
            x = x.round_().clamp_(0, 255)  # cv2 resizes uint8 -> integer grey levels
        x = torch.nn.functional.pad(x, (p["left"], p["right"], p["top"], p["bottom"]), value=114)
        x = x.flip(1)  # BGR -> RGB
        return (x.half() if model.fp16 else x.float()).div_(255)

    predictor.preprocess = preprocess
    return True



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
