# Pipeline Performance Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run inference on the RTX 3050 Ti in FP16 by default, stop buffering frame images in JSON-only batches, and log the active device — no MJPEG behavior change.

**Architecture:** A recreated `.venv` carries the `cu128` torch build and `onnxruntime-gpu`. `ModelConfig.half` resolves to FP16 on CUDA; `gpu_utils.fp16_model_path()` produces a cached FP16 ONNX file because Ultralytics reads ONNX precision from the file. `Pipeline.run()` returns `RunResult(frame_count, frames)` and keeps images only when rendering.

**Tech Stack:** Python 3.12, `.venv`, PyTorch `cu128`, `onnx`, `onnxruntime-gpu`, Ultralytics, structlog, pytest, Ruff.

**Rule:** Run every Python command with `.venv/Scripts/python.exe`. Never use the system Python except to create the venv.

---

## File structure

- Modify `requirements.txt`: add `onnx`, `onnxruntime-gpu`.
- Modify `README.md`: CUDA install command, FP16 default, `half` override.
- Modify `src/config.py`, `config/default.yaml`: `ModelConfig.half`.
- Modify `src/serve/pipeline/gpu_utils.py`: `execution_summary()`, `fp16_model_path()`.
- Modify `src/serve/pipeline/tracker.py`, `src/backbone/ball_detector.py`: accept and pass `half`.
- Modify `src/model/frame.py`: `RunResult`.
- Modify `src/serve/pipeline/pipeline.py`: FP16 model paths, device log, `RunResult`, no buffering.
- Modify `main.py`: use `RunResult.frame_count`.
- Create `tests/test_gpu_utils.py`, `tests/test_config.py`; modify `tests/test_pipeline.py`.

## Task 1: Recreate `.venv` with the CUDA stack

**Files:** Modify `requirements.txt`, `README.md`

- [ ] **Step 1: Add the ONNX GPU dependencies**

```
# requirements.txt — after "torch>=2.0.0"
onnx>=1.16.0
onnxruntime-gpu>=1.20.0
```

- [ ] **Step 2: Create the venv and install torch from the cu128 index first**

```bash
D:/Apps/Python312/python.exe -m venv .venv
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt pytest ruff
```

Installing torch first stops `pip install -r requirements.txt` from pulling the CPU wheel.

- [ ] **Step 3: Verify the GPU stack**

```bash
.venv/Scripts/python.exe -c "import torch, onnxruntime; print(torch.__version__, torch.cuda.is_available()); print(onnxruntime.get_available_providers())"
```

Expected: a `+cu128` version, `True`, and a list containing `'CUDAExecutionProvider'`.

If `CUDAExecutionProvider` is missing, run `.venv/Scripts/python.exe -m pip install "onnxruntime-gpu[cuda,cudnn]"` and verify again. If it is still missing, stop and report the import error.

- [ ] **Step 4: Baseline test run**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`

Expected: `66 passed`.

- [ ] **Step 5: Document the install in `README.md`**

Add a "GPU acceleration" subsection after "Quick Start" with the three commands from Step 2 and one sentence: `pip install -r requirements.txt` alone installs CPU-only torch on Windows.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt README.md
git commit -m "build: install CUDA cu128 inference stack"
```

## Task 2: Resolve FP16 automatically on CUDA

**Files:** Modify `src/config.py`, `config/default.yaml`; Create `tests/test_config.py`

- [ ] **Step 1: Write failing tests** in `tests/test_config.py`

```python
from src.config import ModelConfig


def test_half_defaults_off_on_cpu() -> None:
    assert ModelConfig(device="cpu").half is False


def test_half_defaults_on_for_cuda_device() -> None:
    assert ModelConfig(device="0").half is True


def test_explicit_half_wins() -> None:
    assert ModelConfig(device="0", half=False).half is False
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`

Expected: FAIL — `ModelConfig` has no `half` field.

- [ ] **Step 3: Implement**

```python
# src/config.py — ModelConfig
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
```

Import `model_validator` from pydantic. In `config/default.yaml` under `model:` add `half: null  # null = FP16 on CUDA, FP32 on CPU`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/config.py config/default.yaml tests/test_config.py
git commit -m "feat: default to FP16 on CUDA devices"
```

## Task 3: Cached FP16 ONNX models and device logging

**Files:** Modify `src/serve/pipeline/gpu_utils.py`, `src/serve/pipeline/tracker.py`, `src/backbone/ball_detector.py`, `src/serve/pipeline/pipeline.py`; Create `tests/test_gpu_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gpu_utils.py
"""Tests for FP16 model conversion and device diagnostics."""

import os

import onnx
from onnx import TensorProto, helper

from src.serve.pipeline.gpu_utils import execution_summary, fp16_model_path


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
    first = os.path.getmtime(out)
    assert os.path.getmtime(fp16_model_path(str(src))) == first
    os.utime(src, (first + 10, first + 10))
    assert os.path.getmtime(fp16_model_path(str(src))) > first


def test_fp16_model_path_keeps_pt_unchanged() -> None:
    assert fp16_model_path("yolo26s.pt") == "yolo26s.pt"


def test_execution_summary_reports_device_half_and_providers() -> None:
    summary = execution_summary("cpu", half=False)
    assert summary["device"] == "cpu"
    assert summary["half"] is False
    assert isinstance(summary["cuda_available"], bool)
    assert "CPUExecutionProvider" in summary["onnx_providers"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_gpu_utils.py -v`

Expected: FAIL — cannot import `execution_summary`, `fp16_model_path`.

- [ ] **Step 3: Implement in `gpu_utils.py`**

```python
from pathlib import Path

import onnxruntime


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


def execution_summary(device: str, half: bool) -> dict:
    """Report the device, precision, and ONNX Runtime providers available."""
    return {
        "device": device,
        "half": half,
        "cuda_available": torch.cuda.is_available(),
        "onnx_providers": onnxruntime.get_available_providers(),
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_gpu_utils.py -v`

Expected: 4 passed.

- [ ] **Step 5: Pass `half` through the detectors**

`Tracker.__init__(model_name, tracker_config, device="0", half=False)` stores `self._half`; `Tracker.track()` adds `half=self._half` to the `safe_predict(self._model.track, …)` call (`src/serve/pipeline/tracker.py:44-54`). Do the same in `BallDetector.__init__` / `detect_ball()` (`src/backbone/ball_detector.py:20-45`).

- [ ] **Step 6: Wire into `Pipeline.__init__`**

```python
device = model_cfg.device
half = model_cfg.half
player_model = fp16_model_path(model_cfg.player_detector) if half else model_cfg.player_detector
self._tracker = Tracker(model_name=player_model, tracker_config="ocsort.yaml", device=device, half=half)
# BallDetector: same pattern with ball_cfg.model
_log.info("pipeline_device", **execution_summary(device, half))
```

Add `from src.logger import get_logger`, `from src.serve.pipeline.gpu_utils import execution_summary, fp16_model_path`, and `_log = get_logger(__name__)` to `pipeline.py`.

- [ ] **Step 7: Run all tests**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`

Expected: 73 passed (66 + 3 config + 4 gpu_utils).

- [ ] **Step 8: Commit**

```bash
git add src/serve/pipeline/gpu_utils.py src/serve/pipeline/tracker.py src/backbone/ball_detector.py src/serve/pipeline/pipeline.py tests/test_gpu_utils.py
git commit -m "feat: cached FP16 ONNX models and device logging"
```

## Task 4: `RunResult` and no frame buffering

**Files:** Modify `src/model/frame.py`, `src/serve/pipeline/pipeline.py`, `main.py`, `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests** — update `tests/test_pipeline.py`

`test_processes_all_frames` and `test_output_frames_have_correct_shape` set `config.pipeline.output_video = str(Path(tmp) / "out.mp4")` and read `result = pipeline.run(video_path)`; assert `result.frame_count == 5` / iterate `result.frames`. In `test_json_only_batch_skips_renderer`, replace `assert len(frames) == 1` with `assert result.frame_count == 1 and result.frames == []`. Add:

```python
def test_json_only_run_returns_no_buffered_frames(tmp_path) -> None:
    video_path = str(tmp_path / "test.mp4")
    frame_json = str(tmp_path / "frames.json")
    _make_test_video(video_path, frame_count=4)
    pipeline = Pipeline(_make_config(video_path, output_json=frame_json))

    result = pipeline.run(video_path)

    assert result.frame_count == 4
    assert result.frames == []
    with open(frame_json, encoding="utf-8") as fh:
        assert len(json.load(fh)) == 4
```

Add `import json` at the top.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline.py -v`

Expected: FAIL — `list` has no attribute `frame_count`.

- [ ] **Step 3: Implement**

```python
# src/model/frame.py
@dataclass
class RunResult:
    """Outcome of one Pipeline.run() call.

    Attributes:
        frame_count: Frames processed.
        frames: Annotated frames; empty unless display or video output was requested.
    """

    frame_count: int
    frames: list[np.ndarray] = field(default_factory=list)
```

In `Pipeline.run()` (`src/serve/pipeline/pipeline.py:104-175`): change the return annotation to `RunResult`; write `annotated_frames[frame_index] = annotated` only `if render:`; build `ordered_annotated` only `if render` else `[]`; `return RunResult(frame_count=total_frames, frames=ordered_annotated)`.

- [ ] **Step 4: Migrate `main.py`**

```python
    result = pipeline.run(source)
    _log.info("pipeline_done", total_frames=result.frame_count)
```

- [ ] **Step 5: Run all tests and Ruff**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
.venv/Scripts/ruff.exe check src/ tests/ main.py
```

Expected: 74 passed; Ruff reports only the pre-existing findings in untouched files.

- [ ] **Step 6: Commit**

```bash
git add src/model/frame.py src/serve/pipeline/pipeline.py main.py tests/test_pipeline.py
git commit -m "perf: return RunResult and stop buffering unused frames"
```

## Task 5: FP16 parity gate and benchmark

**Files:** Modify `docs/superpowers/specs/2026-09-19-pipeline-performance-design.md`; throwaway `.scratch/fp16_parity.py` (not committed)

- [ ] **Step 1: Write the parity script**

`.scratch/fp16_parity.py` loads each model twice with `ultralytics.YOLO` (FP32 source, FP16 from `fp16_model_path`), runs `predict(device="0")` on the first 50 frames of `data/sample.mp4`, matches boxes greedily by IoU ≥ 0.5, and prints per model: matched recall, mean IoU, mean |Δconf|.

- [ ] **Step 2: Run it**

Run: `.venv/Scripts/python.exe .scratch/fp16_parity.py`

Expected per model: recall ≥ 0.99, mean IoU ≥ 0.98, mean |Δconf| ≤ 0.02. If a model fails, set `model.half: false` in `config/default.yaml`, commit that, and report which model failed and its numbers.

- [ ] **Step 3: Benchmark FP32-CUDA and FP16-CUDA**

Run the existing `.scratch/verification.yaml` twice (set `model.half: false`, then `null`):

```bash
.venv/Scripts/python.exe -c "from src.config import load_config; from src.serve.pipeline.pipeline import Pipeline; c=load_config('.scratch/verification.yaml'); print(Pipeline(c).run(c.video.source).frame_count)"
```

Copy `benchmark.json` after each run.

Expected: `player_track.median_ms` ≤ 23.3 ms in at least the FP16 run; startup log shows `CUDAExecutionProvider` and `half`.

- [ ] **Step 4: Record results** — append a table to the spec's Evidence section: CPU baseline 233.0 ms, FP32-CUDA, FP16-CUDA (median, P95, total), plus the parity numbers.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-09-19-pipeline-performance-design.md config/default.yaml
git commit -m "docs: record CUDA FP16 benchmark and parity"
```
