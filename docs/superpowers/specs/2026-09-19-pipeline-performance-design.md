# Pipeline Performance Fix — Design

## Goal

Fix the measured bottlenecks in the offline pipeline: CPU-only inference, unbounded frame buffering, and missing device diagnostics. Run FP16 by default on CUDA. Keep the MJPEG streaming path unchanged.

## Decisions (from HTML plan review)

| # | Question | Answer |
|---|---|---|
| 1 | CUDA build of torch | `cu128` |
| 2 | TensorRT FP16 export now | No — measure CUDA first |
| 3 | `run()` return when not rendering | New `RunResult(frame_count, frames)` |
| 4 | Precision on CUDA | FP16 automatically (user request); configurable |

## Evidence

Benchmark from `.scratch/verification/benchmark.json` (1505-frame batch, JSON-only):

| Stage | Median/frame | Share of total |
|---|---:|---:|
| `player_track` | 233.0 ms | ~99.7% |
| `decode` | 2.7 ms | ~0.3% |
| `json_write` (whole batch) | 1017 ms once | <0.3% |

Root cause, observed in the previous `.venv`:
```
torch 2.14.0+cpu, cuda_available=False
onnxruntime providers: ['AzureExecutionProvider', 'CPUExecutionProvider']
```
Host GPU: RTX 3050 Ti Laptop, driver 592.00.

`Pipeline.run()` (`src/serve/pipeline/pipeline.py:104-175`) stores every frame image in `annotated_frames` and returns a full list, even when `output_video` is `None`. A 90-minute 1080p match would need hundreds of GB of RAM.

No log line reports the device or ONNX execution provider in use.

The `.venv` directory no longer exists; it must be recreated before any task runs.

## Design

### 1. GPU inference stack

Recreate `.venv` with Python 3.12, install `torch`/`torchvision` from `https://download.pytorch.org/whl/cu128`, then `requirements.txt`. Add `onnx` and `onnxruntime-gpu>=1.20,<1.27` to `requirements.txt`: from 1.27, PyPI `onnxruntime-gpu` is built for CUDA 13 and fails to load `cublasLt64_13.dll` next to torch cu128 (CUDA 12.8). The CUDA torch index stays out of `requirements.txt` (it breaks non-GPU and CI installs) and goes into `README.md`.

Ultralytics' ONNX backend (`ultralytics/nn/backends/onnx.py`) selects `CUDAExecutionProvider` when torch reports CUDA and `onnxruntime-gpu` is installed; otherwise it logs a warning and uses CPU.

### 2. FP16 by default on CUDA

`ModelConfig.half: bool | None = None`. `None` resolves to `device != "cpu"` after device resolution. An explicit `true`/`false` overrides it.

Ultralytics decides ONNX precision from the model file, not from `half=True`:
```python
self.fp16 = "float16" in self.session.get_inputs()[0].type
```
So FP16 needs an FP16 ONNX file. The repo has only the FP32 exports (`models/player_detector.onnx`, `models/ball_detector.onnx`) and no fine-tuned `.pt` sources. `gpu_utils.fp16_model_path()` converts an `.onnx` once with `onnxruntime.transformers.float16.convert_float_to_float16(keep_io_types=False)`, caches it beside the source as `*.fp16.onnx` (under the git-ignored `models/`), and regenerates it when the source is newer. Non-ONNX paths (`.pt`) are returned unchanged, and `half=` is also passed to `track()`/`predict()` so `.pt` models use FP16 on CUDA.

FP16 must not change detections materially. A parity check on `data/sample.mp4` compares FP32 and FP16 outputs per model. If a model fails the gate, `config/default.yaml` sets `model.half: false` and the result is reported; there is no silent fallback.

### 3. Memory-safe batch run with `RunResult`

```python
@dataclass
class RunResult:
    frame_count: int
    frames: list[np.ndarray] = field(default_factory=list)  # empty unless rendering
```

`Pipeline.run()` returns `RunResult`. It stores frame images only when `render` is `True` (`display` or `output_video`). All callers migrate: `main.py` logs `result.frame_count`; tests read `result.frames` / `result.frame_count`.

### 4. Device logging

`gpu_utils.execution_summary(device, half)` returns `{"device", "half", "cuda_available", "onnx_providers"}`. `Pipeline.__init__` logs it once as `pipeline_device`.

## Out of scope

TensorRT export, ball-detector batching, removing the decode thread (decode is 2.7 ms), and any change to `Pipeline.process_frame()` / `app.py` streaming.

## Acceptance

- `torch.cuda.is_available()` is `True`; `onnxruntime.get_available_providers()` includes `CUDAExecutionProvider`.
- FP16 parity per model on 50 sample frames: matched recall ≥ 0.99 (IoU ≥ 0.5), mean IoU of matches ≥ 0.98, mean absolute confidence difference ≤ 0.02.
- `player_track.median_ms` on the same 1505-frame benchmark is at least 10× lower than 233.0 ms. Record FP32-CUDA and FP16-CUDA numbers separately.
- A JSON-only run returns `RunResult(frame_count=N, frames=[])`.
- Startup logs `pipeline_device` with device, half, and providers.

## Tests

- `ModelConfig.half` resolves to `False` on CPU and `True` on a CUDA device string; explicit values win.
- `fp16_model_path()` converts a tiny ONNX graph to a float16 input, reuses the cache, regenerates it when the source changes, and returns `.pt` paths unchanged.
- `execution_summary()` returns its four keys.
- JSON-only `run()` returns zero frames and the correct `frame_count`; render-enabled `run()` returns one correctly shaped frame per input frame.

## Results (2026-10-19, RTX 3050 Ti Laptop, driver 592.00)

Stack: torch 2.11.0+cu128, onnxruntime-gpu 1.26.0, Ultralytics 8.4.182. `pipeline_device` logs `onnx_cuda: true`.

`get_available_providers()` alone is not proof: with onnxruntime-gpu 1.31.0 it listed `CUDAExecutionProvider` while every session fell back to CPU. `execution_summary()` now opens a probe session (`onnx_cuda`).

### FP16 parity (50 frames, CUDA)

| Model | FP32 boxes | Recall | Mean IoU | Mean \|Δconf\| | Gate |
|---|---:|---:|---:|---:|---|
| player | 1253 | 0.9984 | 0.9794 | 0.0018 | fail (IoU < 0.98) |
| ball | 69 | 1.0000 | 0.9561 | 0.0017 | fail (IoU < 0.98) |

Per decision 2, `config/default.yaml` first set `model.half: false`. The user then accepted the small IoU drop, so `model.half: null` (FP16 on CUDA) is the default again. Note: ball boxes are 10–20 px, so a sub-pixel shift lowers IoU sharply; recall and confidence match.

### Benchmark (1505 frames, JSON-only, ball detector off)

| Run | `player_track` median | P95 | Total |
|---|---:|---:|---:|
| CPU (before) | 233.0 ms | 268.8 ms | 395.6 s |
| CUDA FP32 | 34.5 ms | 36.9 ms | 59.0 s |
| CUDA FP16 | 28.7 ms | 34.7 ms | 55.2 s |

Speedup: 6.8× (FP32), 8.1× (FP16). The ≥ 10× target is **not met**. `player_track` includes Ultralytics pre/post-processing and OC-SORT, not only the ONNX session; TensorRT (decision 3, deferred) is the next measured step.
