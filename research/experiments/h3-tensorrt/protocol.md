# H3: TensorRT FP16 via ONNX Runtime TensorRT EP

## Hypothesis
TensorRT builds fused FP16 kernels tuned for the RTX 3050 Ti, so raw inference drops ≥ 2× vs the CUDA EP on the same FP16 graphs (player 14.87 → ≤ 7.4 ms; ball 28.61 → ≤ 14.3 ms).

## Method
- Install the TensorRT 10.x runtime wheel (`tensorrt-cu12-libs`) into `.venv`. No system install; libraries stay inside the project venv.
- Sessions: `TensorrtExecutionProvider` (fp16 on, engine cache in `.cache/trt/`) with `CUDAExecutionProvider` fallback.
- The input is the original FP32 ONNX (TRT builds FP16 itself), plus the FP16 ONNX as a second variant.
- Same 200 frames as H1/H2. Measure `session.run` with numpy I/O; warm-up 10 (engine build excluded).
- Parity vs CUDA-EP FP16 outputs on the same frames: matched recall ≥ 0.99 at IoU 0.5, same conf threshold 0.3.

## Decision rule
Supported if both models reach ≥ 2× with parity. Supported for one model only → adopt for that model. If TRT EP cannot load on this machine (library/version mismatch), record it as blocked with the error, and fall back to measuring Ultralytics `export(format="engine")` for the player model.
