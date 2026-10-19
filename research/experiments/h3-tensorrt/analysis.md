# H3 analysis — run_002 (CONFIRMATORY)

190 timed frames, raw `session.run` with numpy I/O. Parity is measured against the CUDA EP FP16 output on the same frames.

| Model | CUDA EP FP16 | TRT ← FP32 ONNX | TRT ← FP16 ONNX | Recall / mean IoU (TRT←FP16) |
|---|---:|---:|---:|---|
| player 960 | 14.77 ms | 8.60 ms (1.72×) | **8.10 ms (1.82×)** | 0.9977 / 0.9973 (4691 boxes) |
| ball 1280 | 22.64 ms | **12.75 ms (1.78×)** | 13.30 ms (1.70×) | 0.9957 / 0.9941 (231 boxes) |

## Verdict
**Partially supported.** The gain is real and parity holds: recall ≥ 0.99, and IoU against CUDA FP16 is ≥ 0.994, i.e. TRT FP16 ≈ CUDA FP16. But it is 1.7–1.8×, short of the ≥ 2× prediction.

## Notes
- The cold engine build took ~20 minutes for 4 engines on this laptop (the first run). Cached runs create the session in < 0.1 s, so the engine cache must persist (`.cache/trt/`, ignored by git).
- The ball CUDA baseline was 22.6 ms here vs 28.6 ms in run_001 (same frames, same file). That is a ~20% run-to-run drift, likely GPU clock/thermal on the laptop. Single runs are therefore noisy at that level; end-to-end tests must interleave runs (adopted in H3b).
- Building from the FP32 vs the FP16 ONNX makes no consistent difference (TRT picks FP16 kernels either way).
