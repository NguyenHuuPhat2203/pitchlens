# Research log

## 2026-10-19 — Bootstrap
- Question: why is the pipeline at ~17–21 fps when published YOLO26s TensorRT latency is 2.5 ms? Workspace under `research/` (the project root `src/`/`data/` are product code/data).
- Proxy metric locked: end-to-end ms/frame on sample.mp4, JSON-only, model load excluded; guard = detection recall ≥ 0.99 at IoU 0.5 vs baseline.
- Hypotheses H1–H5 recorded in `research-state.yaml`.
- No `/loop` scheduler exists in this harness; continuity is via these files.

## 2026-10-19 — H1/H2 latency breakdown (run_001)
- H1 refuted: player raw inference 14.9 ms = 55% of track (predicted ≤ 40%).
- H2 supported: ball raw inference 83% of predict.
- Decision: test TensorRT next (H3), since GPU compute dominates. Added H6 (OC-SORT 6.5 ms).
- Installed `tensorrt-cu12-libs 10.16.1.11` into `.venv` (ORT 1.26 TRT EP links `nvinfer_10.dll`).

## 2026-10-19 — H3 / H3b TensorRT (run_002, run_003)
- Raw TRT FP16: player 14.77 → 8.10 ms, ball 22.64 → 13.30 ms (1.7–1.8×; H3 predicted ≥ 2× → inconclusive). Cold engine build ~20 min total, then cached.
- End to end, interleaved B/T/B/T: 52.31 → 39.85 ms/frame (−23.8%), guards hold → H3b supported.
- Shipped as `model.tensorrt` (default on, falls back to CUDA). Found that Ultralytics rebuilds its predictor when `device`/`quantize`/`tracker` change, so the swap must follow a real first call.

## 2026-10-19 — H4 parallel ball (run_004)
- 40.05 → 36.93 ms/frame, identical outputs. Shipped.

## 2026-10-19 — Outer loop 1→2: bottleneck moved to CPU (run_005–run_007)
- cProfile: OC-SORT ~8 ms, CPU letterbox ~3 ms, post ~1.5 ms; GPU forward 8.8 ms.
- H6b (cheaper tracker) refuted on SportsMOT football GT: ByteTrack −4 ms but +35% ID switches.
- H8 (re-enable OpenCV threads) refuted: contention, unstable (+2.5 ms mean).
- Direction: deepen on removing CPU work (H7b GPU letterbox, H9 smaller player input with GT guard, vectorised OC-SORT).
