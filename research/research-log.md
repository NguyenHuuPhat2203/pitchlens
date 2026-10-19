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
