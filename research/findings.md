# Findings — pipeline throughput

## Research question
Where does per-frame time go in the full offline pipeline on an RTX 3050 Ti Laptop (4 GB), and which changes cut end-to-end ms/frame without changing detections beyond the accepted FP16 tolerance?

## Current understanding
- **Trajectory (default config, JSON-only, 1505 frames)**: 52.3 ms/frame (CUDA EP) → 39.9 (TensorRT EP, H3b) → 36.9 (ball detection overlapped with player tracking, H4). That is 19.1 → 27.1 fps, **−29.4%**, with outputs equal to the baseline: player recall 0.997, identical ball states and shots.
- The bottleneck has moved. At the start, GPU inference was the largest term (H1/H2). After TensorRT the player model's GPU forward is only 8.1–8.8 ms, and **CPU work inside `player_track` (~13 ms) is now larger than the GPU work**: OC-SORT ~6.5–9 ms, CPU letterbox ~3 ms, Ultralytics post ~1.5 ms (H6/H7).
- Copies are not a factor: I/O binding saves ≤ 1.7 ms, and Ultralytics already uses it.

## Patterns and insights
- On this laptop GPU, the big levers came in order of "what the GPU is doing": first a better engine (TensorRT ~1.8× per model), then filling GPU idle time (overlap ~35% of the ball cost). Once the GPU work shrank, Python per-object overhead dominated. The next gains are CPU-side, not GPU-side.
- Overlap is partial because both models share one small GPU: `player_track` rises 21.7 → 30 ms when the ball model runs alongside it. More concurrency will hit diminishing returns.

## Lessons and constraints
- Both ONNX files have **fixed** input shapes (960, 1280); only the player has a `.pt` (`yolo26s.pt`, COCO).
- `onnxruntime-gpu` < 1.27 (CUDA 12). Its TRT EP links `nvinfer_10.dll`, so use `tensorrt-cu12-libs` 10.x (installed: 10.16.1.11).
- Ultralytics always opens ONNX on the CUDA EP. To use TRT, swap `backend.session` after a first **real** call; Ultralytics rebuilds the predictor when `device`/`quantize`/`tracker` change, which drops the swap.
- A cold TensorRT engine build takes ~5 min per model here. The cache in `.cache/trt/` makes later starts < 1 s.
- Single-model latency drifts ~20% between runs (laptop clocks). End-to-end comparisons must interleave variants (B/T/B/T), which gives < 1% spread.
- numpy float16 arithmetic is very slow; never normalise in numpy fp16. motmetrics' `iou_matrix` breaks on NumPy 2 (`np.asfarray`).
- Never commit per-frame dumps (`frames.json` is ~12 MB/run); they are gitignored under `research/`.

## Open questions
- H6b: does ByteTrack/FastTrack cut ≥ 2 ms with IDF1 within 2 points of OC-SORT? (running on SportsMOT football GT)
- H7b: does a GPU letterbox remove ~3 ms with detection parity?
- Is the remaining "other" time (~9 ms: team classification, motion, bookkeeping, decode wait) worth profiling?
