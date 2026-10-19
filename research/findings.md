# Findings — pipeline throughput

## Research question
Where does per-frame time go in the full offline pipeline on an RTX 3050 Ti Laptop (4 GB), and which changes cut end-to-end ms/frame without changing detections beyond the accepted FP16 tolerance?

## Current understanding
- Baseline (full pipeline, CUDA EP FP16, output video on, smoke run 2026-10-19): `player_track` 33.0 ms, `ball_detect` 29.2 ms every other frame (≈ 14.6 ms/frame), `encode` 8.3 ms, `render` 3.0 ms, `decode` 2.8 ms (overlapped). ≈ 59 ms/frame with video, ≈ 48 ms/frame JSON-only, i.e. 17–21 fps.
- **The GPU forward pass is the largest term** (H1/H2): player raw run 14.9 ms, ball raw run 28.6 ms. Copying is not the problem: I/O binding saves ≤ 1.7 ms.
- Outside the session, the player path spends 5.6 ms in Ultralytics pre/post and 6.5 ms in OC-SORT (CPU).

## Patterns and insights
- The 1280 px fixed-input ball model costs ~2× the 960 px player model per call: pixel count (1.78×) roughly explains it. Halving its call rate already happened (every other frame); the remaining lever is a faster engine.

## Lessons and constraints
- Both ONNX files have **fixed** input shapes (960, 1280); only the player has a `.pt` (`yolo26s.pt`, COCO). Any resolution change for the ball needs the original weights, which we don't have.
- `onnxruntime-gpu` must stay < 1.27 (CUDA 12); its TensorRT EP links `nvinfer_10.dll`, so TensorRT must be a 10.x runtime.
- numpy float16 arithmetic is slow (45–114 ms per frame in a harness). Never normalise in numpy fp16.

## Open questions
- H3: does TensorRT FP16 give ≥ 2× raw inference with parity? (running)
- Why is `player_track` 33 ms in the pipeline vs 27 ms in isolation? (contention hypothesis, untested)
- H6: can OC-SORT's 6.5 ms be reduced (e.g., ByteTrack) without ID-switch regressions? Needs a tracking metric (no labels for sample.mp4; SoccerNet tracking has them).
