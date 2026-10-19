# Literature / reference notes

## Ultralytics YOLO26 detection benchmarks
- URL: https://docs.ultralytics.com/tasks/detect/#detection-coco (summarised in `docs/research-real-time-football-tracking.md` §2.1, §10)
- YOLO26s: 2.5 ms/frame, T4 + TensorRT10, batch 1, 640 px. TensorRT export documented as "up to 5× GPU" over PyTorch.
- Relevance: the reference point. Scaling 640→960 is 2.25× pixels, so roughly 5–6 ms is plausible for our player model on a faster GPU. We measure `player_track` = 33 ms on an RTX 3050 Ti Laptop (ORT CUDA EP, FP16).

## ONNX Runtime TensorRT execution provider
- URL: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
- Version table (as published): ORT 1.22 → TensorRT 10.9, CUDA 12.0–12.8; 1.21 → TRT 10.8. The table stops at 1.22; we run ORT 1.26 (CUDA 12.8 build). [INFERENCE: still TRT 10.x]
- Runs an existing ONNX graph with TensorRT, falling back to CUDA EP for unsupported nodes. Engine caching (`trt_engine_cache_enable`) avoids rebuilding the engine each start (minutes → seconds).
- Relevance: our ball detector exists only as ONNX (no `.pt`), so ORT-TRT is the only TensorRT route for it. The player model also has `yolo26s.pt`, so Ultralytics `export(format="engine")` is a second route.

## ONNX Runtime CUDA EP note (from the previous perf work)
- From 1.27, PyPI `onnxruntime-gpu` targets CUDA 13; pinned `<1.27` for torch cu128.
