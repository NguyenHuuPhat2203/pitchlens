# Findings — pipeline throughput

## Research question
Where does per-frame time go in the full offline pipeline on an RTX 3050 Ti Laptop (4 GB), and which changes cut end-to-end ms/frame without changing detections beyond the accepted FP16 tolerance?

## Current understanding
- **Shipped trajectory** (default config, JSON-only, sample.mp4, 1505 frames; each value is the variant mean of its own interleaved run):

  | Change | ms/frame |
  |---|---:|
  | Baseline (CUDA EP FP16) | 52.3 |
  | TensorRT EP (H3b) | 39.9 |
  | Ball model parallel to player tracking (H4) | 36.9 |
  | GPU letterbox (H7b) | 32.5 |
  | Batched OC-SORT port (H6d) | 27.6 |

  Overall 19.1 → 36.3 fps, **−47%**. Baselines drift ≤ 0.9 ms between sessions, so each step was measured against its own same-session baseline.
- Output safety per step:
  - TensorRT and the GPU letterbox change detections within FP16-level noise: player recall ≥ 0.996, IoU ≥ 0.99; ball, shots and possession unchanged.
  - Parallel ball and the OC-SORT port are byte-identical to their predecessors.
- **Where the time goes now:** TensorRT made the GPU forward passes small (player 6.6 ms, ball 13 ms every other frame). The remaining cost is CPU work on the main thread, plus GPU and CPU contention between the two detectors running at once. `player_track` takes 15 ms alone but ~22 ms next to the ball worker.

## Patterns and insights
- **Removing CPU work beats parallelising it.**
  - H8 (more OpenCV threads) made things slower and unstable.
  - H7b (removing the letterbox) saved more than its own 3 ms (5.4 ms), because it also took load off the ball worker thread that competes with the main thread.
  - H6d (cheaper tracker) saved more than the tracker's own 3 ms (4.5 ms), because it also dropped the track-callback copies.
- **Identity metrics decide tracker and input changes; MOTA misleads here.** The COCO person detector also boxes unannotated crowd and staff. Dropping those raises MOTA even when real players get lost (H9: 384×640 had the best MOTA and the worst IDF1).
- **Input geometry matters beyond scale.** At the same 0.75 scale, removing letterbox padding (544×960 vs 960×960) changed 5.5% of player boxes, cut crowd detections at the top of the frame, and missed slightly more mid-field players (H9b).

## Lessons and constraints
- Both ONNX files have fixed inputs (960², 1280²). The player model is `yolo26s.pt` (COCO, verified by re-export: max Δ 6e-5); the ball model's training weights are not in the repo.
- `onnxruntime-gpu` < 1.27 (CUDA 12); its TRT EP needs `tensorrt-cu12-libs` 10.x. The engine cache in `.cache/trt/` survived moving the project (cache hit after the move). A new engine takes 4–6 min to build.
- Ultralytics is pinned at 8.4.182: `enable_tensorrt`, `enable_gpu_letterbox` and the OC-SORT port depend on its internals. Ultralytics rebuilds its predictor when `device`/`quantize`/`tracker` change, which drops both patches. Apply them after one real call with the production arguments.
- "Identical output" must be checked at the product boundary. The H6c port was exact at the tracker level, but Ultralytics' `Results.update` also clips boxes to the image (H6c failed on that; H6d fixed it).
- Batched NumPy matmul, `np.linalg.solve` and `sqrt(dot)` reproduced the per-track BLAS calls bit-for-bit in thousands of random trials. Verify such claims empirically before relying on them.
- Single-model latency drifts ~20% within a session (laptop clocks). Only interleaved comparisons (S/V/S/V) are trusted; their spread is < 1 ms.
- Small input perturbations (±1 grey level) move SportsMOT IDF1 by ~1 point and ID switches by ~7%. Guards are set relative to that noise.
- motmetrics' `iou_matrix` breaks on NumPy 2 (`np.asfarray`); `research/tools/mot.py` has a replacement.
- Never commit per-frame dumps (`frames.json`, ~12 MB per run); they are gitignored under `research/`.

## Open questions
- **Ball model padding:** 1280×1280 for a 1280×720 frame wastes 44% of a 13 ms forward pass. Exporting at 736×1280 needs the training checkpoint (`notebooks/training/02_*`), and the H9b lesson says it must be re-validated (ball recall), not assumed equal.
- **Two detectors on one 4 GB GPU:** run them as one batch? They are different models, so not directly. Scheduling the ball forward between player frames, or a CUDA stream per model, is untested.
- **Remaining CPU on the main thread:** Ultralytics `predict` overhead (per-call `setup_source`, Results objects), team classification (one k-means `predict` per crop), motion bookkeeping. Not yet profiled after H6d.
