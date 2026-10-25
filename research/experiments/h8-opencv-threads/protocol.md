# H8: restore OpenCV threading that Ultralytics disables

## Observation (EXPLORATORY, from run_005 follow-up)
`ultralytics/utils/__init__.py:176` runs `cv2.setNumThreads(0)` at import, "incompatible with PyTorch DataLoader". The pipeline never uses a DataLoader, yet every OpenCV call in the process runs single-threaded: letterbox `cv2.resize` (2.2 ms) + `copyMakeBorder` (0.95 ms) per player call, the 1280 ball letterbox, team-classifier colour conversions, renderer drawing.

## Hypothesis
Re-enabling OpenCV's default thread pool after Ultralytics is imported cuts the CPU letterbox and other cv2 work. Prediction: end-to-end ms/frame drops by ≥ 1.5 ms vs the current product (36.9 ms) with **identical** outputs (same operations; OpenCV's parallel resize is deterministic).

Risk / competing outcome: oversubscription. ORT/TensorRT, the decode thread, and the ball worker thread already compete for the laptop's cores, so extra cv2 threads could add contention and make things no faster or slower.

## Method
- Default config + TRT + parallel ball (current product), JSON-only, 1505 frames.
- Variant M: `cv2.setNumThreads(-1)` (OpenCV default) right after `Pipeline(...)` is constructed, so Ultralytics' import-time setting is overridden. Log `cv2.getNumThreads()` in both variants.
- Interleave S, M, S, M. Guard: M outputs identical to S.

## Decision rule
Supported if mean M ≤ mean S − 1.5 ms with an exact guard. Then set it in the product right after the detector models load, with a comment pointing here.
