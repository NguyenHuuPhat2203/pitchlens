# H4: run ball detection concurrently with player tracking

## Hypothesis
On even frames, the ball detector (≈18 ms with TRT) and `player_track` (≈22 ms) run back to back. Each contains CPU phases (Ultralytics letterbox/tensor build, result objects; OC-SORT for the player) during which the GPU idles. ORT releases the GIL inside `session.run`, so a second thread can keep the GPU busy. Prediction: ≥ 25% of the ball cost is hidden, i.e. end-to-end ms/frame drops by ≥ 0.25 × 9 ≈ 2.3 ms from the TRT baseline (≈ 39.9 ms).

Competing explanation (would refute): torch preprocessing runs on the legacy default CUDA stream, which implicitly synchronises with other streams, and Ultralytics/OC-SORT Python code holds the GIL. Overlap would then be small (< 1 ms) or negative (contention).

## Method
- Harness subclass of `Pipeline` (no product change): on even frames, submit `detect_balls(image)` to a 1-worker thread pool before `player_track`; `_update_ball` consumes the future.
- Same proxy as H3b: default config + TRT, JSON-only, 1505 frames; interleave S (serial, current product) and P (parallel) as S, P, S, P.
- Guard: P outputs identical to S (same models, same inputs; only ordering changes). Expect player recall 1.0 / ball and shots identical.

## Decision rule
Supported if mean P ≤ mean S − 2.3 ms and the guard is exact. Supported with a smaller gain (≥ 1 ms) → consider. Otherwise refuted; record the mechanism.
