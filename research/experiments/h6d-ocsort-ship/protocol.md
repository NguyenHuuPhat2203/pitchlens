# H6d: ship the batched OC-SORT (fixed integration)

## Change from H6c
- Tracker integration clips the output boxes to the frame, as Ultralytics' `Results.update` does in the `model.track` path.
- Port v2: direction cost, velocities and Kalman update batched. These were bit-identical to the per-track computations in random trials, and the equivalence tests still pass.

## Hypothesis
`Tracker` = `model.predict` + port v2 + clip produces **byte-identical** frames.json to the current `model.track` path. It is faster end to end by more than the H6c run: prediction ≥ 3 ms/frame, because the port v2 takes ~0.7 ms less than v1.

## Method
1. Replay the product detector's output (SportsMOT 875 frames, sample.mp4 1505 frames) through Ultralytics' `OCSORT` and port v2: exact equality required; median `update` ms for both.
2. End to end: product config, JSON-only, 1505 frames, interleaved S (current) / V (candidate) × 2. frames.json of V must equal S exactly.

## Decision rule
- Ship if (1) is exact, frames.json V == S, and the mean gain is ≥ 2 ms. A tracker that is identical in output carries no accuracy risk, so the bar is set just above run-to-run noise (~0.5 ms) and H6c's observed 2.65 ms, not at the H6c 3 ms prediction.
- The hypothesis (≥ 3 ms) is judged separately: supported only if the gain is ≥ 3 ms.
