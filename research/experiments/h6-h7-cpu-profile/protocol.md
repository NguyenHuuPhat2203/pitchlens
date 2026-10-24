# H6/H7: where do the ~13.6 CPU ms inside `player_track` go?

## Context
With TRT, `player_track` ≈ 21.7 ms, of which the TRT run is ≈ 8.1 ms. The remaining ~13.6 ms is CPU work. H1 suggested ~5.6 ms Ultralytics predict overhead and ~6.5 ms OC-SORT.

## Hypotheses
- H6: OC-SORT update ≥ 5 ms/frame, dominated by per-track Python Kalman steps.
- H7: Ultralytics preprocess + postprocess + `Results` construction ≥ 4 ms/frame.

## Method
- `Tracker.track` on frames 100–399 of sample.mp4 with the product config (TRT FP16, conf 0.3), after 10 warm-up frames.
- `cProfile` over the 290 measured calls; report cumulative ms/call for: `predictor.preprocess`, `predictor.inference`, `predictor.postprocess`, the tracker `update` (on_predict_postprocess_end callback), the `Results` / `Boxes` construction, and the `.cpu().numpy()` conversions in `Tracker.track`.
- Also report tracks per frame (OC-SORT cost should scale with it).

## Decision rule
The largest CPU block decides the next intervention: H6 → a faster association/Kalman implementation; H7 → a thin ORT wrapper for the end2end output. Both get their own protocol with an output guard; this run only measures.
