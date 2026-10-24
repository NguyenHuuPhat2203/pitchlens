# H6c: batched OC-SORT with identical outputs

## Context
H6b showed that swapping OC-SORT for a cheaper tracker costs identity quality (+35% ID switches), so the tracker logic must stay. H6/H7 (run_005) put Ultralytics' OC-SORT at ~6.5–9 ms per frame. That time is per-track Python work: one `STrack` object per detection, per-track Kalman update, per-track velocity and direction-cost loops, properties recomputed tens of thousands of times. On top of that comes the `model.track` callback path (Results slicing, Boxes rebuild).

## Hypothesis
A port that keeps Ultralytics' pools, association order and edge cases but batches the numerics (`src/serve/pipeline/ocsort.py`), fed by `model.predict` instead of `model.track`, produces identical tracks and cuts end-to-end ms/frame by ≥ 3 ms vs the current product.

## Method
1. Equivalence, synthetic (`tests/test_ocsort.py`): Ultralytics `OCSORT` vs the port on 4 seeded 400-frame scenes, exact array equality per frame. Branch coverage and three mutants were checked before this protocol: all mutants fail, the original passes.
2. Equivalence, real: record the product detector's output (TRT FP16, conf 0.3, person) for SportsMOT `v_gQNyhv8y0QY_c013` (875 frames) and sample.mp4 (1505 frames). Replay both trackers on them; outputs must be exactly equal. Also report the median per-frame `update` time of each.
3. End to end: product config, JSON-only, 1505 frames, interleaved S (current `Tracker`) / V (predict + port) × 2.

## Decision rule
Supported if (1) and (2) are exactly equal, frames.json of V equals S, and mean V ≤ mean S − 3 ms. Then ship: `Tracker` = predict + `OCSort`, and drop the tracker YAML path.
