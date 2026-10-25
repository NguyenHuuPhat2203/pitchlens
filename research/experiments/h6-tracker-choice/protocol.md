# H6b: cheaper tracker without worse identities

## Context (run_005, CONFIRMATORY for H6/H7)
- OC-SORT `update` ≈ 8.4 ms/frame under cProfile (~6.5 ms unprofiled, H1), dominated by per-track Python Kalman steps plus OC-SORT's velocity-direction cost and observation-centric re-update (`_velocity_direction_cost` 2.0 ms, `_compute_velocity` 0.7 ms).
- Preprocess 4.4 ms (CPU letterbox resize 2.2 + pad 0.95); postprocess 1.5 ms.

## Hypothesis
ByteTrack-family trackers (Ultralytics `bytetrack`, `fasttrack`) skip OC-SORT's velocity-direction cost, so they cut `player_track` by ≥ 2 ms/frame. Prediction on identity quality: IDF1 within 2 points of OC-SORT, since broadcast football at 25 fps is mostly linear motion between frames.

## Method
- Ground truth: SportsMOT `v_gQNyhv8y0QY_c013` (football, 875 frames, 1280×720, 25 fps, 10,557 boxes).
- Same detector for all: product player model, TRT FP16, conf 0.3, imgsz 960, person class.
- Trackers: `ocsort.yaml` (baseline), `bytetrack.yaml`, `fasttrack.yaml`, with Ultralytics defaults.
- Metrics with `motmetrics`: IDF1, MOTA, ID switches, at IoU ≥ 0.5. The COCO-person detector also fires on non-annotated people (referees, staff), which lowers MOTA equally for all trackers; compare relative values.
- Speed: median `Tracker.track` ms over the sequence, plus the same on sample.mp4 frames 100–399. Order: O, B, F, O, B, F (two passes) to control drift.

## Decision rule
Adopt a tracker if it is ≥ 2 ms faster AND IDF1 ≥ OC-SORT − 2 AND ID switches ≤ OC-SORT × 1.2. If none qualifies, H6b is refuted; the remaining CPU lever is the tracker implementation itself (vectorised Kalman), out of scope for a config change.
