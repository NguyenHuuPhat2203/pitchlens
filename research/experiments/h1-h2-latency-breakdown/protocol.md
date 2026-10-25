# H1/H2: where does detector time go?

## Hypotheses
- H1: raw player `session.run` ≤ 12 ms, ≤ 40% of `player_track` (33 ms). The rest is Ultralytics preprocessing on CPU, copies, postprocessing, OC-SORT.
- H2: raw ball `session.run` ≥ 60% of `ball_detect` (29 ms).

## Method
200 frames of `data/sample.mp4` (frames 100–299), CUDA, the FP16 ONNX files the pipeline uses. 10 warm-up frames are discarded. Median per stage:

1. `preprocess`: Ultralytics `LetterBox` + BGR→RGB + CHW + /255 + FP16 on CPU (reimplemented with the same ops).
2. `run_numpy`: `session.run` with numpy input/output (includes H2D/D2H copies), the way Ultralytics calls it.
3. `run_iobind`: `session.run_with_iobinding` with input already on the GPU (pure GPU compute + output copy).
4. `predict`: `YOLO.predict` (player) / `BallDetector.detect_balls`, end to end.
5. `track`: `Tracker.track` (predict + OC-SORT), player only.

## Decision rule
- H1 supported if `run_numpy` ≤ 12 ms AND `run_numpy / track` ≤ 0.40.
- H2 supported if `run_numpy / predict` ≥ 0.60 for the ball model.
- The biggest stage by median sets the next hypothesis to test.
