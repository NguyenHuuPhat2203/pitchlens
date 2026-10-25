# E5: the shipped configuration, end to end on sample.mp4 (H6)

## Hypothesis
- **H6**: `src/serve/pipeline/auto_calibrator.py` (`AutoCalibrator`: line search at full resolution to start or after loss, line alignment at 640 px width with an analytic Jacobian to track, frame by frame), started cold on frame 0 and run over all 1,505 frames of data/sample.mp4:
  - on the 425 dense-GT frames: Acc@1m >= 95 %, median projection error <= 0.30 m, 12-frame displacement error median <= 0.10 m;
  - a homography on >= 95 % of all 1,505 frames (the 1,080 frames with broadcaster graphics included);
  - tracking step <= 30 ms median per frame (CPU, measured while the GPU trains E2, so the CPU is shared).
- Mechanism: E4 shows the search either answers correctly or refuses on this view; E3 shows line-only tracking stays within 1 m at 25 fps. Half resolution keeps the line centres (1-2 px wide at full size) visible while cutting the mask cost about 4x; the analytic Jacobian avoids 8 finite-difference evaluations per iteration.

## Method (fixed)
- `AutoCalibrator()` defaults at this commit: track width 640, min inlier share 0.3 (else lost, search again at once), search retry every 50 frames while lost; search acceptance F >= 0.55 and >= 80 matched samples (E4 values, resolution-scaled parameters equal E4's at 1280 px).
- Errors: projection error against `gt-sample/gt_dense.json`, the better of the two 180-degree labellings per frame (the search may pick either goal). Displacement: frames n and n + 12 with both predicted and GT homographies, same labelling as the frame-n choice.

## Decision rule
H6 supported if all four numbers hold. Ship it as the pipeline's calibration if (a) and (b) hold even when (c) misses; then (c) decides whether calibration runs every frame or on a stride.
