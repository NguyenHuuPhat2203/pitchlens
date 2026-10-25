# E3 analysis - training-free tracking (CONFIRMATORY)

Started from the GT homography of each sequence's first frame; no overlay mask.

| Variant | sample Acc@1m | sample median / p95 / max | 12-frame displacement median / p95 | TS-WC Acc@1m | TS-WC median |
|---|---:|---:|---:|---:|---:|
| ORB + line alignment (H4) | **100 %** | 0.254 / 0.427 / 0.619 m | 0.064 / 0.199 m | 51.7 % | 0.98 m |
| ORB only | 32.9 % | 1.09 / 2.33 / 2.42 m | 0.154 / 0.421 m | 39.0 % | 1.45 m |
| Line alignment only | **100 %** | **0.222 / 0.324 / 0.327 m** | **0.012** / 0.173 m | 18.9 % | 4.17 m |

GT vs its own 5-frame moving average: 0.004 m (12-frame displacement), so the dense GT is far smoother than any tracker error above.

## Verdict
**H4 refuted** by its own rule: the three sample.mp4 criteria hold with a wide margin (100 % Acc@1m, median 0.25 m, displacement 0.064 m vs <= 0.15 m), but TS-WorldCup reaches 51.7 % Acc@1m, not 80 %.

## What it means
- **On the target footage, calibration can be tracked without any model**: 425 frames, every frame within 0.62 m, and a 0.48 s speed-window error of 1-6 cm (0.1-0.5 km/h). Line alignment alone is the smoothest variant at 25 fps (per-frame motion stays inside its 6 px capture range).
- **Why TS-WorldCup breaks it** (per-sequence errors in `results/rows_*.json`):
  1. Its frames are sub-sampled (3-19 px motion per step, about 5 fps), outside the line-only capture range - lines-only loses track within 1-35 frames.
  2. Side views often show 1-2 marking families: line alignment then leaves directions unconstrained and drifts along them; in one sequence ORB alone stays at 0.25 m while ORB + lines drifts to 0.72 m.
  3. Static TV logos bias ORB toward zero motion when the camera moves slowly.
- Design consequence: use the ORB prediction as a prior inside the alignment (lines correct only what they observe), mask static overlays automatically, and watch the inlier score to detect loss and re-initialise.
- The blocking piece for full automation is **initialisation** on the end-on view (E1: no public model does it). Tracking is solved for the product's frame rate.
