# E5 results: the shipped AutoCalibrator on all of sample.mp4 (CONFIRMATORY for H6)

Protocol: `protocol.md` (commit cbf2169). Cold start at frame 0, all 1,505 frames, every frame through `AutoCalibrator.update`. CPU shared with E2 training on the GPU (two dataloader workers) during every run.

| Run | Code | Frames with H | GT Acc@1m | GT median / max (m) | 12-frame disp. median / p95 (m) | Track ms (median) | Search |
|---|---|---|---|---|---|---|---|
| 1 | protocol commit cbf2169 | 100 % | 100 % | 0.268 / 0.474 | 0.039 / 0.175 | 37.1 | 1 x 12.8 s |
| 2 | + thickness fix + across-line NMS | 96.7 % | 88.2 % | 0.268 / 0.452 | 0.025 / 0.156 | 36.4 | frame 0 refused, found at 50 |
| **3 (shipped)** | + thickness fix only, line maps shared in search | **100 %** | **100 %** | **0.235 / 0.356** | **0.029 / 0.093** | 36.1 | 1 x 6.5 s |

H6: accuracy, stability and coverage hold in every run that registered frame 0; tracking cost misses the 30 ms bound (36 ms with the CPU shared). By the decision rule the configuration ships; calibration runs on its own thread, overlapped with the GPU tracker.

## Changes after the protocol run (found by a rendered-pitch unit test, not by these numbers)
- **Thickness filter**: `area / bounding-box extent` treated a connected outline (box, goal area) as one thick blob, so a clean frame produced no line centres at all. Mean thickness is now `2 * area / boundary pixels` (equal to the width of one line and of a network of lines). Run 3 shows it also helps the real clip (more evidence: max error 0.47 -> 0.36 m).
- **Across-line NMS (reverted)**: keeping only the maximum across the line thins near-vertical lines to 1 px; Hough support fell under `min_support` and frame 0's second line family shrank from 4 to 2 lines (run 2). The union of horizontal and vertical maxima stays; `min_support` is tuned for it.
- **Front-of-camera sign** (`_front`) was a no-op (`inv(H) @ (H @ x)`); now the sign of the visible bottom-centre point. Identical output on sample.mp4.
- **Line maps computed once per searched image** (were recomputed for each of the 30 refinements): identical homographies, search 12.6 -> 8.9 s on frame 0.
- **Search backoff**: a failed search waits 50, 100, 200, ... frames (at most 800) before the next, so footage with no readable pitch costs a few searches, not one every 2 s.

Raw: `results/summary*.json`, `results/rows*.json` (per frame: mode, ms, error).

## Pipeline throughput (whole sample.mp4, TensorRT, video + JSON + timeline, machine otherwise idle)
| Calibration | Run time | Steady ms/frame | Note |
|---|---|---|---|
| off | 64.8 s | 43.0 | |
| `AutoCalibrator` on a thread | ~114 s | ~69 (+26) | GIL contention: player_track 23.8 -> 30.5 ms, ball 23.8 -> 36.1 ms |
| **`CalibrationProcess`, submitted at decode (shipped)** | 85.6 s | **~52.6 (+9.6)** | run time includes the one-off 6.5 s search; `calibrate_wait` median 0.0 ms |

Player speeds over the clip: 99.2 % of player-frames have one; median 3.9, p90 10.2, p99 17.6 km/h; 2 of 32,767 above 36 km/h (max 39.1, box jitter). Shot ball speeds include 149 and 160 km/h: an airborne ball projected onto the ground plane, a limit of the existing shot-speed estimate, not of the calibration.

Rejected on dev frames 0-239: looser least-squares tolerances (1e-4 / 1e-3) cut tracking 39 -> 27 ms but doubled the 12-frame displacement error (0.030 -> 0.068 / 0.071 m).
