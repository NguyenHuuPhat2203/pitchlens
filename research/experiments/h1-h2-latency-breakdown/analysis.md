# H1/H2 analysis — run_001 (CONFIRMATORY)

Medians over 190 frames (frames 110–299 of sample.mp4), CUDA EP, FP16 ONNX.

| Stage | Player (960) | Ball (1280) |
|---|---:|---:|
| `session.run` numpy I/O | 14.87 ms | 28.61 ms |
| `session.run` I/O binding (input on GPU) | 13.90 ms | 26.88 ms |
| Ultralytics `predict` end to end | 20.48 ms | 34.44 ms |
| `Tracker.track` (predict + OC-SORT) | 27.00 ms | — |

Derived:
- Player: raw run is 55% of `track`. Ultralytics pre/post adds 5.6 ms; OC-SORT adds 6.5 ms (CPU).
- Ball: raw run is 83% of `predict`.
- I/O binding saves only ~1–1.7 ms, so host↔device copies are not the bottleneck.

## Verdicts
- **H1 refuted.** Raw inference is 14.9 ms (> 12 ms), 55% (> 40%) of `track`. Work outside the session is substantial (12 ms), but the GPU compute itself is the largest single term.
- **H2 supported.** 83% of the ball cost is the 1280×1280 session run.

## Sanity / caveats
- The `preprocess` column (45 / 114 ms) is a harness artefact: numpy float16 division is slow. Ultralytics normalises on the GPU with torch. Excluded from conclusions.
- In the full pipeline `player_track` measured 33 ms vs 27 ms here [EXPLORATORY]. Likely contention: in the pipeline the ball model and the player model alternate on the same GPU, and the frame order differs. Not yet tested.

## What this means
Pure GPU inference is ~14.9 + 28.6/2 ≈ 29 ms per frame, the largest share of the ~48 ms JSON-only frame. The highest-leverage lever is a faster inference engine (H3, TensorRT). Second: OC-SORT's 6.5 ms CPU cost (new H6). Copy elimination (I/O binding) is ruled out as a meaningful win.
