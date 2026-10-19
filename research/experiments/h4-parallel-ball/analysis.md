# H4 analysis — run_004 (CONFIRMATORY)

Default config with TRT, JSON-only, 1505 frames, interleaved.

| Run | Variant | ms/frame | `player_track` median | `ball_detect` median* |
|---|---|---:|---:|---:|
| 1 | S (serial) | 40.06 | 21.57 | 17.80 |
| 2 | P (parallel) | 37.01 | 30.55 | 0.01 |
| 3 | S | 40.03 | 21.85 | 18.01 |
| 4 | P | 36.84 | 29.68 | 0.01 |

\*In P the main thread only waits on an already-finished future, so `ball_detect` measures the wait, not the work.

Mean S 40.05 → mean P 36.93 ms/frame: **−3.12 ms (−7.8%)**. Ball cost per frame ≈ 9 ms, so ~35% is hidden (predicted ≥ 25%).

Guard: P vs S is exact (player recall 1.0 / IoU 1.0, same ball states, same shots).

## Verdict
**Supported.**

## Mechanism
The overlap is partial. `player_track` rises from 21.7 to ~30 ms because both models now share the GPU (and the GIL during Python pre/post work), so the ball's 18 ms is not hidden for free. The net win (3.1 ms) is the CPU-side idle time that the second thread fills. The competing explanation (legacy default-stream sync makes overlap ≈ 0) is ruled out: there is a measurable net gain. But contention is real, which caps this approach: running more work concurrently on a 4 GB laptop GPU will not scale linearly.
