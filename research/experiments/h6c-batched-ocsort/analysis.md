# H6c analysis — run_011 (CONFIRMATORY)

## Equivalence
- Synthetic (`tests/test_ocsort.py`, 4 × 400 frames): exact. Branch coverage: 175 ORU replays, 2,766 OCR matches, 18 resurrections of Removed-state tracks, 14 negative-x observations, duplicate removal. Three mutants (no removal delay, no ORU, no duplicate removal) all fail.
- Real replay of the product detector's output (input dtype float32): **0 mismatched frames** on SportsMOT (875) and sample.mp4 (1505).

## Tracker cost (median per frame, replay)
| Sequence | Ultralytics `OCSORT.update` | Port |
|---|---:|---:|
| SportsMOT | 3.62 ms | 1.92 ms |
| sample.mp4 | 4.87 ms | 2.46 ms |

## End to end (product: TensorRT, GPU letterbox, parallel ball)
| Run | Variant | ms/frame | `player_track` |
|---|---|---:|---:|
| 1 | S | 32.04 | 26.55 |
| 2 | V | 28.57 | 23.31 |
| 3 | S | 31.84 | 26.59 |
| 4 | V | 30.01 | 23.69 |

Mean S 31.94 → V 29.29: −2.65 ms. frames.json: **not identical** (56 of 1505 frames; player recall 1.0, mean IoU 0.9999).

## Verdict
**Refuted as locked.** The gain was below 3 ms, and the identity guard failed.

## Why
1. **Identity**: the candidate tracker returned Kalman boxes as is. Ultralytics' track callback passes them through `Results.update`, which clips boxes to the image. All 56 differing frames differ only at the image border (e.g. x2 = 1280.6 vs 1280.0, x1 = −0.13 vs 0.0). This was a bug in the candidate's integration, not in the port: the replay, which compares tracker outputs before that callback, is exact.
2. **Speed**: the prediction assumed Ultralytics' tracker cost 6.5–9 ms (run_005, under cProfile). Unprofiled it costs 3.6–4.9 ms; the rest of the earlier estimate was profiler overhead. The port's first version (1.9–2.5 ms) still ran per-track Python for the direction cost, velocities and the Kalman update.

## Follow-up (EXPLORATORY, before H6d)
- Batched versions of those three per-track loops were checked bit-for-bit against the per-track NumPy/BLAS calls: 3,000 random trials for the direction matmul and `norm = sqrt(dot)`, 2,000 for the Kalman dots, 0 mismatches. They then passed the equivalence tests and the mutants. Port: 2.29 → 1.78 ms/frame on the synthetic scene.
- The missing clip goes into the candidate tracker. This becomes H6d, with its own locked rule.
