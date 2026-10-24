# H6d analysis — run_012 (CONFIRMATORY)

## Replay of the product detector's output (exact equality required)
| Sequence | Frames | Mismatched | Ultralytics `OCSORT.update` | Port v2 |
|---|---:|---:|---:|---:|
| SportsMOT | 875 | 0 | 3.61 ms | 1.50 ms |
| sample.mp4 | 1505 | 0 | 4.85 ms | 1.86 ms |

## End to end (product: TensorRT, GPU letterbox, parallel ball; JSON-only)
| Run | Variant | ms/frame | `player_track` |
|---|---|---:|---:|
| 1 | S (`model.track`) | 32.04 | 26.78 |
| 2 | V (`predict` + port v2 + clip) | 27.34 | 22.51 |
| 3 | S | 32.15 | 26.78 |
| 4 | V | 27.76 | 22.86 |

Mean S 32.09 → V 27.55: **−4.54 ms (−14.1%)**. frames.json V == S: **byte-identical** (same tracks, IDs, boxes, teams, ball, events).

## Verdict
**Supported** (prediction ≥ 3 ms) and **shipped** (exact replay, identical output, gain ≥ 2 ms).

## Mechanism
The tracker update itself saves ~3 ms per frame (4.85 → 1.86 on sample.mp4). The remaining ~1.5 ms comes from dropping Ultralytics' track callback: `result[idx]` slicing, rebuilding `Boxes` with a fresh CUDA tensor, then copying it back to the CPU in our wrapper. The main thread is the critical path, so CPU removed there is the end-to-end gain. The ball worker's time is unchanged (21.2 ms).

## Lesson
"Identical outputs" has to be checked at the product boundary, not only at the component boundary. The H6c port was exact at the tracker level, but the product output also depended on a post-tracker step (`Results.update` clipping) in the code path being replaced.
