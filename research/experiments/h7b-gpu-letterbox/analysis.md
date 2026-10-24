# H7b analysis — run_008 (CONFIRMATORY)

## Input parity (frames 100–299)
| Model | Input | Values differing | Max \|Δ\| |
|---|---|---:|---:|
| player | 1×3×960×960 | 5.2% | 1.12 grey levels |
| ball | 1×3×1280×1280 | 0 | 0 (bit-identical: no resize, pad only) |

The player differences are the expected ±1 grey level from OpenCV's fixed-point bilinear vs torch float bilinear + round. The 1.12 comes from the 1/255 step in fp16.

## End to end (product: TensorRT + parallel ball; JSON-only, 1505 frames)
| Run | Variant | ms/frame | `player_track` | `ball_detect` (worker) |
|---|---|---:|---:|---:|
| 1 | S | 37.90 | 30.65 | 26.36 |
| 2 | G | 32.21 | 26.91 | 21.29 |
| 3 | S | 37.72 | 31.48 | 30.01 |
| 4 | G | 32.69 | 27.34 | 21.94 |

Mean S 37.81 → mean G 32.45 ms/frame: **−5.36 ms (−14.2%)**. Prediction was ≥ 2 ms.

Guard G vs S: player recall 0.9963, mean IoU 0.9927; ball in-play 93.49% both; shots identical [514, 882, 1014, 1072, 1310]. The S-vs-S control is identical, so the pipeline is deterministic.

## Verdict
**Supported**, above prediction.

## Mechanism
The prediction counted only the player's CPU letterbox (~3.1 ms on the main thread). The extra ~2 ms comes from the ball worker. Its 1280×1280 pad (4.9 MB `copyMakeBorder`) and the larger upload (4.9 MB padded vs 2.8 MB raw) are gone, so its time drops 26–30 → 21–22 ms, and that thread no longer competes with the main thread for CPU. This matches H8: the CPU is the contended resource, so removing CPU work pays more than its own duration.

## Note
Baseline S is 37.8 ms here vs 36.9 ms in run_004 (different session, rebuilt venv, laptop thermals). Only the interleaved comparison is used.
