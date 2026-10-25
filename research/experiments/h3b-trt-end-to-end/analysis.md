# H3b analysis — run_003 (CONFIRMATORY)

The full default pipeline, JSON + timeline, 1505 frames, interleaved B/T.

| Run | Variant | ms/frame | `player_track` median | `ball_detect` median |
|---|---|---:|---:|---:|
| 1 | B (CUDA EP) | 52.52 | 28.32 | 27.12 |
| 2 | T (TensorRT EP) | 39.90 | 21.72 | 17.92 |
| 3 | B | 52.10 | 28.34 | 27.04 |
| 4 | T | 39.80 | 21.73 | 17.94 |

Mean B 52.31 → mean T 39.85 ms/frame: **−23.8%** (19.1 → 25.1 fps). Run-to-run spread is < 1%. Interleaving removed the drift seen in single-model runs.

Guard, T vs B (run2 vs run1):
- Players: recall 0.9974, mean IoU 0.9948. The B-vs-B control is 1.0 / 1.0, so the pipeline is deterministic and this is TRT numerical drift.
- Ball in-play: 93.49% in both.
- Shots: identical frames [514, 882, 1014, 1072, 1310].

## Verdict
**Supported** (≥ 20% predicted, 23.8% observed; guards hold).

## Mechanism check
The predicted saving was 6.7 + 4.7 ≈ 11.4 ms; observed 12.5 ms. In the pipeline, `player_track` fell 6.6 ms, matching the raw-inference gain (14.8 → 8.1), so Ultralytics pre/post and OC-SORT are unchanged, as assumed. Ball: 27.1 → 17.9 ms per call, ≈ 4.6 ms/frame at every other frame.

## Remaining budget (T, per frame)
`player_track` 21.7 = TRT 8.1 + Ultralytics pre/post ~5.6 + OC-SORT ~6.5 (+ ~1.5 unexplained); ball ~9.0; the rest (team, motion, timeline bookkeeping, decode wait) ≈ 9 ms.
