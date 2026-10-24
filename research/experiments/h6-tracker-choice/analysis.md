# H6b analysis — run_006 (CONFIRMATORY)

SportsMOT `v_gQNyhv8y0QY_c013` (football, 875 frames), the same TRT detector for all trackers, two interleaved passes. The metrics are identical across passes (deterministic); speeds are the mean of both passes.

| Tracker | IDF1 | MOTA | ID switches | FP | FN | `track` ms (SportsMOT) | `track` ms (sample.mp4) |
|---|---:|---:|---:|---:|---:|---:|---:|
| OC-SORT (current) | **61.27** | **60.54** | **100** | 3338 | 728 | 19.80 | 21.40 |
| ByteTrack | 61.04 | 59.86 | 135 | 3318 | 785 | 16.58 | 17.24 |
| FastTrack | 58.92 | 59.63 | 144 | 3313 | 805 | 17.14 | 18.10 |

(FP is high for all trackers because the COCO-person detector also boxes people SportsMOT does not annotate. It is the same detector for all.)

## Verdict
**Refuted.** ByteTrack saves 3.2–4.2 ms (prediction met), and its IDF1 is within 0.23, but it makes **1.35× the ID switches** (rule: ≤ 1.2×). FastTrack fails IDF1 (−2.35) and switches (1.44×).

## What it rules out / suggests
- OC-SORT's extra cost (velocity-direction consistency, observation-centric re-update) is not waste: it removes ~26% of identity swaps on football footage. The tactical timeline attributes possession by `track_id`, so swaps turn into fake passes/turnovers. Keep OC-SORT.
- Speeding up the tracker must keep OC-SORT's association logic: e.g. a vectorised Kalman/association implementation with identical outputs. That is a code-level project, not a config switch. Deferred.
- I did not tune ByteTrack's thresholds to chase the guard. That would be post-hoc fitting on the only labelled sequence.
