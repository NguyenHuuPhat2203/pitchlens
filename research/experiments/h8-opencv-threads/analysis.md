# H8 analysis — run_007 (CONFIRMATORY)

Current product (TRT + parallel ball), JSON-only, 1505 frames, interleaved.

| Run | Variant | cv2 threads | ms/frame | `player_track` | `ball_detect` (worker) |
|---|---|---:|---:|---:|---:|
| 1 | S | 1 | 37.50 | 30.69 | 26.14 |
| 2 | M | 16 | 36.36 | 25.85 | 26.15 |
| 3 | S | 1 | 38.57 | 30.94 | 27.36 |
| 4 | M | 16 | **44.79** | 33.45 | 28.58 |

Mean S 38.04, mean M 40.58 ms/frame. Guard exact (identical outputs).

## Verdict
**Refuted.** No reliable gain; the mean is worse and M is unstable (36.4 → 44.8 ms between its two runs, while S moved by 1 ms).

## Mechanism
This is the competing explanation written in the protocol: CPU oversubscription. The process already runs the decode thread, the ball worker thread, ORT/TensorRT host threads and Python. OpenCV's 16-thread pool for a ~2 ms resize adds scheduling contention, which helps sometimes (run 2) and hurts badly at other times (run 4). Ultralytics' `cv2.setNumThreads(0)` is accidentally right for this workload.

## Side observations (EXPLORATORY)
- All variants run ~1–1.5 ms slower than in run_004 (36.9). The laptop warms up over a long session; this is why only interleaved comparisons are trusted.
- The parallel ball worker's `ball_detect` is ~26 ms vs 18 ms serial: GPU sharing, consistent with H4's mechanism.

## Implication
The CPU is effectively saturated. Further CPU wins must **remove** work (e.g. move letterbox to the GPU, a vectorised tracker), not parallelise it.
