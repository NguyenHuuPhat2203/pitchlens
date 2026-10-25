# H6/H7 analysis — run_005 (CONFIRMATORY)

`cProfile` over 290 `Tracker.track` calls (TRT FP16, 23.2 tracks/frame). Values are cumulative ms per frame. cProfile inflates code made of many small Python calls, so the tracker numbers are upper bounds; H1's unprofiled estimate was 6.5 ms.

| Block | ms/frame | Notes |
|---|---:|---|
| `inference` → TRT `forward` | 8.80 | GPU |
| Tracker callback (`on_predict_postprocess_end` → `ByteTracker.update`) | 9.24 | OC-SORT |
| ↳ `_first_association` | 6.06 | |
| ↳ per-track `oc_sort.update` (6,639 calls) | 3.38 | Kalman `update` 2.0 (`linalg.solve` 0.8 on 4×4) |
| ↳ `get_dists` + `_velocity_direction_cost` | 2.53 / 2.01 | OC-SORT-specific |
| ↳ `tlwh`/`xyxy` properties | ~1.5 | 94k calls |
| `preprocess` | 4.37 | letterbox: `cv2.resize` 2.18 + `copyMakeBorder` 0.95 |
| `postprocess` + results | 1.53 | |

## Verdicts
- **H6 supported**: the tracker is ≥ 5 ms (largest CPU block), made of per-object Python overhead rather than heavy math.
- **H7 supported**: pre + post = 5.9 ms ≥ 4 ms. The single biggest piece is the CPU letterbox (3.1 ms).

## What it suggests
1. Tracker cost is O(tracks) Python calls. Cheaper association (H6b) or a vectorised implementation would cut it.
2. Letterbox on the GPU (upload uint8 1280×720, resize+pad with torch) would remove ~3 ms of CPU from every player call. The output changes slightly (different interpolation), so it needs a parity guard.
