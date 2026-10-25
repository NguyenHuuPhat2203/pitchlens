# Research log

## 2026-10-19 — Bootstrap
- Question: why is the pipeline at ~17–21 fps when published YOLO26s TensorRT latency is 2.5 ms? Workspace under `research/` (the project root `src/`/`data/` are product code/data).
- Proxy metric locked: end-to-end ms/frame on sample.mp4, JSON-only, model load excluded; guard = detection recall ≥ 0.99 at IoU 0.5 vs baseline.
- Hypotheses H1–H5 recorded in `research-state.yaml`.
- No `/loop` scheduler exists in this harness; continuity is via these files.

## 2026-10-19 — H1/H2 latency breakdown (run_001)
- H1 refuted: player raw inference 14.9 ms = 55% of track (predicted ≤ 40%).
- H2 supported: ball raw inference 83% of predict.
- Decision: test TensorRT next (H3), since GPU compute dominates. Added H6 (OC-SORT 6.5 ms).
- Installed `tensorrt-cu12-libs 10.16.1.11` into `.venv` (ORT 1.26 TRT EP links `nvinfer_10.dll`).

## 2026-10-19 — H3 / H3b TensorRT (run_002, run_003)
- Raw TRT FP16: player 14.77 → 8.10 ms, ball 22.64 → 13.30 ms (1.7–1.8×; H3 predicted ≥ 2× → inconclusive). Cold engine build ~20 min total, then cached.
- End to end, interleaved B/T/B/T: 52.31 → 39.85 ms/frame (−23.8%), guards hold → H3b supported.
- Shipped as `model.tensorrt` (default on, falls back to CUDA). Found that Ultralytics rebuilds its predictor when `device`/`quantize`/`tracker` change, so the swap must follow a real first call.

## 2026-10-19 — H4 parallel ball (run_004)
- 40.05 → 36.93 ms/frame, identical outputs. Shipped.

## 2026-10-19 — Outer loop 1→2: bottleneck moved to CPU (run_005–run_007)
- cProfile: OC-SORT ~8 ms, CPU letterbox ~3 ms, post ~1.5 ms; GPU forward 8.8 ms.
- H6b (cheaper tracker) refuted on SportsMOT football GT: ByteTrack −4 ms but +35% ID switches.
- H8 (re-enable OpenCV threads) refuted: contention, unstable (+2.5 ms mean).
- Direction: deepen on removing CPU work (H7b GPU letterbox, H9 smaller player input with GT guard, vectorised OC-SORT).

## 2026-10-24 — Environment
- Project moved to `D:/Side_Projects/football-tracker`; `.venv` was gone. Recreated with the measured versions (torch 2.11.0+cu128, onnxruntime-gpu 1.26.0, tensorrt-cu12-libs 10.16.1.11) and pinned `ultralytics==8.4.182`, because our code patches its internals.
- The move left `.cache/Ultralytics/settings.json` pointing at the old folders (seeded once). Fixed `src/__init__.py` to re-point them on every import, with a regression test.
- Harness scripts written before H6d use the old `Tracker(model, tracker_config, ...)` API. Re-run them at their own commit.

## 2026-10-24 — H7b GPU letterbox (run_008)
- Input parity: player ±1 grey level on 5.2% of values; ball bit-identical. End to end 37.81 → 32.45 (−5.4 ms, predicted ≥ 2). Guards hold. Shipped.

## 2026-10-24 — H9 input shape (run_009, run_010)
- Provenance confirmed (re-export max Δ 6e-5). 768×768: IDF1 −5.9 → refuted. 544×960: eligible, but its predictions failed (accuracy changed, forward −26% < 30%). End to end −1.38 ms < 1.5 ms → not shipped. Exports deleted (regenerate with `code/export.py`).

## 2026-10-24 — H6c / H6d batched OC-SORT (run_011, run_012)
- H6c: replay exact, but frames.json differed at the image border (missing `Results.update` clip), and the gain was 2.65 ms < 3 → refuted as locked.
- Exploratory: batched the direction cost, velocities and Kalman update after checking bit-exactness (port 2.29 → 1.78 ms on the synthetic scene).
- H6d: replay exact, frames.json byte-identical, −4.54 ms → supported and shipped.

## 2026-10-24 — Outer loop 2→3
- 52.3 → 27.6 ms/frame shipped (−47%). Remaining levers: the ball model's padding (needs training weights), contention between the two detectors on one GPU, and main-thread CPU outside the tracker. Direction: broaden (profile what's left), conclude this round.
