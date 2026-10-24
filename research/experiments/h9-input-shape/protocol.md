# H9: player-detector input shape

## Context
The player ONNX has a fixed 960×960 input. A 1280×720 broadcast frame is scaled by 0.75 to 960×540 and padded to 960×960, so 44% of every forward pass processes grey padding. With TensorRT the forward pass is ~8 ms per frame. The ball ONNX (fixed 1280×1280, no `.pt` in the repo) cannot be re-exported, so this study covers the player model only.

## Hypotheses
- **H9b (rectangular, same scale)**: a 544×960 (h×w) export of the same weights keeps the 0.75 scale and drops the padding. Prediction: SportsMOT IDF1 and MOTA within ±0.5 of 960×960, and raw TRT forward time ≥ 30% lower (pixels: 0.57×).
- **H9 (as locked in research-state)**: 768×768 (scale 0.6) cuts ≥ 3 ms/frame with an IDF1 drop ≤ 1 point.
- Also measured, to map the curve: 448×768 (scale 0.6, rectangular) and 384×640 (scale 0.5, rectangular).

## Method
1. Provenance: confirm `models/player_detector.onnx` is `yolo26s.pt` exported at 960. Re-export with the same export arguments (from the ONNX metadata) and compare outputs on 20 frames (ORT CPU, FP32). Required: identical boxes up to float noise. If this fails, H9 is blocked: other sizes would be different weights, not the same model.
2. Export the variants from `yolo26s.pt` with the same arguments, `imgsz=(h, w)`.
3. For each variant, with the product path (FP16 ONNX, TensorRT, GPU letterbox, OC-SORT, conf 0.3):
   - raw TRT `session.run` median over sample.mp4 frames 100–299;
   - SportsMOT `v_gQNyhv8y0QY_c013`: IDF1, MOTA, ID switches, FP, FN (motmetrics, IoU ≥ 0.5), as in H6b.
4. End to end: current (A) vs the chosen variant, interleaved A/X × 2, JSON-only, 1505 frames.

## Decision rule
Eligible variants: IDF1 ≥ A − 1.0, MOTA ≥ A − 1.0, ID switches ≤ 1.2 × A, on SportsMOT. Ship the fastest eligible variant if its end-to-end gain is ≥ 1.5 ms. H9 (768×768) is supported only if it is eligible and its gain is ≥ 3 ms.
