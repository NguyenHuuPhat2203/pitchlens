# H7b: letterbox on the GPU

## Context
In Ultralytics 8.4.182, `BasePredictor.preprocess` already moves uint8 to the GPU and does BGR→RGB, CHW, fp16 and /255 there. Only `pre_transform` runs on the CPU: `LetterBox` = `cv2.resize` (1280×720 → 960×540 for the player model) + `cv2.copyMakeBorder` (pad to 960×960; to 1280×1280 for the ball model, which needs no resize). run_005 measured resize 2.18 + pad 0.95 ms per player call on the main thread. The ball pad (4.9 MB) runs in the ball worker thread.

## Hypothesis
Uploading the raw frame and letterboxing with torch (`F.interpolate` bilinear, align_corners=False, rounded to integers like cv2's uint8 output; `F.pad` with 114) removes that CPU work. The upload size is unchanged for the player (960² = 1280×720 pixels) and smaller for the ball. Prediction: end-to-end ms/frame drops by ≥ 2 ms vs the current product (TensorRT + parallel ball).

Geometry is taken from `LetterBox.get_params`, so `scale_boxes` maps detections back exactly as before. The only numeric difference is cv2's fixed-point bilinear vs torch float bilinear + round, expected at ±1 grey level on some pixels of the resized player input. The ball input needs no resize and should be bit-identical.

## Method
1. Input parity on 200 frames of sample.mp4 (frames 100–299): fraction of input values that differ and max |Δ| (in /255 units), player and ball.
2. Harness: patch `predictor.preprocess` on both detectors after the first real call (same constraint as `enable_tensorrt`). Product config, JSON-only, 1505 frames, interleaved S (current) / G (GPU letterbox) × 2.
3. Guard G vs S: player recall ≥ 0.99 and mean IoU ≥ 0.99 at IoU 0.5; ball in-play rate within 2 pp; shots compared (EXPLORATORY: threshold-sensitive).

## Decision rule
Supported if mean G ≤ mean S − 2 ms and the guards hold → ship as the default preprocessing for ONNX detectors on CUDA. 0 < gain < 2 ms with guards holding → still ship (the change only removes work), recorded as below prediction. Otherwise refuted.
