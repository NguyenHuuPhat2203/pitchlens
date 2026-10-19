# H3b: does the TensorRT gain survive end to end?

## Hypothesis
Swapping both detectors' ORT sessions to the TensorRT EP (FP16 graphs, cached engines) cuts end-to-end ms/frame by ≥ 20%. Reasoning: raw inference falls ~6.7 ms (player) + ~4.7 ms/frame (ball, every other frame) ≈ 11 ms out of ~48 ms ≈ 23%, provided Ultralytics pre/post and OC-SORT are unaffected.

## Method
- Full default config, `display=false`, `output_video=null`, frames JSON + timeline on, `data/sample.mp4` (1505 frames).
- Pipeline constructed and both models warmed on one dummy frame before timing; `run()` wall time / frames.
- TRT variant: after warm-up, replace `backend.session` with a TRT EP session on the same FP16 ONNX file and rebind I/O to the same output tensors (harness-only swap; no product change yet).
- Order B, T, B, T to separate the treatment from GPU thermal drift. Report each run and the mean.
- Guard (T vs B, per frame): players matched with recall ≥ 0.99 at IoU 0.5; ball `in_play` frame count within 2 percentage points; shot frame list compared (EXPLORATORY: shots are rare and threshold-sensitive).

## Decision rule
Supported if the mean T is ≥ 20% lower than the mean B and the guards hold. Then implement the provider choice in the product (config `model.engine: tensorrt|cuda`).
