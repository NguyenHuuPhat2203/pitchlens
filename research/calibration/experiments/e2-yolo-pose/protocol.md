# E2: a shippable keypoint model trained on SoccerNet only (H3)

## Hypothesis
- **H3**: A YOLO26s-pose model (our weights, Ultralytics/TensorRT stack) trained **only** on SoccerNet-Calibration-2023 train, with 32 Roboflow-layout keypoints projected through homographies fitted to the human line annotations, then a RANSAC homography, matches the zero-shot reference:
  - TS-WorldCup test Acc@1m within 10 points of PnLCalib SV, and median error <= 1.3x PnLCalib's;
  - registers >= 6 of 8 clean end-on sample.mp4 frames (overlay on the painted lines);
  - >= 5x lower latency than PnLCalib (both on the RTX 3050 Ti, before TensorRT).
- Mechanism: registration is mostly *knowing which marking is which* from the whole view. Multi-view SoccerNet labels (incl. behind-the-goal cameras) let a one-stage pose model learn that context; a homography over >= 6 spread keypoints averages out their pixel noise.

## Data (fixed)
- Labels: `code/build_dataset.py`. Homography per image fitted to straight-line and circle annotations (DLT + LM, pixel residuals); a fit is used only if RMS <= 3 px and max residual <= 10 px (960 x 540). Keypoints visible iff in front of the camera and inside the image; >= 4 visible required.
- Train = SoccerNet train; YOLO val (model selection) = SoccerNet valid. No TS-WorldCup, WC14 or sample.mp4 data in training.
- E1 showed the target view (high camera behind a goal, looking along the pitch length) is the failure mode of public models. Before training, report (exploratory) the share of accepted SoccerNet train images whose view direction on the pitch is within 37 degrees of the length axis (|cos| > 0.8), to know how well the data covers that view.

## Training (fixed)
`yolo26s-pose.pt` init, kpt_shape [32, 3], imgsz 640, 40 epochs, patience 10, batch 16 (8 if out of memory), `mosaic=0` (virtual keypoints need the whole view), `fliplr=0` (a mirrored end-on view would need a different keypoint permutation than a mirrored side view), `scale=0.3`, `translate=0.1`, other defaults. Best checkpoint by Ultralytics fitness on SoccerNet valid.

## Evaluation (fixed)
- Same code and metrics as E1 (`tools/fieldreg.py`): TS-WorldCup test, WC14 test, sample.mp4 frames 0, 60, 120, 180, 230, 1330, 1400, 1480 (+ 400, 800, 1100).
- Inference: imgsz 640, most confident pitch box, keypoints with conf >= 0.5, pitch -> image RANSAC at 4 px per 1280 width, >= 4 points.
- Also SoccerNet **test** (not used for selection): projection error against the fitted homographies, taking the better of the two 180-degree-symmetric labelings (SoccerNet's own ambiguity rule).

## Decision rule
H3 supported if all three bullets hold. If the model registers end-on frames but misses the accuracy bar, the next experiment is line-based refinement (dense evidence) rather than a bigger network.
