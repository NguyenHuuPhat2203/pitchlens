# E3: training-free tracking - ORB camera motion + line alignment (H4)

## Hypothesis
- **H4**: Given a correct homography on the first frame of a shot, tracking it frame to frame with ORB camera motion (prediction) followed by alignment of the projected markings to the painted-line centres (correction) keeps registration accurate and smooth, without any trained model:
  - sample.mp4 dense GT (425 frames: 0-239 started at 0, 1320-1504 started at 1330 in both directions as built): Acc@1m >= 95 % and median projection error <= 0.30 m;
  - speed-relevant 12-frame displacement error (0.48 s, the product's speed window) median <= 0.15 m (<= 1.1 km/h);
  - TS-WorldCup test (10 sequences, sub-sampled frames, human GT, started at each sequence's first GT): Acc@1m >= 80 %.
- Mechanism: ORB matches give the inter-frame camera rotation/zoom (all scene points move by one homography), so the prediction lands within a few pixels; line alignment removes the drift that pure visual odometry accumulates.

## Ablations (same runs)
- ORB only (no line alignment): expected to drift (error grows along the sequence).
- Line alignment only (previous frame's homography as the prediction): expected to lose track when motion exceeds the 6 px capture range.

## Method (fixed, `tools/linefit.py` at this commit)
- Line-centre mask: luminance top-hat (15 px) on low-saturation pixels inside the largest grass region (eroded 5 px), elongated pieces only, 1-D non-maximum suppression.
- Alignment: pitch markings sampled every 0.25 m, robust least squares (soft-L1, scales 3 then 1.5 px) on the distance transform, residual capped at 6 px.
- ORB: 3,000 features, ratio test 0.75, RANSAC 3 px. **No manual overlay mask** (the broadcast logos stay in; product conditions).
- Pitch 105 x 68 m for sample.mp4; WC template for TS-WorldCup.

## Ground truth and its limits
- sample.mp4 GT: `../gt-sample/` - 8 frames from hand-read rough lines (verified overlays, RMS 1.9-2.2 px) and 425 dense frames, each fitted to its own snapped line points (RMS median 2.15 px). The GT and this tracker both rely on the painted lines, so errors common to both (lens distortion, pitch dimensions) are invisible here; TS-WorldCup (independent human GT) is the cross-check.
- Reported alongside: the GT's own 12-frame displacement "noise floor" is not measurable without a second GT; we report the displacement error of the GT against a 5-frame moving average of itself as an indication.

## Decision rule
H4 supported if all three bullets hold. Then the shipped design is: initialise once per shot (E2 model, or a few user clicks on any landmarks), track with H4.
