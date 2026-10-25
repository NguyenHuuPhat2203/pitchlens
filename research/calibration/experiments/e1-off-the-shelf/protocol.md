# E1: off-the-shelf registration models (H1, H2)

## Hypotheses
- **H1**: PnLCalib single-view weights (`SV_kp`, `SV_lines`, trained on SoccerNet-Calibration) with PnL line refinement work zero-shot: on TS-WorldCup test, completeness >= 90 % and mean projection error <= 1.0 m; on `data/sample.mp4` (end-on high view) the projected markings sit on the painted lines for the fixed frames below.
  - Mechanism: SoccerNet contains many broadcast camera types including behind-the-goal views; dense keypoints plus line refinement.
- **H2**: Roboflow `football-pitch-detection.pt` (YOLOv8x-pose, 32 keypoints, ~300 side-view training images) is clearly worse: TS-WorldCup mean projection error >= 1.5x H1, and it fails (no homography or visibly wrong) on most end-on sample frames.
  - Mechanism: tiny single-viewpoint training set; on an end-on view the keypoints' appearance is unlike anything it saw.

## Method (fixed before running)
- PnLCalib commit 8c87391; thresholds = its `inference.py` demo defaults (kp 0.3434, line 0.7867), `heuristic_voting_ground(refine_lines=True)`, input resized to 960x540. Output image -> SoccerNet ground (centred metres), shifted by (52.5, 34) into the pitch frame.
- Roboflow: predict at the checkpoint's training `imgsz`, take the most confident pitch box, keypoints with confidence >= 0.5 (>= 4 needed), homography pitch -> image by RANSAC (4 px at 1280 width), inverted.
- No threshold is tuned on any test set.
- Metrics (tools/fieldreg.py): Acc@1m over all frames (primary), mean/median projection error over registered frames, completeness, displacement error between consecutive TS-WorldCup frames, latency.
- Datasets: TS-WorldCup test (887), WC14 test (186). Sample frames: 0, 60, 120, 180, 230, 1330, 1400, 1480 (clean) plus 400, 800, 1100 (broadcaster graphics on screen).

## Decision rule
- H1 supported if both TS-WC numbers hold and >= 6 of 8 clean sample frames overlay correctly (visual check, overlays saved).
- H2 supported if its TS-WC mean error >= 1.5x H1's and <= 2 of 8 clean sample frames overlay correctly.
- Either way, E1 decides which family to build the shippable model on.
