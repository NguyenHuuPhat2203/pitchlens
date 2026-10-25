# Literature: automatic soccer field registration (camera calibration)

Question: can the pipeline find the pixel→pitch mapping itself, every frame, instead of 4 manual clicks?
Per-paper notes are in this folder; this file is the running summary.

## Families of methods
| Family | How it finds correspondences | Representative work | Notes for this project |
|---|---|---|---|
| Keypoint heatmaps / regression | CNN predicts named pitch points (line intersections, circle points, grid points) → DLT + RANSAC homography | Nie et al. 2021 (91-point grid), KpSFR 2022, NBJW 2024 (57 kp + lines), Sportlight 2023, Roboflow sports (YOLOv8-pose, 32 kp) | Best accuracy; needs a trained model. A YOLO-pose variant fits our Ultralytics/TensorRT stack. |
| Line / segment based | Segment pitch markings, fit lines/conics, optimise camera to align them | sn-calibration baseline (DeepLabv3), TVCalib 2023 (differentiable), PnLCalib 2024/26 (PnL refinement) | Lines give sub-pixel refinement; refinement step can be training-free once an initial estimate exists. |
| Template retrieval | Match the image's edge map to a database of synthetic views, then refine | Sharma et al. 2018, Chen & Little 2019 | Needs a large synthetic database per camera setup. |
| Learned error / direct regression | Regress homography or learn an error to optimise | Jiang et al. 2020, Sha et al. 2020 | Superseded by keypoint methods on benchmarks. |
| Temporal / PTZ tracking | Track the camera between frames (features on the static scene), re-anchor periodically | Hayet et al. 2004, Lu, Chen & Little 2019 (PTZ SLAM) | Directly relevant: our camera pans; per-frame jitter becomes speed noise. |

## Benchmarks and metrics
- **SoccerNet-Calibration 2023** (Magera et al. 2024 protocol): line-point annotations, no camera GT. JaC@5 px (a marking counts only if all its points reproject within 5 px) × completeness. 2023 leaderboard: Sportlight 0.55 (JaC@5 73.2 %, completeness 75.6 %), Spiideo 0.53 (52.9 %, 99.96 %), baseline 0.08.
- **WorldCup 2014** (Homayounfar et al. 2017): 209 train / 186 test images with GT homographies. 59 MB.
- **TS-WorldCup** (Chu et al. 2022): 3,812 time-sequence images from 43 videos (2,925 train / 887 test) with GT homographies. 615 MB.
- Homography metrics (KpSFR): IoU_part / IoU_entire of projected field masks; **projection error in metres** (2,500 pixels sampled from the visible field, projected with predicted vs GT homography); re-projection error.
- Reference numbers on TS-WorldCup test: Nie et al. 0.57 m mean / 0.51 m median (WC-trained); KpSFR 0.45 / 0.41 m (WC-trained), 0.36 / 0.33 m (fine-tuned).

## Gaps that matter here
1. Benchmarks score single frames; a speed estimate differences positions 0.5 s apart, so frame-to-frame jitter matters as much as mean error. Few papers report temporal stability.
2. Most SOTA code is research-licensed or GPL (PnLCalib GPL-2.0, NBJW unclear, Sportlight none): usable as a reference, not as shipped code here.
3. Central/close views lack keypoints (Magera et al. 2025); our footage is a high wide view, which is the easy case.
