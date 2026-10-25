# No Bells, Just Whistles (NBJW) and PnLCalib
- Authors: Marc Gutiérrez-Pérez, Antonio Agudo
- Venues: CVPR Workshops 2024 (NBJW, pp. 3325-3334); Computer Vision and Image Understanding 2026 (PnLCalib, extension)
- URLs: https://arxiv.org/abs/2404.08401 · https://github.com/mguti97/No-Bells-Just-Whistles (licence NOASSERTION) · https://github.com/mguti97/PnLCalib (GPL-2.0)

## Key findings
- Generates a dense keypoint set (57 points: line intersections, tangency and conic points) from SoccerNet line annotations, and detects keypoints plus line extremities with two HRNetV2-W48 heatmap models.
- Calibration by DLT (NBJW), then PnLCalib adds a non-linear points-and-lines refinement and lens distortion.
- Strong results on SoccerNet-Calibration, WC14 and TS-WorldCup in both 3D calibration and homography estimation.
- Released weights: single-view `SV_kp`, `SV_lines` (SoccerNet), fine-tuned WC14/TSWC/WorldPose variants; each ~265 MB.

## Relevance
The practical SOTA with public weights, so it is the reference to beat or match. GPL-2.0 (PnLCalib) and heavy HRNet-W48 models make it a poor fit to ship inside this AGPL/Ultralytics pipeline; use it as an upper-bound baseline only.
