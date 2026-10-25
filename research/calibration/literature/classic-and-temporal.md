# Earlier and temporal methods (one file, several short entries)

## Homayounfar, Fidler, Urtasun — Sports field localization via deep structured models (CVPR 2017)
- Introduces the WorldCup 2014 dataset (209 train / 186 test, GT homographies, https://nhoma.github.io/). Field lines/circles + MRF inference.
- Relevance: small standard single-frame benchmark (59 MB).

## Nie, Chen, Hamid — A Robust and Efficient Framework for Sports-Field Registration (WACV 2021)
- Grid of uniformly distributed keypoints over the whole field, so even textureless areas give correspondences; dense features for alignment refinement.
- Relevance: grid keypoints keep enough correspondences when few markings are visible.

## Chen & Little — Sports camera calibration via synthetic data (CVPRW 2019)
- Two GANs extract field edges; a siamese network retrieves the nearest view from a 90k synthetic database; Lucas-Kanade refinement.

## Jiang et al. — Optimizing through learned errors for accurate sports field registration (WACV 2020)
- Learns a registration-error predictor and optimises the homography through it.

## Theiner & Ewerth — TVCalib: Camera Calibration for Sports Field Registration in Soccer (WACV 2023)
- Segment-point detection + differentiable optimisation of full camera parameters; MIT code (https://github.com/MM4SPA/tvcalib); used by the SoccerNet Game State baseline.

## Magera et al. — Can Geometry Save Central Views for Sports Field Registration? (arXiv 2504.20052, 2025)
- Turns circle correspondences into points and lines, so central close-ups (only the centre circle and halfway line visible) can be registered.

## Hayet, Piater, Verly — Robust incremental rectification of sports video sequences (BMVC 2004)
## Lu, Chen, Little — Pan-tilt-zoom SLAM for sports videos (BMVC 2019)
- Track the camera between frames from the static scene and re-anchor on field features; a PTZ model constrains drift.
- Relevance: our camera pans continuously; tracking between detections should cut per-frame jitter, which is what corrupts speed estimates.
