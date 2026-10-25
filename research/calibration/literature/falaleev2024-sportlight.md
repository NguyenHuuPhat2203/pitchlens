# Enhancing Soccer Camera Calibration Through Keypoint Exploitation (Sportlight)
- Authors: Nikolay S. Falaleev, Ruilong Chen
- Venue: ACM MMSports 2024 (arXiv 2410.07401); winner of SoccerNet Camera Calibration Challenge 2023
- URL: https://arxiv.org/abs/2410.07401 · code https://github.com/NikolasEnt/soccernet-calibration-sportlight (no licence)

## Key findings
- Multiplies usable correspondences with line-line and line-conic intersections, points on conics, and other pitch geometry; data fitting cleans imperfect annotations.
- Deep keypoint + line detection, real pitch dimensions as constraints, and a voter algorithm that iteratively keeps the most reliable keypoints.
- 2023 challenge: combined 0.55, JaC@5 73.22 %, completeness 75.59 % (Spiideo 0.53 with 99.96 % completeness; baseline 0.08).

## Relevance
Confirms that many correspondences + robust selection is what wins. The accuracy/completeness trade-off is explicit in the leaderboard: a method can refuse hard frames to stay accurate, which our pipeline must handle (frames without calibration).
