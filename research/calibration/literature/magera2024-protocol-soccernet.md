# A Universal Protocol to Benchmark Camera Calibration for Sports (SoccerNet-Calibration)
- Authors: Floriane Magera, Thomas Hoyoux, Olivier Barnich, Marc Van Droogenbroeck
- Venue: CVPR Workshops (CVsports) 2024; dataset and code https://github.com/SoccerNet/sn-calibration
- Related: Cioppa et al., "SoccerNet 2023 Challenges Results", arXiv 2309.06006

## Key findings
- SoccerNet-Calibration annotates the extremities of 26 pitch-marking classes (and points on circles) on broadcast images; no camera ground truth.
- JaC@t: a predicted marking is a true positive only if every annotated point of it lies within t pixels of its reprojection; JaC@5 × completeness ranks methods. The left/right ambiguity of goal views is resolved by taking the better of the two mirrored labelings.
- Download without NDA: `SoccerNetDownloader(...).downloadDataTask(task="calibration-2023", split=[...])`.

## Relevance
Largest and most diverse camera-view dataset; a second, independent check if TS-WorldCup results need confirmation on more camera types.
