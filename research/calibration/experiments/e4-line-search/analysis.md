# E4 analysis - training-free line search (CONFIRMATORY on sample test; side views partial)

The run was stopped at the user's request after the sample test, all of WC14 test (every 2nd frame) and 33 of 178 TS-WorldCup test frames; per-frame errors from the log are in `results_partial.json`.

| Test set | Frames | Answered | Acc@1m | Wrong (> 2 m) | Median / max error |
|---|---:|---:|---:|---:|---:|
| sample.mp4 1320-1504 (every 5th) | 37 | 31 | 83.8 % | **0 %** | 0.23 / 0.89 m |
| WC14 test (every 2nd) | 93 | 13 | 8.6 % | 5.4 % | 0.51 / 48.5 m |
| TS-WorldCup test (every 5th, partial) | 33 | 5 | 6.1 % | 6.1 % | 1.04 / 11.2 m |

Latency: 6-12 s per frame (CPU), only needed when a shot starts or tracking is lost.

## Verdict
**H5 refuted, narrowly**: Acc@1m 83.8 % < 90 % because 6 of 37 frames were refused; the safety bullet holds (no wrong answer). Side views: the method mostly refuses (14 % and 15 % answered), and the wrong-answer rate (5-6 %) is lower than the predicted > 10 % - the F >= 0.55 gate plus the 80-sample minimum refuse most one-family views instead of guessing.

## What it means
- On the target view, a frame either gets a sub-metre homography or no answer. In a video that is enough: one accepted frame starts the tracker (E3: 100 % within 1 m over 425 frames).
- Side broadcast views need a learned detector (E2 was not run: the training process died on a Windows paging-file error while other jobs shared the RAM; deprioritised by the user's time budget).
