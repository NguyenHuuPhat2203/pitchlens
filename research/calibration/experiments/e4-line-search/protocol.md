# E4: training-free initialisation by line-family search (H5)

## Hypothesis
- **H5**: On the end-on high tactical view, a single frame can be registered with no trained model: straight painted lines grouped into two vanishing-point families, ordered, assigned to model lines (every ordered pair x every ordered pair), scored by chamfer precision, refined by line alignment, accepted on the precision-recall F-score.
  - sample.mp4 **test** frames 1320-1504, every 5th (37 frames, includes GT frames 1330, 1400, 1480): Acc@1m >= 90 % (refusals count as misses) and wrong answers (error > 2 m) <= 5 % of frames.
- Prediction for side views (TS-WorldCup test every 5th frame, WC14 test every 2nd): Acc@1m low and **wrong-answer rate above 10 %** - dev showed wrong answers up to F 0.72 on views with one marking family, so the search must not be trusted alone there.
- Mechanism: the end-on view shows both marking families (goal line, box and goal-area fronts, halfway line; touchlines and box/goal-area sides), so the four-line correspondence is determined; side views often show only one family.

## Dev / test split (fixed before any test run)
- Dev (used to build and tune): TS-WorldCup train, WC14 train_val, sample.mp4 frames 0-239 (dense GT) and the broadcast-graphics frames 300, 600, 900, 1200 (no GT, visual).
- Test: sample.mp4 1320-1504; TS-WorldCup test; WC14 test.

## Method (fixed, `tools/linesearch.py` + `tools/linefit.py` at this commit)
- Line-centre mask: top-hat on low-saturation pixels inside the grass region (filled from the border), orientation coherence > 0.6 (separates the goal net from the lines), elongated components with mean thickness <= 12 px, 1-D non-maximum suppression.
- Hough + band refit, families by vanishing point (max 5 lines each), all ordered model assignments, plausibility (in front, grass within 15 m of the pitch, real-camera handedness), coarse chamfer on 2 m samples, top 30 refined, ranked by F = 2PR/(P+R).
- Acceptance: F >= 0.55 (dev: keeps 11/11 correct sample answers; keeps 11/22 wrong side-view answers), >= 80 matched samples.
- Errors are the better of the two 180-degree-symmetric labellings (end-on views are ambiguous between the two goals; SoccerNet's rule).

## Decision rule
H5 supported if both sample bullets hold. A side-view wrong-answer rate above 10 % confirms the predicted limitation; it does not refute H5.
