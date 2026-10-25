# Findings: automatic pitch calibration

## Research question
Can the pipeline find the image -> pitch homography on every frame by itself, accurately and stably enough for positions and speeds, instead of 4 manual corner clicks?

## Current understanding
- **Yes, for the project's end-on view, without any trained model.** The shipped `AutoCalibrator` (line-family search to start, line alignment at 640 px to track) registers all 1,505 frames of sample.mp4 from a cold start: Acc@1m 100 % on 425 GT frames, median 0.235 m, max 0.356 m, 12-frame displacement median 0.029 m (E5).
- The 4-corner UI could never calibrate sample.mp4 (no corner in view; the camera pans), so speeds were impossible before.
- Public learned models do not transfer to this view: PnLCalib (SoccerNet SOTA, GPL) registers 0/11 sample frames; Roboflow's 32-kp YOLOv8x is wrong on every frame (E1). PnLCalib is excellent on side views (TS-WorldCup Acc@1m 99.9 %).
- Side-on broadcast views remain the gap: the line search answers 13/93 WC14 frames (mostly one line family visible). A learned keypoint model trained on SoccerNet (E2: dataset built, training moved to Colab) is the route for them.

## Patterns and insights
- Refusing beats guessing: an F-score gate (precision x recall of the markings, tau 0.55, >= 80 matched samples) gave 0 wrong answers on 37 test frames (E4); a wrong homography would corrupt every speed until the next loss.
- Tracking by the lines alone is the most stable variant (12-frame displacement 1-3 cm); ORB camera motion adds drift on this footage (E3).
- Speed must use each frame's own homography for each sample: projecting old pixels with today's homography turns a pan into running speed.
- Calibration CPU work must leave the pipeline's process: on a thread its GIL-holding Python (the least-squares loop) slowed tracking and ball detection too (+26 ms/frame); in a child process, started when the frame is decoded, it costs +9.6 ms/frame and is never waited for (E5).

## Lessons and constraints
- Homographies are defined up to scale including sign: test "in front of the camera" against a known-visible pitch point, not `w > 0` (TS-WorldCup GT has negative scale).
- WC14 / TS-WorldCup GT maps pixels to a 115 x 74 yd template; metres = yards x 0.9144.
- Keep the ranking metric honest: KpSFR's projection-error script drops points whose prediction falls off the pitch; ours keeps them, and Acc@1m counts failures as misses.
- Bounding-box "thickness" (area / extent) rejects connected marking outlines; use 2 x area / boundary pixels (E5).
- Thinner line masks lower Hough support: retune `min_support` together with any change to the line-centre NMS (E5 run 2).
- Loosening the solver tolerances halves tracking stability (12-frame displacement 0.030 -> 0.07 m on dev frames): not a latency lever.
- Synthetic test views need realistic grass extent: the plausibility check rejects grass more than 15 m off the pitch.

## Open questions
- Does a SoccerNet-trained keypoint model (E2) handle side views at product latency, and the end-on view at all?
- Remaining calibration cost +9.6 ms/frame (43.0 -> 52.6 on the RTX 3050 Ti laptop): a cheaper line mask or a single fine pass while tracking are the candidates.
