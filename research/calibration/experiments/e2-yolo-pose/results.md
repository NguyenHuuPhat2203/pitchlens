# E2 results: YOLO26s-pose on SoccerNet 32-kp labels (H3)

Training: Colab T4, `notebooks/training/04_train_pitch_keypoints_sn32.ipynb`, protocol settings (args.yaml checked). ~5.1 min/epoch. **Stopped at epoch 26/40 by patience 10**: Ultralytics fitness is box mAP50-95 + pose mAP50-95, and the box (one whole-view "pitch" box) jumps between 0.58 and 0.82, so epoch 16's box outlier (0.816) stayed best while pose mAP50-95 kept rising (0.835 at 16 -> 0.869 at 26). `best.pt` = epoch 16 (protocol selection, CONFIRMATORY); `last.pt` = epoch 26 (EXPLORATORY).

| | PnLCalib SV (E1) | **E2 best, epoch 16** | E2 last, epoch 26 (exploratory) |
|---|---|---|---|
| TS-WorldCup test Acc@1m | 99.9 % | **97.3 %** | 97.3 % |
| TS-WorldCup median / mean (m) | 0.304 / 0.342 | **0.448** / 0.505 | 0.395 / 0.452 |
| TS-WorldCup 1-step displacement median (m) | | 0.141 | 0.136 |
| WC14 test Acc@1m (median) | 75.8 % | 52.7 % (0.82 m) | 50.5 % (0.94 m) |
| SoccerNet test Acc@1m (median), 2,081 fitted images | | 75.2 % (0.41 m) | 76.2 % (0.37 m) |
| sample.mp4, protocol 8 end-on frames < 1 m | 0/8 (no answer) | **0/8** (6 wrong answers, 7.5-159 m) | 0/8 (8 wrong answers) |
| sample.mp4, 425 dense-GT frames < 1 m | | 0 % (363 answered, median 20-67 m) | 0 % (420 answered) |
| Latency median, RTX 3050 Ti, no TensorRT | 383 ms | **31.6 ms** (12x) | 30.8 ms |

`sample_gt.py` judges the protocol's sample criterion with the GT homographies (error < 1 m) instead of by eye; the overlays (`results/sample_yolo_*_sheet.jpg`) agree: projected markings visibly off the painted lines.

## H3: refuted
- TS-WC Acc@1m within 10 points of PnLCalib: yes (97.3 vs 99.9).
- TS-WC median <= 1.3x PnLCalib (0.395 m): **no** (0.448 m). The exploratory epoch-26 checkpoint lands exactly on the bar (0.395).
- >= 6/8 end-on sample frames: **no, 0/8**, and the model answers wrongly instead of refusing.
- >= 5x lower latency: yes (12x).

## What it means
- SoccerNet teaches side-on registration well at product latency (97 % within 1 m on TS-WorldCup, 31 ms).
- It does not teach the high end-on view (7.45 % of the training images), and its failures there are confident wrong homographies: the model must never be used without a verification gate. The protocol's fallback ("line-based refinement if it registers end-on frames but misses accuracy") does not apply: it does not register them at all.
- Training longer helps side views (epoch 26 vs 16: TS-WC median 0.448 -> 0.395 m, SoccerNet test 0.41 -> 0.37 m); a rerun should select by pose mAP and not stop on the box metric.
- Next (E6): line search first; when it refuses, the model's homography as a starting point -> line alignment -> the same F-score gate. Success = WC14 / TS-WC frames gained with no wrong answers on sample.mp4.

Raw: `results/summary_e2_{best,last}.json`, `results/rows_e2_{best,last}.json`, `results/sample_gt_{best,last}.json`, overlays `results/sample_yolo_*`.
