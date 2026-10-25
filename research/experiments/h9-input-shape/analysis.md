# H9 analysis — run_009 (variants), run_010 (end to end) (CONFIRMATORY)

## Provenance
`yolo26s.pt` (Ultralytics 8.3.222, 2026-01-05) re-exported at 960×960 with the original export arguments matches `models/player_detector.onnx` on 20 frames: same detection count on every frame, max |Δ box/conf| 6.1e-5. Other input sizes are therefore the same model, not different weights.

## Variants (TensorRT FP16, GPU letterbox, OC-SORT; SportsMOT `v_gQNyhv8y0QY_c013`)
| Input (h×w) | Scale | TRT forward | IDF1 | MOTA | ID sw | FP | FN | `track` ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 960×960 (current) | 0.75 | 6.59 | 62.13 | 60.39 | 107 | 3355 | 720 | 15.03 |
| 544×960 | 0.75 | 4.85 | 63.03 | 65.99 | 125 | 2655 | 810 | 13.66 |
| 768×768 | 0.60 | 4.56 | 56.21 | 64.75 | 140 | 2636 | 945 | 12.88 |
| 448×768 | 0.60 | 3.84 | 58.89 | 69.79 | 138 | 1994 | 1057 | 12.40 |
| 384×640 | 0.50 | 3.67 | 52.49 | 70.13 | 162 | 1636 | 1355 | 12.23 |

Eligible (IDF1 ≥ 61.13, MOTA ≥ 59.39, ID switches ≤ 128): **544×960 only**.

## End to end, 960×960 (A) vs 544×960 (B)
| Run | Variant | ms/frame | `player_track` |
|---|---|---:|---:|
| 1 | A | 32.47 | 26.56 |
| 2 | B | 31.71 | 24.52 |
| 3 | A | 32.22 | 27.11 |
| 4 | B | 30.21 | 25.00 |

Mean A 32.34 → B 30.96: −1.38 ms. On sample.mp4, B finds 94.5% of A's player boxes (mean IoU 0.972). Ball and shots are identical (the ball model is unchanged).

## Verdicts
- **H9 (768×768, ≥ 3 ms with IDF1 drop ≤ 1): refuted.** IDF1 −5.9, ID switches +31%.
- **H9b (544×960: accuracy within ±0.5, forward ≥ 30% faster): refuted on both predictions.** MOTA +5.6, FP −21%, FN +12.5%, forward only −26%. The variant passes the identity guard, but its end-to-end gain (1.38 ms) is below the 1.5 ms shipping threshold → **not shipped**.

## Mechanism (EXPLORATORY: detections vs ground truth by box-bottom y)
| y band | 960×960 FP | 544×960 FP | 960×960 FN | 544×960 FN |
|---|---:|---:|---:|---:|
| 0–100 | 635 | 438 | 34 | 35 |
| 100–250 | 1842 | 1356 | 51 | 60 |
| 250–450 | 997 | 933 | 115 | 154 |
| 450–720 | 132 | 69 | 9 | 11 |

At the same 0.75 scale, the square input surrounds the frame with 210 px of grey above and below; the rectangular one puts the frame edges at the tensor border. The detector's output depends on that context, not only on object scale. Most of the FP drop is people in the upper image (stands, sideline staff), but true mid-field players are also missed more often. Downscaling (0.6, 0.5) mostly adds misses (FN +31–88%), which fragments tracks: IDF1 falls 3–10 points even as MOTA rises from fewer crowd FPs.

## Lessons
- Changing the input shape changes accuracy even at equal scale. IDF1 on ground truth, not MOTA, is the deciding metric here: MOTA rewards dropping unannotated crowd detections.
- Noise floor: H7b's ±1-grey-level input change alone moved this sequence from IDF1 61.27 / 100 switches (run_006) to 62.13 / 107. Identity metrics on one 875-frame sequence move by ~1 point and ~7% switches from tiny perturbations.
- The bigger padding waste is the ball model (1280×1280 for a 1280×720 frame, 44% grey). Re-exporting it needs the training weights, which are not in the repo.
