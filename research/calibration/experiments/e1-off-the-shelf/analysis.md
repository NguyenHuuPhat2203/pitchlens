# E1 analysis - off-the-shelf registration (CONFIRMATORY)

| Model (zero-shot) | TS-WC Acc@1m | TS-WC mean / median | TS-WC displacement | WC14 Acc@1m | WC14 mean | sample.mp4 (end-on) | Latency |
|---|---:|---:|---:|---:|---:|---|---:|
| PnLCalib SV (SoccerNet-trained) | 99.9 % | 0.34 / 0.30 m | 0.23 m | 75.8 % | 0.86 m | no homography on 11/11 frames | 383 ms |
| Roboflow YOLOv8x-pose (32 kp) | 0.0 % | 54 / 8.7 m | 82.7 m | 0.5 % | 19.4 m | homography on 11/11, all visibly wrong | 45 ms |

Overlays: `results/sample_*_sheet.jpg`. Per-frame rows: `results/rows_*.json`.

## Verdict
- **H1 refuted** on its own decision rule: TS-WC numbers hold by a wide margin (better than KpSFR fine-tuned, 0.36 m), but 0 of 8 clean sample frames register.
- **H2 supported**: 160x worse than H1 on TS-WC and wrong on every end-on frame.

## What it means
- Accuracy on side views is a solved problem with public models; the open problem for this product is the **viewpoint**. `sample.mp4` is a high tactical camera behind a goal. PnLCalib's refusal (no homography) is the safe failure; Roboflow's confident wrong homographies are the dangerous one, so any shipped method needs a sanity gate.
- Next (E2, protocol written, not yet committed or run): train our own YOLO26s-pose on SoccerNet-Calibration (many camera types, incl. behind-the-goal), labels from homographies fitted to the human line annotations. First check how many SoccerNet images resemble the end-on tactical view; if few, add views synthesised by re-warping the pitch plane.
- Diagnostic still open: whether PnLCalib finds too few keypoints on sample frames or rejects them geometrically (the check hit a CUDA OOM while the GPU was in use by another program).

## EXPLORATORY (after the locked run)
- Why PnLCalib refuses the sample: its strongest keypoint on sample frames scores 0.30 (frames 0 and 1400), below the 0.34 demo threshold; on a WC14 frame 8 keypoints score 0.54-0.78. The detector is unsure, the geometry never gets inputs.
- With its SoccerNet-evaluation thresholds (kp 0.0966, line 0.3441) it returns a homography on 8/8 clean sample frames, and all 8 are visibly wrong (`results/exploratory_pnlcalib_sn_thresholds_sheet.jpg`). Lower thresholds turn a safe refusal into confident garbage: the end-on tactical view is out of its training distribution.
