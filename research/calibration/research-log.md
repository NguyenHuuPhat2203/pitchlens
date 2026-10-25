# Research log: automatic pitch calibration

## 2026-10-25 - Bootstrap
- Question from the user: detect the pitch instead of clicking 4 corners.
- Code today: `Calibrator.fit` on 4 clicked corners at frame 0 (display only), or a SynLoc camera file; one homography for the whole video.
- `data/sample.mp4` is a high end-on view from behind a goal: no pitch corner is visible, so the 4-corner UI cannot calibrate it at all. The camera also pans.
- Literature (literature/survey.md): keypoint and line methods dominate; TS-WorldCup gives temporal GT in metres; PnLCalib (GPL-2.0) is the strongest public model; Roboflow's YOLOv8-pose 32-kp model matches our stack.
- Data: WC14 (186 test), TS-WorldCup (887 test / 2,925 train), HF Roboflow 32-kp set (255/34/28), SoccerNet-Calibration-2023 downloading (train 2.6 GB, valid/test 0.5 GB each).
- GT conventions checked by overlay: WC14 and TS-WorldCup homographies map pixels to a 115 x 74 yd template (TS-WorldCup matrices carry a negative scale; sign-safe projection needed).
- No `/loop` or cron tool in this harness: continuity comes from these files.

## 2026-10-25 - E1 results (off-the-shelf), then paused at the user's request
- PnLCalib SV zero-shot: TS-WorldCup test Acc@1m 99.9 %, mean 0.34 m / median 0.30 m, displacement 0.23 m; WC14 Acc@1m 75.8 %, mean 0.86 m; 383 ms/frame. But **no homography on any of the 11 sample.mp4 frames** (end-on high tactical view).
- Roboflow YOLOv8x-pose: TS-WorldCup Acc@1m 0 %, median 8.7 m; wrong homography on every sample frame; 45 ms/frame.
- H1 refuted on the target domain (0/8 sample frames), H2 supported. The real problem is the end-on tactical viewpoint, not accuracy on side views.
- SoccerNet fitter checked on 300 valid images: 68 % fitted, RMS median 1.07 px; overlays align, including a behind-the-goal view.
- Paused: SoccerNet dataset build (`e2-yolo-pose/code/build_dataset.py`) stopped mid-run; E2 protocol written but not committed; nothing trained yet.
- Resume: rerun build_dataset.py (6 workers), commit E2 protocol, train, evaluate; check how many SoccerNet images are high behind-the-goal views.

## 2026-10-26 - Ground truth, E3 tracking, E4 line search
- Dense GT for sample.mp4 (gt-sample/): 425 frames fitted from snapped line annotations, RMS median 2.15 px. Dev frames 0-239, test frames 1320-1504.
- E3 (tracking from a GT start): lines-only alignment is best on the clip (Acc@1m 100 %, median 0.22 m, 12-frame displacement 0.012 m); ORB + lines 0.064 m; ORB alone drifts (32.9 %). TS-WorldCup 18.9-51.7 %: frames sub-sampled to ~5 fps, partial side views. H4 refuted on TS-WC only.
- E4 (training-free initial registration): 31/37 test frames answered, 0 wrong, median 0.23 m; WC14 13/93 answered. 6-12 s per frame. H5 refuted narrowly by refusals.
- E2: SoccerNet 32-kp dataset built (10,810 / 2,079); 7.45 % of train images are end-on. First training run died: Windows paging file exhausted by 4 dataloader workers.

## 2026-10-26 - E5: shipped configuration, product integration
- `src/serve/pipeline/auto_calibrator.py`: search at full resolution, tracking at 640 px with an analytic Jacobian, re-search on loss with backoff. Pipeline: calibration on its own thread each frame, speeds from pitch positions fixed when observed; 4-click UI, `Calibrator` and `SpeedEstimator` removed.
- E5 (protocol eeb9830): 100 % frames, Acc@1m 100 %, median 0.268 m, 12-frame displacement 0.039 m, track 37 ms (CPU shared with training).
- A rendered-pitch unit test exposed two bugs: thickness filter rejecting connected outlines (fixed), `_front` a no-op (fixed). An across-line NMS fix lost frame 0 (support under `min_support`) and was reverted. Final run: median 0.235 m, max 0.356 m, displacement 0.029 m, one 6.5 s search.
- E2 restarted at the user's request on the GPU: batch 16 needed 5.2 GB on the 4 GB card (spilled to shared memory, ~10 s/iteration), now batch 8 with 2 workers: 3.0 GB, ~9 min/epoch.
- User: no more local training. Local E2 run stopped at epoch 2 (deleted). E2 now runs on Colab: `notebooks/training/04_train_pitch_keypoints_sn32.ipynb`, protocol settings unchanged (batch 16 fits a T4); data and checkpoints through a private HF dataset repo (`scripts/hf_sync.py`, pattern from Research/3dgs-nvs); checkpoints pushed every 5 epochs so a dead session resumes from `last.pt`. `code/train.py` removed (the notebook replaces it).
