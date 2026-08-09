# SoccerNet 2026 Challenges — Integration Research

> **Date:** 2025-08-09
>
> **Source:** [SoccerNet 2026 Challenges](https://www.soccer-net.org/challenges/2026)

## Summary

4 of 6 challenges are applicable to this project. Each evaluated for data access, difficulty, and integration effort.

---

## 1. Spiideo SynLoc — Auto Pitch Calibration

**Task:** Single-frame world-coordinate athlete detection & localization. Detect all players and predict their pitch positions in meters directly from a broadcast image.

**Why relevant:** Replaces manual 4-click calibration with a model that directly outputs pitch coordinates. The biggest missing piece in the current pipeline.

**Data:**
- Download via `SoccerNet` pip package: `task="SpiideoSynLoc"`
- 4K images + camera model, `pip install sskit`
- Synthetic + real broadcast data, precise calibration
- Source: [github.com/Spiideo/sskit](https://github.com/Spiideo/sskit)
- Data portal: [research.spiideo.com](https://research.spiideo.com/)

**Dev kit:** `pip install sskit` — camera model (`sskit.camera`), COCO eval (`sskit.coco`), metric mAP-LocSim

**Integration:**
- Train/fine-tune a detector that outputs `position_on_pitch` (x, y, z meters)
- Replace `Calibrator.fit()` manual click with model inference
- Bonus: `sskit.camera` provides proper camera projection for all frames

**Difficulty:** Medium — needs model training, but data is clean and dev kit is complete.

---

## 2. Player-Centric Ball Action Spotting (FOOTPASS)

**Task:** Identify **who** performs **what** action and **when** — `(frame, team, jersey, class)`.

**Why relevant:** Current shot detector only knows *that* a shot happened. FOOTPASS adds: which player, which team, and 8 action classes (Drive, Pass, Cross, Shot, Header, Throw-in, Tackle, Block) + 13 tactical roles.

**Data:**
- 54 complete matches, 102K events
- Full HD broadcast videos + tracking + team/jersey/role
- Source: [github.com/JeremieOchin/FOOTPASS](https://github.com/JeremieOchin/FOOTPASS)
- Dataset: [HuggingFace SN-PCBAS-2026](https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026/tree/main)
- **Videos require NDA** (SoccerNet sign-up)

**Dev kit:** 3 baselines — TAAD (visual), TAAD+GNN (graph), TAAD+DST (sequence denoising)

**Integration:**
- Replace `ShotDetector` with action classifier outputting 8 classes + player assignment
- Leverage existing tracker output (track_id, team) for player-centric prediction
- Play-by-play output format: `(frame, team, jersey, class, confidence)`

**Difficulty:** High — needs NDA, full training pipeline, but directly extends current architecture.

---

## 3. Ball Action Anticipation (FAANTRA)

**Task:** Anticipate ball-related actions **5 seconds before** they happen. 10 action classes.

**Why relevant:** Extends event detection into predictive analytics. Could show "anticipated pass/shot" on the stream.

**Data:**
- SoccerNet Ball Action Anticipation (BAA) dataset
- Source: [HuggingFace ActionAnticipation](https://huggingface.co/datasets/SoccerNet/ActionAnticipation/tree/main)
- **Videos require NDA password**
- Can also use existing Ball Action Spotting (BAS) dataset

**Dev kit:** FAANTRA (FUTR-based transformer), config-driven training, mAP@δ metric

**Integration:**
- Feed tracking + ball trajectory into anticipation model
- Output predicted actions with time-to-event
- Can run in parallel with current pipeline

**Difficulty:** Medium-High — NDA required, model is complex, but fits as a parallel module.

---

## 4. FIFA Skeletal Tracking Light

**Task:** Generate accurate 3D player pose (15 keypoints) from single broadcast camera.

**Why relevant:** Adds skeleton visualization to player tracking. YOLO-pose (17 keypoints) is already available in ultralytics — this challenge provides 3D elevation.

**Data:**
- Source: [HuggingFace FIFA-Skeletal-Tracking-Light-2026](https://huggingface.co/datasets/tijiang13/FIFA-Skeletal-Tracking-Light-2026/tree/main)
- Preprocessed 2D/3D skeletons + bounding boxes + camera parameters
- **Open access** — no NDA needed (unlike other challenges)

**Dev kit:** Baseline with camera pose estimation, field marking refinement, 3D projection/optimization. Uses SAM-3D-body for preprocessing.

**Integration:**
- Add YOLO26s-pose (built into ultralytics) for 2D keypoints
- Use challenge's camera model for 3D lifting
- Render skeleton overlay on player bbox

**Difficulty:** Low-Medium — data is open, YOLO-pose is built-in, 3D lifting is well-documented.

---

## Recommendation

| Priority | Challenge | Reason |
|----------|-----------|--------|
| **1** | SynLoc | Auto calibration — removes the last manual step, highest impact |
| **2** | FIFA Skeletal | Open data, built-in YOLO-pose, easy win |
| **3** | FOOTPASS | Player-centric actions — natural evolution of shot detector |
| **4** | FAANTRA | Anticipation — cool but needs more infrastructure first |

## Data Access Summary

| Challenge | Account needed | NDA? | Data format |
|-----------|---------------|------|-------------|
| SynLoc | SoccerNet + Spiideo | No | 4K images, COCO JSON |
| FOOTPASS | HuggingFace + SoccerNet NDA | Yes | Videos + JSON play-by-play |
| FAANTRA | HuggingFace + SoccerNet NDA | Yes | Video clips, frames |
| FIFA Skeletal | HuggingFace | No | NPZ skeletons, camera params |

**Start with SynLoc + FIFA Skeletal** — both have the lowest access barrier and provide immediate pipeline improvements.
