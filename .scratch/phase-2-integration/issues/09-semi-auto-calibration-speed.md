# 09 — Semi-Auto Pitch Calibration + Speed Overlay

**What to build:** On the first frame, the user clicks 4 pitch landmarks (e.g.
corner flags, penalty marks). A homography is computed and used for the rest of
the video. Every player and the ball then shows real-world speed in km/h next to
their label (e.g. "#3 A · 22 km/h"). Direction is shown as a short arrow.

**Blocked by:** 07 — Wire Team Classification, 08 — Wire Ball and Shot

**Status:** ready-for-agent

- [ ] First frame: user clicks 4+ points in order (top-left corner flag, top-right,
  bottom-right, bottom-left of the pitch) via `cv2.setMouseCallback`
- [ ] `Calibrator.fit(image_points, pitch_points)` computes homography from clicks
- [ ] Each subsequent frame: `Calibrator.transform()` converts pixel centroids to
  pitch metres
- [ ] `SpeedEstimator.estimate()` computes km/h and direction from consecutive
  centroids in pitch space
- [ ] `Renderer` adds speed label (e.g. "22 km/h") and a direction arrow to each
  player and the ball
- [ ] `ProcessedFrame.players` includes `speed_kmh` and `direction_deg` fields
- [ ] `config/default.yaml` gains `calibration` section (pitch dimensions, click
  labels for UI guidance)
- [ ] Tests: verify homography produces consistent speed values on synthetic data
