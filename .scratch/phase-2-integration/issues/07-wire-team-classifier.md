# 07 — Wire Team Classification into Pipeline

**What to build:** After running on a football video, every tracked player displays
their team label ("A", "B", or "referee") next to their track ID. The classifier
auto-fits on the first 30 frames (collects torso crops from player bboxes, runs
k-means), then predicts per-frame. No manual labelling needed.

**Blocked by:** 06 — Fix Import Ordering

**Status:** ready-for-agent

- [ ] Pipeline collects player crops from the first N frames and calls `TeamClassifier.fit()`
- [ ] After fitting, each frame's `PlayerTrack` gets a team label via `TeamClassifier.predict()`
- [ ] `Renderer.render_tracks` displays team label next to track ID (e.g. "#3 A")
- [ ] `ProcessedFrame.players` includes team label in serialized JSON output
- [ ] `config/default.yaml` gains a `team_classifier` section with `fit_frames` (default 30)
- [ ] Tests: pipeline integration test verifies team labels appear on output
