# 03 — Tracking (OC-SORT)

**What to build:** Track detections across frames using OC-SORT. Each player gets a persistent integer ID displayed above their bounding box. A trail path shows their recent movement. The tracker is configured via a YAML file tuned for football.

**Blocked by:** 02 — Detection (YOLO26s)

**Status:** ready-for-agent

- [ ] `config/trackers/ocsort.yaml`: OC-SORT config tuned for football (lower `track_high_thresh`, longer `track_buffer` for occlusion tolerance)
- [ ] `src/model/track.py`: `PlayerTrack` dataclass (track_id, bbox, class_id, confidence)
- [ ] `src/service/tracker.py`: `Tracker` class wrapping `ultralytics` built-in track mode. Method `update(detections) -> list[PlayerTrack]`. Uses the ocsort config.
- [ ] `src/controller/pipeline.py`: `process_frame` now runs detector → tracker → passes `PlayerTrack` list to renderer.
- [ ] `src/view/renderer.py`: Extend `Renderer` to draw track IDs with `sv.LabelAnnotator` and trail paths with `sv.TraceAnnotator`.
- [ ] `tests/test_tracker.py`: verifies `Tracker.update` assigns consistent IDs to the same person across consecutive frames.
