# 02 — Detection (YOLO26s)

**What to build:** Extend the pipeline so every frame runs through a YOLO26s detector. Bounding boxes for detected persons appear on the displayed video. The detector wraps `ultralytics.YOLO` and outputs `Detection` objects.

**Blocked by:** 01 — Scaffold + video playback

**Status:** ready-for-agent

- [ ] `src/model/detection.py`: `Detection` dataclass (bbox x1/y1/x2/y2, class_id, confidence)
- [ ] `src/service/detector.py`: `Detector` class wrapping `ultralytics.YOLO("yolo26s.pt")`. Method `detect(frame) -> list[Detection]`. Filters to `person` class (class_id=0) only.
- [ ] `src/controller/pipeline.py`: `process_frame` now calls `detector.detect()` and passes detections to renderer.
- [ ] `src/view/renderer.py`: `Renderer` class using `supervision.BoxAnnotator`. Method `render(frame, detections) -> np.ndarray` returns annotated frame.
- [ ] `tests/test_detector.py`: verifies `Detector.detect` returns `Detection` objects with valid bbox coordinates on a known test image.
