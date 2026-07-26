# 01 — Scaffold + Video Playback

**What to build:** A runnable project skeleton that reads a football video file and displays it frame-by-frame in an OpenCV window. Press Q to quit. This is the thinnest possible tracer bullet through the entire pipeline — no detection, no tracking, just I/O → display.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `requirements.txt` with `ultralytics`, `supervision`, `opencv-python`, `numpy`
- [ ] `src/` package with `__init__.py`
- [ ] `config/default.yaml` with model path, confidence threshold, video source placeholder
- [ ] `src/model/frame.py`: `Frame` dataclass (index, timestamp, image array)
- [ ] `src/repository/video_reader.py`: `VideoReader` — iterator that opens a video path and yields `Frame` objects one at a time. Must not load all frames into memory.
- [ ] `src/controller/pipeline.py`: `Pipeline.run(source)` — reads frames from `VideoReader`, passes each through an empty `process_frame` method, displays with `cv2.imshow`. Quits on `q` key.
- [ ] `src/__main__.py`: entry point — loads config, creates Pipeline, calls run.
- [ ] `tests/test_video_reader.py`: verifies `VideoReader` yields frames with correct index and non-None image arrays from a short test video.
