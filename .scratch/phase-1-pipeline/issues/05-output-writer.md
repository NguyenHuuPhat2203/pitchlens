# 05 — Output Writer (Video + JSON)

**What to build:** After processing, the pipeline saves two outputs: an annotated `.mp4` video and a `.json` file containing per-frame data in the agreed envelope format. The JSON wraps every response in `{"success": true, "data": {...}, "meta": {...}}` per RULES.md.

**Blocked by:** 04 — ThreadPool parallelism

**Status:** ready-for-agent

- [ ] `src/model/frame.py`: `ProcessedFrame` dataclass (frame_number, timestamp, players: list[PlayerTrack], ball: BallTrack | None, events: list[ShotEvent])
- [ ] `src/repository/output_writer.py`: `OutputWriter` class. Method `write_video(frames: Iterator[np.ndarray], output_path, fps)` using `cv2.VideoWriter`. Method `write_json(frames: list[ProcessedFrame], output_path)` wrapping in success envelope.
- [ ] `src/view/serializer.py`: `Serializer` class. Method `serialize(frame: ProcessedFrame) -> dict` produces the JSON-serializable dict with envelope.
- [ ] `src/controller/pipeline.py`: After all frames processed, call `OutputWriter` to save video and JSON.
- [ ] `tests/test_output_writer.py`: verifies output `.mp4` is playable with correct frame count, `.json` is valid and parseable.
