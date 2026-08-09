# 06 — Fix Import Ordering (Wide Refactor)

**What to build:** Every source file under `src/` conforms to PEP 8 import ordering:
stdlib imports → blank line → third-party imports → blank line → local imports.
No functional changes — this is a mechanical formatting pass so later integration
tickets don't create messy diffs.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `src/controller/pipeline.py`: blank lines between stdlib / third-party / local groups
- [ ] `src/service/detector.py`: blank line between third-party and local
- [ ] `src/service/tracker.py`: blank line between third-party and local
- [ ] `src/service/ball_detector.py`: blank line between third-party and local
- [ ] `src/service/ball_tracker.py`: blank line between third-party and local
- [ ] `src/service/shot_detector.py`: blank line between stdlib and local
- [ ] `src/service/speed_estimator.py`: blank line between stdlib and third-party
- [ ] `src/service/team_classifier.py`: blank line between stdlib and third-party
- [ ] `src/repository/output_writer.py`: blank lines between all three groups
- [ ] `src/repository/video_reader.py`: blank lines between all three groups
- [ ] `src/view/serializer.py`: blank line between stdlib and local
- [ ] `src/__main__.py`: blank line between stdlib and third-party
- [ ] All 40 existing tests still pass
