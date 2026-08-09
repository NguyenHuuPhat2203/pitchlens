# 08 — Wire Ball Detection, Tracking, and Shot Detection

**What to build:** The football appears on screen as a small red circle with a
trail. When a shot is detected (sudden direction change + acceleration), a
"SHOT!" flash appears on screen. Ball state (in_play / lost) is displayed.

**Blocked by:** 06 — Fix Import Ordering

**Status:** ready-for-agent

- [ ] Pipeline calls `BallDetector.detect_ball()` each frame
- [ ] Pipeline feeds results into `BallTracker.update()` (Kalman filter handles misses)
- [ ] Pipeline feeds `BallTrack` into `ShotDetector.update()` each frame
- [ ] `Renderer` draws ball position as a distinct marker (small filled circle) + short trail
- [ ] `Renderer` flashes "SHOT! XX km/h" overlay for 1 second after a shot event
- [ ] `ProcessedFrame` includes `ball: BallTrack | None` and `events: list[ShotEvent]`
- [ ] `config/default.yaml` gains `ball_detector` and `shot_detector` sections
- [ ] Tests: pipeline integration test verifies ball and shot data in output JSON
