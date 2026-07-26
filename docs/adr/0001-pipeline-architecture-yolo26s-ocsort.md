# Pipeline Architecture for Real-Time Football Tracking on RTX 3050 Ti (4GB VRAM)

The system must run on a laptop-grade RTX 3050 Ti (4GB VRAM), processing broadcast
football footage at near-real-time. YOLO26s was chosen as the detector, OC-SORT as the
tracker, and HSV clustering for team classification — rejecting both the smaller
YOLO26n (insufficient accuracy for small ball detection) and the larger YOLO26m
(exceeds VRAM budget when combined with ball detector). The tracker choice rejects
ByteTrack (insufficient for non-linear player motion in football) and Deep OC-SORT
(adds ReID overhead without benefit, since full identity preservation is out of scope).

Considered options:

- **YOLO26n + ByteTrack**: Fastest, but YOLO26n's lower mAP (40.9 vs 48.6) hurts
  ball detection at broadcast resolution. ByteTrack's linear Kalman handles player
  direction changes poorly.
- **YOLO26m + Deep OC-SORT**: Best accuracy, but 20.4M params + ReID model exceeds
  4GB VRAM when running ball detection in parallel.
- **YOLO26s + OC-SORT (chosen)**: 9.5M params. OC-SORT's observation-centric Kalman
  corrects for abrupt turns and sprints. Room left for a small ball detector or
  pose head in later phases.

Consequences: team classification is limited to HSV clustering (no jersey number
recognition). Speed accuracy depends on pitch homography quality, which degrades
during fast camera pans. Shot detection uses a simplified acceleration-threshold
heuristic rather than trajectory analysis.
