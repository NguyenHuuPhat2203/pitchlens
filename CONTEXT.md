# Football Tracker

Real-time football match tracking system using computer vision. Detects players, ball, referee;
tracks movement with identity preservation through short occlusions; classifies teams;
computes real-world speed vectors via pitch homography; and detects shot events from
ball acceleration.

## Language

### Core Entities

**Player**:
An outfield participant belonging to one of two teams (Team A or Team B). Detected as a
person-class bounding box in the frame. Has a track ID, team label, role label, and
optional speed vector.
_Avoid_: Subject, person, athlete

**Goalkeeper**:
A player with role `goalkeeper`, distinguished from outfield players by position
proximity to goal area and/or different jersey color. Detectable as a separate class
from `player`.
_Avoid_: Keeper, goalie, GK

**Referee**:
An on-field official (center referee only; linesmen excluded for phase 1). Role is
`referee`. Distinct from players by jersey color (typically black/yellow).
_Avoid_: Umpire, official, linesman

**Ball**:
The spherical object in play. Detected via dedicated small-object detector. Has
position, trajectory, speed, and acceleration. Speed and acceleration changes are
used to detect shot events.
_Avoid_: Football, soccer ball

**Team**:
A group of players wearing the same jersey color. Labeled as Team A or Team B —
the actual club names are not inferred from video. Assignment is via HSV clustering
without manual labeling.
_Avoid_: Side, squad, club

### Tracking Concepts

**Detection**:
A single-frame observation: bounding box + class label + confidence score. Raw output
from the YOLO model before tracking is applied.

**Track**:
A sequence of detections across consecutive frames assigned the same integer ID.
Tracking preserves identity through short occlusions (a few seconds). If a player
leaves the frame and returns after >5 seconds, re-identification is not guaranteed.
_Avoid_: Trace, trajectory, path

**ID Switch**:
When the tracker incorrectly reassigns an existing track ID to a different real-world
person, or starts a new ID for a person who already has one. A key failure mode of
the tracking stage.

**Occlusion**:
A period during which a tracked object is partially or fully hidden from the camera.
The tracker should bridge short occlusions (< track_buffer frames) via Kalman
prediction.

### Spatial Concepts

**Bounding Box (bbox)**:
A rectangle defined by `(x1, y1, x2, y2)` in pixel coordinates, enclosing a detected
object.

**Speed Vector**:
A 2D vector (magnitude in km/h, direction in degrees) representing the real-world
velocity of a player or the ball. Computed from frame-to-frame displacement in pitch
coordinates via homography.
_Avoid_: Velocity, motion vector

**Pitch Coordinates**:
Real-world 2D coordinates (meters) on the football field, obtained by applying a
homography matrix to pixel coordinates. Origin and axes follow the pitch layout.
_Avoid_: World coordinates, field position

**Homography**:
A 3×3 perspective transformation matrix mapping camera pixel space to pitch
coordinates. Must be recomputed periodically for broadcast footage because the
camera pans, tilts, and zooms.
_Avoid_: Calibration matrix, perspective transform

### Events

**Shot**:
An event detected when two conditions hold simultaneously:
1. The ball's velocity vector changes direction (angle delta exceeds threshold), AND
2. The ball's acceleration vector aligns with the velocity vector AND its magnitude
   exceeds a threshold (indicating an intentional strike).
_Avoid_: Kick, strike, hit

**Pass** (future):
A controlled ball transfer from one player to a teammate. Not implemented in Phase 1.

### Pipeline Stages

**Pipeline**:
The end-to-end processing chain: Input → Detect → Track → Classify Teams →
Calibrate Pitch → Compute Speed → Detect Events → Render → Output.
_Avoid_: Workflow, chain, process

**Tier**:
An architectural layer. Backbone (YOLO wrappers), Features (transforms),
Serve (inference pipeline, artifacts, HTTP adapters), Training (dataset prep).
Tiers follow Clean Architecture: inner tiers never import outer ones.
_Avoid_: Layer, module, package

**Frame**:
A single image extracted from the video source. Input granularity to the pipeline.

**Phase**:
A development milestone grouping related pipeline stages. Phase 1 = Detection +
Tracking. Phase 2 = Team classification. Phase 3 = Ball tracking. Phase 4 = Speed
vectors. Phase 5 = Shot detection.
_Avoid_: Milestone, sprint, iteration
