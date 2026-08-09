# Football Tracker

Real-time football match tracking with computer vision. Detects players
and the ball, tracks movement with persistent identities, classifies teams
by jersey color, estimates speed, and detects shot events.

**Demo:** paste a YouTube URL → watch annotated stream frame-by-frame.

Built with [Ultralytics YOLO26](https://docs.ultralytics.com/),
[Supervision](https://supervision.roboflow.com/), and FastAPI.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run on a local video
python main.py
```

1. Click 4 pitch corners on the first frame (or press Q to skip calibration)
2. Pipeline runs — annotated frames appear in an OpenCV window
3. Press Q to quit

## Features

| Feature | Status |
|---------|--------|
| Player detection + tracking (OC-SORT) | Done |
| Team classification (HSV k-means) | Done |
| Ball detection + Kalman tracking | Done |
| Shot event detection | Done |
| Pitch calibration (4-click, desktop only) | Done |
| Direction arrows (always on) | Done |
| Speed overlay (km/h, calibration only) | Done |
| Annotated video output (.mp4) | Done |
| JSON output (item schema) | Done |
| FastAPI backend + WebSocket streaming | Done |
| Browser demo (paste YouTube URL) | Done |

## Architecture

```
YouTube URL → download → Pipeline → WebSocket JPEG stream → browser
                                  ↓
                            JSON result array
```

Pipeline tiers (hybrid layout):

```
src/
├── backbone/       YOLO wrappers (detector, ball_detector)
├── features/       Team classifier (HSV k-means)
├── model/          Data classes (Detection, Track, Frame, Event)
├── serve/
│   ├── pipeline/   Inference engine (tracker, ball_tracker,
│   │               calibrator, speed, shot, renderer, video_reader)
│   ├── artifacts/  Serializer (item schema), OutputWriter
│   └── adapt/      FastAPI + WebSocket (app, schemas)
├── training/       Train-only (datasets, conversion)
├── config.py       Pydantic config models
└── logger.py       Structured JSON logging
```

## Usage

### Desktop (local video)

```bash
python main.py
```

Place a video at `data/sample.mp4` or edit `config/default.yaml`.

### Web API

```bash
uvicorn src.serve.adapt.app:create_app --factory --port 8000
```

Then open `demo/index.html`, paste a YouTube URL, and click Analyze.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/analyze` | Accept `{"url": "..."}`, returns `session_id` |
| WS | `/v1/stream/{id}` | Binary JPEG frames, then final JSON array |

### Training

Open `notebooks/football_tracker_training.ipynb` in Jupyter or Colab.
Covers dataset download, MOT→YOLO conversion, player fine-tuning,
ball detector training, and pipeline evaluation.

## Configuration

`config/default.yaml` is a YAML file validated against a Pydantic model
at startup. Key sections:

```yaml
model:
  player_detector: "yolo26s.pt"
  device: "auto"                  # CUDA if available, else CPU

pipeline:
  display: true                   # Show OpenCV window
  output_video: null              # Save annotated .mp4
  output_json: null               # Save per-frame JSON

team_classifier:
  enabled: true                   # Auto-fit on first 30 frames

ball_detector:
  enabled: true

shot_detector:
  angle_threshold: 30.0
  accel_threshold: 50.0

calibration:
  enabled: true                   # Click-to-calibrate
  speed_window: 5                 # Frames for speed smoothing
```

## Output Format

Item schema (array of per-frame items):

```json
[
  {
    "status": "success",
    "result": {
      "frame_number": 0,
      "players": [
        {
          "track_id": 3,
          "bbox": [120.0, 80.0, 180.0, 240.0],
          "team": "A",
          "speed_kmh": 22.5,
          "direction_deg": 45.0
        }
      ],
      "ball": { "centroid": [320.0, 200.0], "state": "in_play" },
      "events": []
    }
  }
]
```

## Development

```bash
# Run tests
pytest tests/ -v          # 40 tests

# Lint
ruff check src/ tests/ main.py

# Pre-commit
pre-commit install
```

## Docs

- [Glossary](CONTEXT.md)
- [ADR](docs/adr/0001-pipeline-architecture-yolo26s-ocsort.md)
- [Research](docs/research-real-time-football-tracking.md)
- [Implementation plan](docs/implementation-plan.md)
