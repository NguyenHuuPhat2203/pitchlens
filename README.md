# Football Tracker ⚽

Real-time football match tracking using computer vision. Detects players, the ball, and referees; tracks movement with persistent identities; classifies teams by jersey color; computes real-world speed via pitch homography; and detects shot events from ball acceleration.

Built with [Ultralytics YOLO26](https://docs.ultralytics.com/) + [Supervision](https://supervision.roboflow.com/).

## Demo

```
python -m src
```

1. Click 4 pitch corners on the first frame (top-left, top-right, bottom-right, bottom-left)
2. Press Q to skip calibration (speed won't be computed)
3. The pipeline runs — press Q to quit anytime

## Features

| Feature | Status |
|---------|--------|
| Player detection + tracking (OC-SORT) | ✅ |
| Team classification (HSV k-means) | ✅ |
| Ball detection + Kalman tracking | ✅ |
| Shot event detection (acceleration heuristic) | ✅ |
| Pitch calibration (semi-auto, 4 clicks) | ✅ |
| Real-world speed + direction arrows | ✅ |
| Annotated video output (.mp4) | ✅ |
| Per-frame JSON output (RULES.md envelope) | ✅ |

## Installation

```bash
# Create venv
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10, CUDA-capable GPU recommended (works on CPU, slower).

## Usage

### Quick start

1. Drop a football video in `data/sample.mp4` (or edit `config/default.yaml`)
2. Run the pipeline:

```bash
python -m src
```

### Configuration

Edit `config/default.yaml`:

```yaml
video:
  source: "data/sample.mp4"

model:
  player_detector: "yolo26s.pt"   # YOLO model
  detection_conf: 0.3
  device: 0                        # GPU device (0 = first GPU, "cpu")

pipeline:
  display: true                    # Show live preview
  output_video: "output/match.mp4" # Save annotated video (null = skip)
  output_json: "output/match.json" # Save per-frame data (null = skip)
  cpu_workers: 2

team_classifier:
  enabled: true
  fit_frames: 30                   # Frames to sample for k-means

ball_detector:
  enabled: true
  max_lost_frames: 15              # Frames before declaring ball lost

shot_detector:
  angle_threshold: 30.0            # Min velocity direction change (deg)
  accel_threshold: 50.0            # Min acceleration magnitude (px/s²)

calibration:
  enabled: true                    # Enable click-to-calibrate + speed
```

### Output format

Per-frame JSON follows the RULES.md envelope convention:

```json
{
  "success": true,
  "data": {
    "frames": [
      {
        "frame_number": 0,
        "timestamp": 0.0,
        "players": [
          {
            "track_id": 3,
            "bbox": [120.0, 80.0, 180.0, 240.0],
            "class_id": 0,
            "confidence": 0.92,
            "team": "A",
            "speed_kmh": 22.5,
            "direction_deg": 45.0
          }
        ],
        "ball": {
          "track_id": 0,
          "centroid": [320.0, 200.0],
          "velocity": [10.0, -2.0],
          "state": "in_play"
        },
        "events": []
      }
    ]
  },
  "meta": {
    "timestamp": "2025-07-26T12:00:00+00:00",
    "request_id": "...",
    "total_frames": 1500
  }
}
```

## Architecture

```
Video → Detect (YOLO26s) → Track (OC-SORT) → Classify Teams (HSV)
                                    ↓
                            Ball Detect → Ball Track (Kalman) → Shot Detect
                                    ↓
                          Calibrate (Homography) → Speed + Direction
                                    ↓
                              Render + Output (.mp4 + .json)
```

```
src/
├── model/          Data classes (Detection, Track, Frame, Event)
├── repository/     I/O (VideoReader, OutputWriter)
├── service/        Business logic (8 services)
├── controller/     Pipeline orchestrator (ThreadPool)
└── view/           Renderer (Supervision) + Serializer
```

## Development

### Run tests

```bash
pytest tests/ -v
```

40 tests covering all services and pipeline integration.

### Train custom models

Notebooks in `notebooks/`:
- `01_train_player_detector.ipynb` — fine-tune YOLO for football
- `02_train_ball_detector.ipynb` — specialized ball detection
- `03_finetune_team_classifier.ipynb` — compare HSV vs OSNet
- `04_evaluate_pipeline.ipynb` — GS-HOTA evaluation

### Domain glossary

See [CONTEXT.md](CONTEXT.md) for the project's ubiquitous language.

### Architecture decisions

See [docs/adr/](docs/adr/) for recorded architectural decisions.

## License

MIT
