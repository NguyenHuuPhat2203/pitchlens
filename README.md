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

1. Pipeline runs — annotated frames appear in an OpenCV window
2. Press Q to quit

Pitch calibration is automatic, also headless: the pitch is found from the painted
lines (a few seconds of CPU at the start), then followed frame by frame, so speeds stay
correct while the camera pans. Frames where the pitch cannot be read (close-ups, most
side-on broadcast shots) get no speed. Details: `research/calibration/`.
Calibration runs in a child process, so a script that builds a `Pipeline` needs
the usual `if __name__ == "__main__":` guard on Windows (`main.py` has it).

### GPU acceleration

On Windows, `pip install -r requirements.txt` alone installs a CPU-only
`torch`. Install the CUDA build first, inside the project `.venv`:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

With CUDA, inference runs in FP16 by default (`model.half: null`). The
pipeline converts each `.onnx` model once to a cached `*.fp16.onnx` beside
it. Set `model.half: false` to force FP32.

For another ~24% speed-up, install the TensorRT runtime (optional, ~1 GB, must be 10.x):

```bash
.venv/Scripts/python.exe -m pip install "tensorrt-cu12-libs==10.16.1.11"
```

With it, both detectors run on ONNX Runtime's TensorRT EP (`model.tensorrt: true`).
The first run builds the engines (several minutes per model) and caches
them in `.cache/trt/`; later runs start in seconds. Without the package, or
with `model.tensorrt: false`, the CUDA EP is used. Measurements:
`research/experiments/h3b-trt-end-to-end/analysis.md`.

On CUDA, frames are also letterboxed on the GPU. Player tracking uses
`src/serve/pipeline/ocsort.py`, an exact batched port of Ultralytics' OC-SORT:
same tracks and IDs, about 2.5x cheaper on the CPU. `tests/test_ocsort.py`
checks it against Ultralytics, so re-run it before upgrading the pinned
`ultralytics`. Throughput history and the experiments behind it:
`research/findings.md`.

Model downloads, Ultralytics settings, and torch / Hugging Face caches stay
inside the project: `.cache/`, `weights/`, `runs/`, and `data/datasets/`.
`import src` sets this up; the local notebooks do the same in their first cell.

## Features

| Feature | Status |
|---------|--------|
| Player detection + tracking (OC-SORT) | Done |
| Team classification (Lab colour k-means, grass masked) | Done |
| Ball detection + Kalman tracking | Done |
| Shot event detection | Done |
| Pitch calibration (automatic, per frame, from the painted lines) | Done |
| Direction arrows (always on) | Done |
| Speed overlay (km/h, calibration only, foot point) | Done |
| Annotated video output (.mp4) | Done |
| JSON output (item schema) | Done |
| FastAPI backend + MJPEG streaming | Done |
| Browser demo (paste YouTube URL) | Done |

## Architecture

```
YouTube URL → download → Pipeline → MJPEG stream → browser
                                  ↓
                            JSON result array
```

Pipeline tiers (hybrid layout):

```
src/
├── backbone/       YOLO wrappers (detector, ball_detector)
├── features/       Team classifier, timeline builder
├── model/          Data classes (Detection, Track, Frame, Event)
├── serve/
│   ├── pipeline/   Inference engine (tracker, ocsort, ball_tracker,
│   │               calibrator, speed, shot, renderer, video_reader)
│   ├── artifacts/  Serializer (item schema), OutputWriter
│   └── adapt/      FastAPI + MJPEG stream (app, schemas)
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
| POST | `/v1/analyze` | Accept `{"url": "..."}` (http(s) YouTube hosts only), returns `session_id` |
| POST | `/v1/upload` | Multipart video upload (max 2 GB, must decode), returns `session_id` |
| GET | `/v1/stream/{id}` | MJPEG video stream; `429` when `FT_MAX_SESSIONS` (default 1) streams are running |
| GET | `/v1/progress/{id}` | `{processed, total, done, error}`; kept 30 min after the stream ends |

`/v1/analyze` needs `yt-dlp` (in `requirements.txt`). CORS origins come from
`FT_CORS_ORIGINS` (comma-separated; default `http://localhost:3000,http://localhost:5173`).
The stream has no pitch calibration, so it shows no speed.

### Training

Open `notebooks/football_tracker_training.ipynb` in Jupyter or Colab.
Covers dataset download, MOT→YOLO conversion, player fine-tuning,
ball detector training, and pipeline evaluation.

Pitch keypoints for side-on views (research E2, SoccerNet labels):
`notebooks/training/04_train_pitch_keypoints_sn32.ipynb` on Colab. Data and
checkpoints go through a private Hugging Face dataset repo: put `HF_TOKEN` and
`HF_REPO` in `.env` (see `.env.example`), then `python scripts/hf_sync.py data`
once, and `python scripts/hf_sync.py pull runs` to fetch the trained weights.

### Offline tactical analysis

For batch analysis, disable display and video output so the pipeline writes
facts and timeline JSON without annotation work:

```yaml
pipeline:
  display: false
  output_video: null
  output_json: "outputs/frames.json"

timeline:
  enabled: true
  output_json: "outputs/timeline.json"
  benchmark_json: "outputs/benchmark.json"
```

`timeline.json` contains conservative heuristic events (`pass` within a team,
`turnover` across teams, and `shot`)
with `source: "heuristic"`. Export its events with
`src.training.timeline_export.export_candidates` for human review before
training. The manifest stores source-video paths and frame ranges; it does
not extract clips or pretend to be a trained FOOTPASS model.
## Configuration

`config/default.yaml` is a YAML file validated against a Pydantic model
at startup. Key sections:

```yaml
model:
  player_detector: "models/player_detector.onnx"
  detection_conf: 0.3             # Player and ball confidence threshold
  device: "auto"                  # CUDA if available, else CPU

pipeline:
  display: true                   # Show OpenCV window
  output_video: null              # Save annotated .mp4 (written frame by frame)
  output_json: null               # Save per-frame JSON

team_classifier:
  enabled: true                   # Auto-fit on first 50 frames

ball_detector:
  enabled: true
  gate_px: 150.0                  # Reject detections far from the predicted ball

shot_detector:
  angle_threshold: 30.0           # Degrees turned over 0.25 s
  accel_threshold: 800.0          # Image px/s² over 0.25 s (uses the video's fps)

calibration:
  enabled: true                   # Automatic, per frame, from the painted lines
  speed_window: 12                # Frames for speed smoothing
  camera_dir: null                # SynLoc camera model of a fixed known camera instead
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
pytest tests/ -q

# Lint
ruff check src/ tests/ main.py

# Pre-commit
pre-commit install

# Promo video (real pipeline output + motion graphics, rendered in Edge)
.venv/Scripts/python.exe -m pip install playwright imageio-ffmpeg
.venv/Scripts/python.exe promo/make.py   # -> promo/football-tracker-demo.mp4
```

## Docs

- [Glossary](CONTEXT.md)
- [ADR](docs/adr/0001-pipeline-architecture-yolo26s-ocsort.md)
- [Research](docs/research-real-time-football-tracking.md)
- [Implementation plan](docs/implementation-plan.md)
