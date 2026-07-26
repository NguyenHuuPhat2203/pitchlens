# Implementation Plan — Football Tracker

## Build Order

### Phase 1: Detection + Tracking (core pipeline)
### Phase 2: Team Classification (HSV clustering)
### Phase 3: Ball Detection + Tracking
### Phase 4: Pitch Calibration + Speed Vectors
### Phase 5: Shot Detection

---

## Project Structure

```
football-tracker/
├── CONTEXT.md
├── docs/
│   ├── adr/
│   │   └── 0001-pipeline-architecture-yolo26s-ocsort.md
│   └── research-real-time-football-tracking.md
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── detection.py
│   │   ├── track.py
│   │   ├── event.py
│   │   └── frame.py
│   ├── repository/
│   │   ├── video_reader.py
│   │   └── output_writer.py
│   ├── service/
│   │   ├── detector.py
│   │   ├── tracker.py
│   │   ├── team_classifier.py
│   │   ├── ball_detector.py
│   │   ├── ball_tracker.py
│   │   ├── calibrator.py
│   │   ├── speed_estimator.py
│   │   └── shot_detector.py
│   ├── controller/
│   │   └── pipeline.py
│   └── view/
│       ├── renderer.py
│       └── serializer.py
├── config/
│   ├── default.yaml
│   └── trackers/
│       └── ocsort.yaml
├── notebooks/
│   ├── 01_train_player_detector.ipynb
│   ├── 02_train_ball_detector.ipynb
│   ├── 03_finetune_team_classifier.ipynb
│   └── 04_evaluate_pipeline.ipynb
├── tests/
│   ├── test_detector.py
│   ├── test_tracker.py
│   ├── test_team_classifier.py
│   ├── test_ball_detector.py
│   ├── test_shot_detector.py
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

---

## Parallelism Strategy

### Dependency Graph (per frame)

```
                    ┌─────────────────┐
                    │   VideoReader    │
                    │  (I/O thread)    │
                    └────────┬────────┘
                             │ frame
              ┌──────────────┼──────────────┐
              ▼              ▼              │
    ┌─────────────────┐ ┌─────────────────┐ │
    │   Detector       │ │  BallDetector   │ │
    │  (GPU, ~3ms)     │ │  (GPU, ~1ms)    │ │
    └────────┬────────┘ └────────┬────────┘ │
             │ Detection[]       │ BallDetection
             ▼                   ▼            │
    ┌─────────────────┐ ┌─────────────────┐ │
    │   Tracker        │ │  BallTracker    │ │
    │  (CPU, ~0.5ms)   │ │  (CPU, ~0.3ms)  │ │
    └────────┬────────┘ └────────┬────────┘ │
             │ Track[]           │ BallTrack │
             ▼                   ▼           │
    ┌─────────────────┐ ┌─────────────────┐ │
    │ TeamClassifier   │ │  ShotDetector   │ │
    │  (CPU, ~0.5ms)   │ │  (CPU, ~0.1ms)  │ │
    └────────┬────────┘ └────────┬────────┘ │
             │                   │           │
             └─────────┬─────────┘           │
                       ▼                     │
    ┌─────────────────────────────┐         │
    │     Calibrator              │         │
    │  (CPU, every N frames)      │         │
    └────────────┬────────────────┘         │
                 ▼                           │
    ┌─────────────────────────────┐         │
    │     SpeedEstimator          │         │
    │  (CPU, ~0.2ms)              │         │
    └────────────┬────────────────┘         │
                 ▼                           │
    ┌─────────────────────────────┐         │
    │ Build ProcessedFrame        │         │
    │ (CPU, ~0.1ms)               │         │
    └────────────┬────────────────┘         │
                 │                           │
    ┌────────────┼──────────────┐            │
    ▼            ▼              │            │
┌────────┐ ┌──────────┐        │            │
│Renderer│ │Serializer│        │            │
│(CPU)   │ │(CPU)     │        │            │
└────┬───┘ └────┬─────┘        │            │
     │          │              │            │
     ▼          ▼              ▼            │
┌─────────────────────────────────────┐    │
│         OutputWriter                │    │
│  (I/O thread — write video + JSON)  │    │
└─────────────────────────────────────┘    │
                                           │
Frame N+1 processing starts while Frame N  │
is still rendering/writing.                │
```

### Thread Pool Layout

```
┌─────────────────────────────────────────────────────────┐
│                    MAIN THREAD                           │
│  Pipeline controller — dispatches work, collects results │
└─────────────────────────────────────────────────────────┘
          │                    │                    │
    ┌─────▼─────┐        ┌─────▼─────┐        ┌─────▼─────┐
    │ I/O Pool  │        │ GPU Pool  │        │ CPU Pool  │
    │ 2 threads │        │ 1 thread  │        │ N threads │
    │           │        │           │        │           │
    │ Reader    │        │ Detector  │        │ Tracker   │
    │ Writer    │        │ BallDet   │        │ TeamCls   │
    │           │        │           │        │ Calib     │
    │           │        │           │        │ SpeedEst  │
    │           │        │           │        │ ShotDet   │
    │           │        │           │        │ Renderer  │
    │           │        │           │        │ Serializer│
    └───────────┘        └───────────┘        └───────────┘
```

### What Runs in Parallel

| Parallelism | Threads | Rationale |
|-------------|---------|-----------|
| **Read frame N+1 while processing frame N** | I/O thread + main | I/O không block compute |
| **Detect players + Detect ball** | GPU pool (concurrent CUDA streams) | Hai model YOLO độc lập, cùng GPU. Dùng 2 CUDA stream để không serialize |
| **Track players + Track ball** | CPU pool (2 threads) | Độc lập sau khi có Detection và BallDetection |
| **Team classify + Shot detect** | CPU pool (2 threads) | Team chỉ cần Track[], Shot chỉ cần BallTrack — không phụ thuộc nhau |
| **Render + Serialize + Write** | CPU pool (2 threads) + I/O thread | Render và serialize frame N trong khi frame N+1 đang detect |
| **Calibrator** | CPU pool, throttled to every N frames | Chạy song song với speed estimation của frame trước đó |

### GPU Strategy (Single GPU, 4GB VRAM)

```
CUDA Stream 0:  Player Detector (YOLO26s, ~200MB VRAM)
CUDA Stream 1:  Ball Detector   (YOLO26n or small YOLO, ~150MB VRAM)

Total VRAM for models: ~350MB
Remaining for frame buffers + CUDA context: ~3.5GB
```

Vì chỉ có 1 GPU, true parallelism trên GPU bị giới hạn. Nhưng dùng 2 CUDA stream cho phép **overlap** memory copy của model này với compute của model kia.

### Pipeline Timing (estimated, RTX 3050 Ti laptop)

```
Frame N:
  0ms  ├─ Read frame N (I/O thread) ───────────────────── 1ms
  1ms  ├─ Detect players (GPU stream 0) ───────────────── 4ms
  1ms  ├─ Detect ball    (GPU stream 1) ───────────────── 2ms
  5ms  ├─ Track players  (CPU pool) ───────────────────── 0.5ms
  5ms  ├─ Track ball     (CPU pool) ───────────────────── 0.3ms
  5.5ms├─ Team classify  (CPU pool) ───────────────────── 0.5ms
  5.5ms├─ Shot detect    (CPU pool) ───────────────────── 0.1ms
  6ms  ├─ Calibrate      (CPU pool, every 30 frames) ──── 1ms avg
  6ms  ├─ Speed estimate (CPU pool) ───────────────────── 0.2ms
  6.2ms├─ Build ProcessedFrame ─────────────────────────── 0.1ms
  6.3ms├─ Render frame N (CPU pool) ───────────────────── 2ms
  6.3ms├─ Serialize      (CPU pool) ───────────────────── 0.3ms
  8.3ms├─ Write output   (I/O thread) ─────────────────── 1ms
  9.3ms└─ Frame N complete

Frame N+1:
  1ms  ├─ Read frame N+1 (overlapped with N processing) ─ 1ms
  ... (read và process song song)
```

**~9.3ms per frame ≈ 107 FPS theoretical. Thực tế ~50-60 FPS do GPU scheduling overhead. Thoải mái cho 25-30fps input.**

---

## Notebooks — Training & Finetuning

| Notebook | Mục đích | Dataset | Output |
|----------|---------|---------|--------|
| `01_train_player_detector.ipynb` | Finetune YOLO26s trên football player data. Classes: `player`, `goalkeeper`, `referee` | SoccerNet Tracking | `models/player_detector.pt` |
| `02_train_ball_detector.ipynb` | Train small YOLO model for ball detection at higher resolution (1280px) | SoccerNet GSR + custom annotation | `models/ball_detector.pt` |
| `03_finetune_team_classifier.ipynb` | Experiment with HSV clustering thresholds, try embedding-based approach (OSNet) for comparison | Subset of test video frames | Threshold config in `config/default.yaml` |
| `04_evaluate_pipeline.ipynb` | End-to-end evaluation: run pipeline on annotated clips, compute IDF1, MOTA, GS-HOTA | SoccerNet GSR test set | Metrics report |

---

## Phase 1 Files (First Commit)

```
src/
├── model/
│   ├── detection.py     → Detection dataclass
│   ├── track.py         → PlayerTrack, BallTrack dataclass
│   └── frame.py         → ProcessedFrame dataclass
├── service/
│   ├── detector.py      → YOLO26s inference wrapper
│   └── tracker.py       → OC-SORT wrapper (via ultralytics track mode)
├── controller/
│   └── pipeline.py      → ThreadPool-based frame orchestrator
├── repository/
│   └── video_reader.py  → OpenCV frame generator (iterator, yield từng frame)
└── view/
    └── renderer.py      → Supervision annotators
```

### Phase 1 Key Design Decisions

**`pipeline.py` dùng `concurrent.futures.ThreadPoolExecutor`:**

```python
class Pipeline:
    """Orchestrates services with thread-pool parallelism."""

    def __init__(self, config: dict) -> None:
        self._io_pool = ThreadPoolExecutor(max_workers=2)
        self._cpu_pool = ThreadPoolExecutor(max_workers=4)

    def run(self, source_path: str) -> None:
        reader = VideoReader(source_path)
        pending: dict[int, Future[ProcessedFrame]] = {}

        # Frame N: read in background
        # Frame N-1: process
        # Frame N-2: render + write
        for frame in reader:
            future = self._process_async(frame)
            pending[frame.index] = future
            self._flush_completed(pending)
```

**`video_reader.py` là iterator, không phải load hết vào RAM:**

```python
class VideoReader:
    """Yields frames from a video file."""

    def __init__(self, path: str) -> None: ...

    def __iter__(self) -> Iterator[Frame]: ...
```

### Commit Sequence

1. `chore: initial project structure and dependencies`
2. `feat(model): add Detection, Track, ProcessedFrame dataclasses`
3. `feat(service): add YOLO detector wrapper`
4. `feat(service): add OC-SORT tracker wrapper`
5. `feat(repository): add video reader with frame iterator`
6. `feat(view): add supervision-based renderer`
7. `feat(controller): add parallel pipeline orchestrator`
8. `test: add detector and tracker tests`
9. `docs: add phase 1 completion report`
