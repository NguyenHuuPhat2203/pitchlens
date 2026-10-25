# Tactical Timeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an offline tactical timeline with explicit heuristic evidence, training-ready candidate exports, and measured batch throughput improvements.

**Architecture:** `TimelineBuilder` consumes completed `ProcessedFrame` data after video analysis; it emits a flat event list and never participates in per-frame GPU inference. The pipeline skips all annotation allocation and video encoding when neither display nor video output is requested, then serializes the timeline alongside the existing per-frame item schema.

**Tech Stack:** Python 3.12, dataclasses, NumPy, OpenCV, Pydantic, pytest, Ultralytics ONNX/TensorRT.

---

## File structure

- Create `src/model/tactical_event.py`: immutable consumer-facing timeline event.
- Create `src/features/timeline_builder.py`: derive possession transitions and shots from processed frames.
- Create `src/serve/artifacts/benchmark.py`: stage-timing accumulator and JSON report writer.
- Create `src/training/timeline_export.py`: export reviewed candidate metadata without a model dependency.
- Modify `src/config.py`, `config/default.yaml`: timeline thresholds and output paths.
- Modify `src/serve/pipeline/pipeline.py`: collect timings, avoid rendering in JSON-only batches, build and persist a timeline.
- Modify `src/serve/artifacts/output_writer.py`: write the timeline and benchmark reports.
- Modify `src/serve/artifacts/serializer.py`: serialize existing ball/events fully so the timeline has a stable evidence record.
- Create focused tests in `tests/test_timeline_builder.py`, `tests/test_benchmark.py`, and `tests/test_timeline_export.py`; extend `tests/test_output_writer.py` and `tests/test_pipeline.py`.

## Task 1: Define the timeline contract and validated settings

**Files:**
- Create: `src/model/tactical_event.py`
- Modify: `src/config.py`
- Modify: `config/default.yaml`
- Test: `tests/test_timeline_builder.py`

- [ ] **Step 1: Write failing contract tests**

```python
from src.model.tactical_event import TacticalEvent


def test_tactical_event_keeps_explicit_evidence() -> None:
    event = TacticalEvent(
        frame_start=10,
        frame_end=12,
        type="pass",
        actor_track_id=4,
        team="A",
        confidence=0.75,
        pitch_position=None,
        source="heuristic",
    )
    assert event.actor_track_id == 4
    assert event.source == "heuristic"
```

Add a config test which loads YAML containing `timeline.enabled`, `timeline.max_ball_player_distance_px`, `timeline.min_possession_frames`, `timeline.output_json`, and `timeline.benchmark_json`; assert invalid non-positive distance/frame values are rejected by Pydantic.

- [ ] **Step 2: Run the contract tests to verify they fail**

Run: `pytest tests/test_timeline_builder.py -v`

Expected: FAIL because `src.model.tactical_event` and `TimelineConfig` do not exist.

- [ ] **Step 3: Add the smallest data model and configuration**

```python
# src/model/tactical_event.py
from dataclasses import dataclass


@dataclass(frozen=True)
class TacticalEvent:
    frame_start: int
    frame_end: int
    type: str
    actor_track_id: int | None
    team: str | None
    confidence: float
    pitch_position: tuple[float, float] | None
    source: str
```

Add `TimelineConfig(BaseModel)` to `src/config.py` with defaults:

```python
enabled: bool = True
max_ball_player_distance_px: PositiveFloat = 80.0
min_possession_frames: PositiveInt = 3
output_json: Optional[str] = None
benchmark_json: Optional[str] = None
```

Add `timeline: TimelineConfig = TimelineConfig()` to `AppConfig`, and the matching `timeline` block in `config/default.yaml`. Do not introduce classifier weights, model providers, or a second output API.

- [ ] **Step 4: Run the contract tests to verify they pass**

Run: `pytest tests/test_timeline_builder.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model/tactical_event.py src/config.py config/default.yaml tests/test_timeline_builder.py
git commit -m "feat: define tactical timeline contract"
```

## Task 2: Build conservative heuristic events from frame facts

**Files:**
- Create: `src/features/timeline_builder.py`
- Test: `tests/test_timeline_builder.py`

- [ ] **Step 1: Write failing behavior tests**

Create `ProcessedFrame` fixtures with a ball in play and two players. Cover:

```python
def test_emits_pass_when_possession_changes_teams() -> None:
    timeline = TimelineBuilder(80.0, 3).build(frames)
    assert [(event.type, event.actor_track_id, event.team) for event in timeline] == [
        ("pass", 1, "A"),
    ]


def test_does_not_invent_event_when_ball_is_unowned() -> None:
    assert TimelineBuilder(20.0, 3).build(frames) == []


def test_converts_existing_shot_to_tactical_event() -> None:
    assert TimelineBuilder(80.0, 3).build(frames)[0].type == "shot"
```

The pass test must use a stable owner for three frames, then a stable player on the other team for three frames. Include a noisy one-frame nearest-player change and assert it produces no event.

- [ ] **Step 2: Run the timeline tests to verify they fail**

Run: `pytest tests/test_timeline_builder.py -v`

Expected: FAIL because `TimelineBuilder` does not exist.

- [ ] **Step 3: Implement ownership and event derivation**

Implement `TimelineBuilder(max_ball_player_distance_px, min_possession_frames)` with:

```python
def build(self, frames: list[ProcessedFrame]) -> list[TacticalEvent]:
    # Find nearest visible player within max distance for every in-play ball.
    # Confirm ownership only after min_possession_frames consecutive frames.
    # Emit one pass for stable owner transitions; use the prior owner's ID/team.
    # Convert each ShotEvent to a heuristic shot assigned to the confirmed owner.
```

Use `math.dist` between `ball.centroid` and each player bbox centre. Ignore `ball is None`, `ball.state != "in_play"`, no player inside the threshold, players without a team, and duplicate transitions. For a shot, use the last confirmed owner at or before `ShotEvent.frame_number`; use `None` actor/team when unavailable. Set heuristic confidence to `0.7` for a confirmed pass and `0.8` for an existing shot. `pitch_position` remains `None`: do not pretend the existing calibrator exposes it.

- [ ] **Step 4: Run the timeline tests to verify they pass**

Run: `pytest tests/test_timeline_builder.py -v`

Expected: PASS, including the noisy-transition and missing-evidence cases.

- [ ] **Step 5: Commit**

```bash
git add src/features/timeline_builder.py tests/test_timeline_builder.py
git commit -m "feat: derive conservative tactical timeline"
```

## Task 3: Persist a batch timeline and remove JSON-only render work

**Files:**
- Modify: `src/serve/pipeline/pipeline.py`
- Modify: `src/serve/artifacts/output_writer.py`
- Modify: `src/serve/artifacts/serializer.py`
- Modify: `tests/test_output_writer.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing persistence and batch-mode tests**

Add one writer test:

```python
def test_write_timeline_preserves_event_contract(tmp_path) -> None:
    output = tmp_path / "timeline.json"
    OutputWriter().write_timeline([event], str(output))
    assert json.loads(output.read_text(encoding="utf-8"))[0]["source"] == "heuristic"
```

Add one pipeline test with `display=False`, `output_video=None`, and `output_json`/`timeline.output_json` set. Mock `pipeline._renderer.render_tracks` and assert it is not called; assert the per-frame JSON and timeline JSON both exist. Do not mock timeline behavior itself.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `pytest tests/test_output_writer.py tests/test_pipeline.py -v`

Expected: FAIL because `write_timeline` is absent and JSON-only processing still renders.

- [ ] **Step 3: Make pipeline output conditional on requested artifacts**

Add `OutputWriter.write_timeline(events: list[TacticalEvent], output_path: str)` that serializes `dataclasses.asdict(event)` as one ordered JSON array. In `Pipeline.run`, keep `processed_frames`, but only allocate `annotated_frames` and call `_renderer.render_tracks`, ball drawing, shot overlay, `cv2.imshow`, or `write_video` if display or `output_video` is requested.

Change `_process_frame_image` to accept `render: bool`; return `None` for `annotated` when false. Keep `process_frame()` passing `render=True` because MJPEG consumes annotated images. After the run loop, call `TimelineBuilder.build(processed_frames)` and write it when `timeline.output_json` is configured. Extend `Serializer.serialize` to include a JSON-safe ball object and existing `ShotEvent` fields; do not replace the item schema with the timeline.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `pytest tests/test_output_writer.py tests/test_pipeline.py -v`

Expected: PASS. The JSON-only path processes source frames, writes both JSON artifacts, and never invokes the renderer.

- [ ] **Step 5: Commit**

```bash
git add src/serve/pipeline/pipeline.py src/serve/artifacts/output_writer.py src/serve/artifacts/serializer.py tests/test_output_writer.py tests/test_pipeline.py
git commit -m "feat: export timeline without rendering batches"
```

## Task 4: Measure stage timings before changing inference engines

**Files:**
- Create: `src/serve/artifacts/benchmark.py`
- Modify: `src/serve/pipeline/pipeline.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: Write failing timing tests**

```python
def test_stage_timer_reports_median_and_p95() -> None:
    timer = StageTimer(clock=iter([0.0, 0.010, 0.020, 0.050]).__next__)
    timer.measure("detect", lambda: None)
    timer.measure("detect", lambda: None)
    report = timer.report()
    assert report["detect"]["count"] == 2
    assert report["detect"]["median_ms"] == 20.0
    assert report["detect"]["p95_ms"] == 29.0
```

Add a writer test proving the JSON report includes each observed stage and a `total_ms` value.

- [ ] **Step 2: Run timing tests to verify they fail**

Run: `pytest tests/test_benchmark.py -v`

Expected: FAIL because `StageTimer` does not exist.

- [ ] **Step 3: Implement a small local timer**

Implement `StageTimer` using `time.perf_counter`, a `measure(name, fn, *args, **kwargs)` wrapper, and a `report()` that computes count, median, P95 via `numpy.percentile`, and total milliseconds. Time decode, `player_track`, ball detection, render (only when rendered), encode/write, and timeline build in `Pipeline.run`/`_process_frame_image`. Write the report only when `timeline.benchmark_json` is configured.

Do not export TensorRT or modify ONNX inference in this task: profiling must identify the actual bottleneck first.

- [ ] **Step 4: Run timing tests to verify they pass**

Run: `pytest tests/test_benchmark.py tests/test_pipeline.py -v`

Expected: PASS and a configured run produces a parseable benchmark report.

- [ ] **Step 5: Commit**

```bash
git add src/serve/artifacts/benchmark.py src/serve/pipeline/pipeline.py tests/test_benchmark.py tests/test_pipeline.py
git commit -m "perf: record pipeline stage timings"
```

## Task 5: Export reviewable training candidates, not an unvalidated classifier

**Files:**
- Create: `src/training/timeline_export.py`
- Create: `tests/test_timeline_export.py`
- Modify: `src/serve/artifacts/output_writer.py`

- [ ] **Step 1: Write failing candidate-export tests**

```python
def test_export_candidate_writes_reviewer_label_template(tmp_path) -> None:
    path = export_candidates(
        [event], "matches/sample.mp4", tmp_path / "candidates.jsonl"
    )
    candidate = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert candidate["clip"] == {
        "video_path": "matches/sample.mp4", "frame_start": 10, "frame_end": 12,
    }
    assert candidate["review"] == {
        "type": None, "actor_track_id": None, "team": None,
        "frame_start": None, "frame_end": None,
    }

Add a second test proving an `unknown` event is exported too; reviewers need false-positive evidence.

- [ ] **Step 2: Run the candidate-export tests to verify they fail**

Run: `pytest tests/test_timeline_export.py -v`

Expected: FAIL because `export_candidates` does not exist.

- [ ] **Step 3: Implement a JSONL review manifest**

Implement:

```python
def export_candidates(
    events: Iterable[TacticalEvent], video_path: str | Path, output_path: str | Path
) -> Path:
    # One JSON object per candidate with a source-video clip reference,
    # immutable heuristic evidence, and blank reviewer labels.
```

Each record includes `candidate_id` (`"{frame_start}-{frame_end}-{type}"`), `clip` (`video_path`, `frame_start`, `frame_end`), `heuristic` (`asdict(event)`), `event` (the heuristic type), and the blank `review` object in the test. Create the parent directory. This intentionally does not extract video bytes or download FOOTPASS: the manifest supplies exact clip references without adding an unreviewed training pipeline.

- [ ] **Step 4: Run the candidate-export tests to verify they pass**

Run: `pytest tests/test_timeline_export.py -v`

Expected: PASS, with valid JSONL records for normal and `unknown` candidates.

- [ ] **Step 5: Commit**

```bash
git add src/training/timeline_export.py tests/test_timeline_export.py
git commit -m "feat: export timeline review candidates"
```

## Task 6: Verify the complete offline path and choose the next model work from data

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a focused batch-analysis example**

Document a `config/default.yaml` invocation with `pipeline.display: false`, `pipeline.output_video: null`, `pipeline.output_json`, `timeline.output_json`, and `timeline.benchmark_json`. State that `timeline.output_json` is heuristic output and the candidate manifest requires human review before training.

- [ ] **Step 2: Run static checks**

Run: `ruff check src/ tests/ main.py`

Expected: PASS.

- [ ] **Step 3: Run the targeted behavior suite**

Run: `pytest tests/test_timeline_builder.py tests/test_benchmark.py tests/test_timeline_export.py tests/test_output_writer.py tests/test_pipeline.py -v`

Expected: PASS.

- [ ] **Step 4: Run an actual JSON-only batch smoke test**

Run: `python main.py` with a temporary YAML configuration pointing at `data/sample.mp4`, `display: false`, no video output, and configured frame/timeline/benchmark JSON paths.

Expected: exit successfully; all three JSON files exist; timeline is an array; benchmark includes `player_track`, and renderer-specific timing is absent.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: describe offline tactical analysis"
```

## Deferred model work

Fine-tune the action classifier only after candidate reviews provide a held-out target-camera evaluation set and SoccerNet FOOTPASS access is approved. The next plan should define FOOTPASS ingestion, clip extraction, model architecture, per-class F1/actor attribution evaluation, and the 1 percentage-point TensorRT regression gate. No fake classifier, model fallback, or training notebook is added here.
