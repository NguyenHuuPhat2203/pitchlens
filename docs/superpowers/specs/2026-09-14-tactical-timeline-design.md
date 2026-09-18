# Tactical Timeline Design

## Goal

Add an offline tactical timeline for an analyzed football video. It produces immediately useful candidate events, then optionally refines them with a trained action model. Increase batch throughput without silently reducing analysis quality.

## Scope

- Produce tactical-event candidates from existing player tracks, team labels, ball tracks, and calibration output.
- Preserve one consumer-facing timeline contract for heuristic and model-derived events.
- Fine-tune a player-centric action classifier after an annotated baseline exists.
- Measure and improve batch throughput at pipeline-stage boundaries.

Out of scope: live action prediction, replacing the existing stream API, rebuilding the tracker, and training a calibration model in this delivery.

## Architecture

```text
video -> detect/track/calibrate -> frame facts
                                   |- heuristic event candidates -> timeline
                                   '- candidate clips + facts -> action classifier -> refined timeline
```

The heuristic stage creates events with available evidence. The action classifier runs after frame processing, only on candidate clips, and can refine an event in the same time window. It never executes for every frame.

## Timeline Contract

Each event contains:

- `frame_start` and `frame_end`
- `type`
- `actor_track_id`
- `team`
- `confidence`
- `pitch_position` when calibration is available
- `source`: `heuristic` or `model`

The consumer reads one ordered timeline. A model output replaces a heuristic event only when it has the same actor and its frame-window intersection-over-union is at least 0.5, and its confidence is higher. Events lacking enough ball or actor evidence use `type: unknown`; no inferred action is fabricated.

## Data and Training

### Initial labelled set

Create candidate clips from videos processed by the pipeline. A reviewer corrects action, actor, team, and time bounds. Store the clip reference, labels, and the corresponding player/ball track facts. These labels match the target broadcast conditions and form the evaluation holdout.

### Fine-tuning dataset

Use SoccerNet FOOTPASS for the action model: its player-centric labels cover action, actor, and team. Access requires SoccerNet registration and its video NDA. Fine-tune only after a local baseline and holdout evaluation are available.

SoccerNet GSR and Spiideo SynLoc are future inputs for automatic calibration and richer pitch-position analysis. They are not a dependency of the first timeline delivery.

## Failure Behavior

- Missing model weights, classifier failure, or confidence below threshold leaves heuristic events unchanged and completes the batch.
- Missing calibration omits `pitch_position`; it does not discard the event.
- Timeline processing records model and dataset versions for reproducibility.

## Performance Plan

1. Benchmark decode, player detection, ball detection, tracking, rendering, encoding, and timeline refinement separately.
2. Batch analysis skips frame rendering and MJPEG encoding unless annotated video is explicitly requested.
3. Keep inference on GPU. Compare the current ONNX route against TensorRT FP16 on the target RTX 3050 Ti; adopt TensorRT only if measured faster and the held-out event F1 decreases by no more than 1 percentage point.
4. Run the classifier only for candidate clips.

## Acceptance Metrics

- Heuristic candidate precision/recall on corrected clips.
- Model per-class F1 plus actor/team attribution accuracy on a held-out labelled set.
- Median and P95 latency per pipeline stage, plus end-to-end video throughput.
- Any acceleration change must be compared against the same source videos and report the accuracy delta.

## Tests

- Heuristic timeline tests cover confirmed ownership transitions, missing evidence, and unchanged output when a classifier is unavailable.
- Classifier merge, model failure, and higher-confidence replacement tests belong to the FOOTPASS implementation plan after data access is approved.
- Benchmark command emits stage timings for an analyzed video.
