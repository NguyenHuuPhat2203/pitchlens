# 04 — ThreadPool Parallelism

**What to build:** Refactor the pipeline to use `concurrent.futures.ThreadPoolExecutor` so frame N+1 is read from disk while frame N is being detected. Detection and tracking for the current frame run sequentially (GPU-bound), but I/O overlaps with compute. The pipeline accepts a `max_workers` config for the CPU pool.

**Blocked by:** 03 — Tracking (OC-SORT)

**Status:** ready-for-agent

- [ ] `src/controller/pipeline.py`: Split into `_read_async`, `_process_async`, `_render_async`. Frame N reading overlaps with frame N-1 processing. Use `ThreadPoolExecutor` for I/O pool (1 thread) and CPU pool (configurable, default 2).
- [ ] `Pipeline.run` uses a futures dict keyed by frame index. Completed futures are flushed and rendered in order to preserve video sequence.
- [ ] `tests/test_pipeline.py`: verifies pipeline completes without deadlocks, output frames match input frame count.
