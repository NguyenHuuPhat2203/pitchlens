# Streaming Architecture — MJPEG with Thread-Safe Backpressure

> **Date:** 2025-08-09
> **Status:** Approved
> **Scope:** Replace the current WebSocket streaming adapter with an MJPEG-over-HTTP stream, plus an upload endpoint and progress endpoint.

## Goal

Stream annotated football frames to a browser as a live video feed. Client disconnects end the session immediately (no resume). JSON analysis output is deferred — the architecture leaves a seam but implements nothing for it yet.

## Architecture

### Threading model

The pipeline and the MJPEG generator both run in threads. They communicate through a `queue.Queue` (threading), never an `asyncio.Queue`.

```
Pipeline thread (started on client connect)      MJPEG generator (anyio threadpool)
        │                                                 │
        │  queue.put(jpeg, timeout=0.5)  ← blocks        │  queue.get(timeout=0.5)
        │  (full → timeout → re-check stop_flag)         │  (empty → re-check done)
        ▼                                                 ▼
              queue.Queue(maxsize=100)   [threading]
```

### Backpressure (no frame drop)

The pipeline blocks up to 0.5 s when the queue is full. This throttles the pipeline to the client's consumption rate without dropping frames — the client sees every frame. On client disconnect, `stop_flag` is set; the blocked `put()` times out, the loop re-checks the flag, and the thread exits.

### Lifecycle

```
POST /v1/analyze {url}
  → download video (executor, timeout 120s)
  → scan total frame count
  → create Session (pipeline NOT started)
  → return session_id

GET /v1/stream/{id}
  → atomically claim session (one client per session)
  → start pipeline thread
  → stream MJPEG until done or disconnect
  → cleanup (stop thread, delete temp video)

POST /v1/upload  (multipart file)
  → save temp file
  → create Session (same as analyze, no download)
  → return session_id
```

The pipeline starts at connect time, not at analyze time, so no GPU work is wasted before a client is watching.

### Components

**FrameProducer** (transport-agnostic):
- `queue.Queue(maxsize=100)` — frame buffer
- `stop_flag: threading.Event` — set on client disconnect
- `done: threading.Event` — set when video ends
- `processed: int` — frame counter (single writer, GIL-atomic reads)
- `put(jpeg, timeout)` / `get(timeout)` — both non-blocking with timeout

**Session** (in-memory, no disk writes for frames):
- `session_id`, `video_path`, `total_frames`
- `producer: FrameProducer`
- `thread: threading.Thread`
- `active_client: bool` guarded by `threading.Lock`

### MJPEG format

```
--frame\r\n
Content-Length: <n>\r\n
Content-Type: image/jpeg\r\n
\r\n
<jpeg bytes>\r\n
```

`Content-Type: multipart/x-mixed-replace; boundary=frame`.

### Cleanup

Client disconnect or stream end triggers `finally`:
1. `producer.stop()` — set stop flag
2. `thread.join()` — no timeout; the thread exits at the next frame boundary
3. remove session from the registry
4. delete the temporary video file

### Endpoints

| Method | Path | Response |
|--------|------|----------|
| POST | `/v1/analyze` | `{status, result: {session_id, total_frames}}` |
| POST | `/v1/upload` | `{status, result: {session_id, total_frames}}` |
| GET | `/v1/stream/{id}` | MJPEG stream |
| GET | `/v1/progress/{id}` | `{processed, total}` |

### Error handling

- Unknown session → 404
- Second client → 409 (one client per session)
- Download failure → 400 with error detail
- Pipeline crash → `done` set, stream ends normally

### Files

```
src/serve/adapt/
├── producer.py     # FrameProducer
├── app.py          # analyze, upload, stream, progress
└── schemas.py      # AnalyzeRequest/Response, UploadResponse, ProgressResponse
```

### Testing

- `test_producer.py` — put/get timeout, stop/done flag semantics
- `test_stream.py` — MJPEG boundary format, 409 on second client, cleanup on disconnect
