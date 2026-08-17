# Streaming MJPEG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the WebSocket streaming adapter with an MJPEG-over-HTTP stream with thread-safe backpressure, plus upload and progress endpoints.

**Architecture:** Pipeline thread and MJPEG generator thread communicate via `queue.Queue`. Pipeline starts at client connect, blocks up to 0.5s when the queue is full (no frame drop). Client disconnect sets a stop flag that the pipeline checks at each frame boundary.

**Tech Stack:** FastAPI, Starlette StreamingResponse, threading.Queue, OpenCV.

## Global Constraints

- Python ≥ 3.10
- FastAPI ≥ 0.100 (already installed)
- No disk writes for video frames — all in-memory
- One client per session (second client → 409)
- Pipeline starts at connect, not at analyze
- Item schema `{status, result}` per ML serving rules
- Ruff clean, pytest passing

---

### Task 1: FrameProducer

**Files:**
- Create: `src/serve/adapt/producer.py`
- Test: `tests/test_producer.py`

**Interfaces:**
- Produces: `FrameProducer` with methods `put(jpeg, timeout)`, `get(timeout)`, `stop()`, `mark_done()`, attributes `stop_flag`, `done`, `processed`.

- [ ] **Step 1: Write the failing test**

```python
import queue
import threading
import time

from src.serve.adapt.producer import FrameProducer


def test_put_and_get_roundtrip():
    p = FrameProducer(maxsize=10)
    p.put(b"frame1")
    assert p.get(timeout=0.1) == b"frame1"


def test_put_blocks_when_full_then_times_out():
    p = FrameProducer(maxsize=2)
    p.put(b"a")
    p.put(b"b")
    start = time.monotonic()
    p.put(b"c", timeout=0.2)  # full -> block then timeout
    elapsed = time.monotonic() - start
    assert elapsed >= 0.15


def test_get_times_out_when_empty():
    p = FrameProducer(maxsize=2)
    start = time.monotonic()
    assert p.get(timeout=0.2) is None
    assert time.monotonic() - start >= 0.15


def test_stop_flag_set():
    p = FrameProducer()
    p.stop()
    assert p.stop_flag.is_set()


def test_done_flag_set():
    p = FrameProducer()
    p.mark_done()
    assert p.done.is_set()


def test_processed_increments():
    p = FrameProducer()
    p.put(b"a")
    p.put(b"b")
    assert p.processed == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_producer.py -v`
Expected: FAIL — module `src.serve.adapt.producer` not found

- [ ] **Step 3: Write minimal implementation**

```python
"""Thread-safe frame buffer between pipeline thread and stream generator."""

from __future__ import annotations

import queue
import threading


class FrameProducer:
    """Bounded frame queue with stop/done flags.

    The pipeline thread calls ``put``; the MJPEG generator calls
    ``get``.  Both use timeouts so neither thread blocks forever.
    """

    def __init__(self, maxsize: int = 100) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.stop_flag = threading.Event()
        self.done = threading.Event()
        self.processed = 0

    def put(self, jpeg: bytes, timeout: float = 0.5) -> None:
        """Enqueue a frame.  Blocks up to *timeout* when full."""
        try:
            self._queue.put(jpeg, timeout=timeout)
            self.processed += 1
        except queue.Full:
            pass

    def get(self, timeout: float = 0.5) -> bytes | None:
        """Dequeue a frame.  Returns None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Signal the pipeline to halt at the next frame boundary."""
        self.stop_flag.set()

    def mark_done(self) -> None:
        """Signal that the video has been fully processed."""
        self.done.set()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_producer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/serve/adapt/producer.py tests/test_producer.py
git commit -m "feat(stream): add FrameProducer with bounded queue and stop/done flags"
```

---

### Task 2: Session Management

**Files:**
- Create: `src/serve/adapt/session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: `FrameProducer` from Task 1
- Produces: `Session` class with `claim() -> bool` and `release()`

- [ ] **Step 1: Write the failing test**

```python
import threading

from src.serve.adapt.session import Session
from src.serve.adapt.producer import FrameProducer


def test_claim_first_client_succeeds():
    s = Session(
        session_id="abc",
        video_path="x.mp4",
        total_frames=10,
        producer=FrameProducer(),
    )
    assert s.claim() is True
    assert s.active_client is True


def test_claim_second_client_rejected():
    s = Session(
        session_id="abc",
        video_path="x.mp4",
        total_frames=10,
        producer=FrameProducer(),
    )
    assert s.claim() is True
    assert s.claim() is False


def test_release_resets_client():
    s = Session(
        session_id="abc",
        video_path="x.mp4",
        total_frames=10,
        producer=FrameProducer(),
    )
    s.claim()
    s.release()
    assert s.active_client is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_session.py -v`
Expected: FAIL — module `src.serve.adapt.session` not found

- [ ] **Step 3: Write minimal implementation**

```python
"""Session object — one per analysis request."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from src.serve.adapt.producer import FrameProducer


@dataclass
class Session:
    """In-memory state for one video analysis session.

    One client per session.  ``claim`` atomically reserves the session
    for a client; ``release`` frees it.
    """

    session_id: str
    video_path: str
    total_frames: int
    producer: FrameProducer
    thread: threading.Thread | None = None
    active_client: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def claim(self) -> bool:
        """Atomically reserve the session.  False if already taken."""
        with self._lock:
            if self.active_client:
                return False
            self.active_client = True
            return True

    def release(self) -> None:
        """Free the session for another client."""
        with self._lock:
            self.active_client = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_session.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/serve/adapt/session.py tests/test_session.py
git commit -m "feat(stream): add Session with atomic single-client claim"
```

---

### Task 3: MJPEG Stream Endpoint

**Files:**
- Create: `src/serve/adapt/streaming.py`
- Test: `tests/test_streaming.py`

**Interfaces:**
- Consumes: `FrameProducer` (Task 1), `Session` (Task 2)
- Produces: `mjpeg_generator(session) -> Iterator[bytes]`

- [ ] **Step 1: Write the failing test**

```python
from src.serve.adapt.producer import FrameProducer
from src.serve.adapt.session import Session
from src.serve.adapt.streaming import format_frame, mjpeg_generator


def test_format_frame_has_boundary_and_length():
    jpeg = b"\xff\xd8fakejpeg"
    out = format_frame(jpeg)
    assert out.startswith(b"--frame\r\n")
    assert b"Content-Length: " + str(len(jpeg)).encode() in out
    assert b"Content-Type: image/jpeg\r\n" in out
    assert out.endswith(jpeg + b"\r\n")


def test_generator_yields_frames_until_done():
    session = Session(
        session_id="a",
        video_path="x",
        total_frames=2,
        producer=FrameProducer(maxsize=10),
    )
    session.producer.put(b"f1")
    session.producer.put(b"f2")
    session.producer.mark_done()

    gen = mjpeg_generator(session)
    chunks = list(gen)
    # first chunk = boundary header, then each frame, ends after done
    joined = b"".join(chunks)
    assert b"f1" in joined
    assert b"f2" in joined
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_streaming.py -v`
Expected: FAIL — module `src.serve.adapt.streaming` not found

- [ ] **Step 3: Write minimal implementation**

```python
"""MJPEG stream helpers."""

from __future__ import annotations

import queue
from collections.abc import Iterator

from src.serve.adapt.session import Session

BOUNDARY = b"--frame\r\n"
CONTENT_TYPE = b"Content-Type: image/jpeg\r\n\r\n"


def format_frame(jpeg: bytes) -> bytes:
    """Wrap a JPEG in a multipart MJPEG part."""
    return (
        BOUNDARY
        + b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
        + CONTENT_TYPE
        + jpeg
        + b"\r\n"
    )


def mjpeg_generator(session: Session) -> Iterator[bytes]:
    """Yield MJPEG chunks until done or the producer is stopped."""
    yield format_frame(b"")  # initial boundary
    while True:
        try:
            jpeg = session.producer.get(timeout=0.5)
        except queue.Empty:
            jpeg = None
        if jpeg is None:
            if session.producer.done.is_set() or session.producer.stop_flag.is_set():
                break
            continue
        yield format_frame(jpeg)
        if session.producer.done.is_set():
            break
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_streaming.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/serve/adapt/streaming.py tests/test_streaming.py
git commit -m "feat(stream): add MJPEG generator with multipart framing"
```

---

### Task 4: Wire Endpoints (analyze, upload, stream, progress)

**Files:**
- Modify: `src/serve/adapt/app.py` — replace WebSocket with MJPEG + upload + progress
- Modify: `src/serve/adapt/schemas.py` — add UploadResponse, ProgressResponse
- Test: `tests/test_app_endpoints.py`

**Interfaces:**
- Consumes: `FrameProducer`, `Session`, `mjpeg_generator`, `format_frame` (Tasks 1-3)
- Produces: FastAPI app with 4 endpoints

- [ ] **Step 1: Write the failing test**

```python
import pytest
from fastapi.testclient import TestClient

from src.serve.adapt.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def test_upload_endpoint(client):
    resp = client.post(
        "/v1/upload",
        files={"file": ("clip.mp4", b"fakevideo", "video/mp4")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert "session_id" in body["result"]


def test_stream_unknown_session_returns_404(client):
    resp = client.get("/v1/stream/nonexistent")
    assert resp.status_code == 404


def test_progress_endpoint(client):
    resp = client.get("/v1/progress/nonexistent")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app_endpoints.py -v`
Expected: FAIL — upload endpoint not present, stream returns WebSocket error

- [ ] **Step 3: Write schemas additions**

In `src/serve/adapt/schemas.py`, add:

```python
class UploadResponse(BaseModel):
    """POST /v1/upload success response."""

    status: str = "success"
    result: AnalyzeResult


class ProgressResult(BaseModel):
    """Result payload inside ProgressResponse."""

    processed: int
    total: int


class ProgressResponse(BaseModel):
    """GET /v1/progress/{id} response."""

    status: str = "success"
    result: ProgressResult
```

- [ ] **Step 4: Rewrite app.py**

Replace the entire file with:

```python
"""FastAPI application — MJPEG streaming adapter for the football tracker."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.config import load_config
from src.logger import get_logger
from src.serve.adapt.producer import FrameProducer
from src.serve.adapt.schemas import (
    AnalyzeRequest, AnalyzeResponse, AnalyzeResult,
    ProgressResponse, ProgressResult, UploadResponse,
)
from src.serve.adapt.session import Session
from src.serve.adapt.streaming import mjpeg_generator
from src.serve.pipeline.pipeline import Pipeline
from src.serve.pipeline.video_reader import VideoReader

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "default.yaml"
_sessions: dict[str, Session] = {}
_log = get_logger(__name__)


def _build_session(video_path: str) -> Session:
    """Create a Session for an already-downloaded video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    sid = str(uuid.uuid4())[:8]
    session = Session(
        session_id=sid,
        video_path=video_path,
        total_frames=total,
        producer=FrameProducer(),
    )
    _sessions[sid] = session
    return session


def _run_pipeline(session: Session) -> None:
    """Process frames in a background thread."""
    config = load_config(CONFIG_PATH)
    config.pipeline.display = False
    config.calibration.enabled = False
    pipeline = Pipeline(config)
    reader = VideoReader(session.video_path)

    for frame in reader:
        if session.producer.stop_flag.is_set():
            break
        annotated, _ = pipeline.process_frame(frame.image, frame.index)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        session.producer.put(buf.tobytes(), timeout=0.5)

    session.producer.mark_done()


def create_app() -> FastAPI:
    app = FastAPI(title="Football Tracker API")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.post("/v1/analyze")
    async def analyze(req: AnalyzeRequest):
        tmp_dir = tempfile.mkdtemp(prefix="football_")
        video_path = Path(tmp_dir) / "input.mp4"
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            await loop.run_in_executor(
                pool,
                lambda: subprocess.run(
                    ["yt-dlp", "-o", str(video_path), "--no-playlist", req.url],
                    check=True, capture_output=True, timeout=120,
                ),
            )
        session = _build_session(str(video_path))
        return JSONResponse(AnalyzeResponse(result=AnalyzeResult(
            session_id=session.session_id, total_frames=session.total_frames,
        )).model_dump())

    @app.post("/v1/upload")
    async def upload(file: UploadFile):
        tmp_dir = tempfile.mkdtemp(prefix="football_")
        video_path = Path(tmp_dir) / file.filename
        video_path.write_bytes(await file.read())
        session = _build_session(str(video_path))
        return JSONResponse(UploadResponse(result=AnalyzeResult(
            session_id=session.session_id, total_frames=session.total_frames,
        )).model_dump())

    @app.get("/v1/stream/{session_id}")
    def stream(session_id: str):
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(404, detail="Session not found")
        if not session.claim():
            raise HTTPException(409, detail="Session already in use")

        def _gen():
            try:
                session.thread = threading.Thread(
                    target=_run_pipeline, args=(session,), daemon=True,
                )
                session.thread.start()
                yield from mjpeg_generator(session)
            finally:
                session.producer.stop()
                if session.thread is not None:
                    session.thread.join()
                _sessions.pop(session_id, None)

        return StreamingResponse(
            _gen(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/v1/progress/{session_id}")
    def progress(session_id: str):
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(404, detail="Session not found")
        return JSONResponse(ProgressResponse(result=ProgressResult(
            processed=session.producer.processed,
            total=session.total_frames,
        )).model_dump())

    return app


app = create_app()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_app_endpoints.py -v`
Expected: PASS

- [ ] **Step 6: Run full suite + lint**

Run: `pytest tests/ -q && ruff check src/ tests/ main.py`
Expected: All pass, ruff clean

- [ ] **Step 7: Commit**

```bash
git add src/serve/adapt/app.py src/serve/adapt/schemas.py tests/test_app_endpoints.py
git commit -m "feat(stream): replace WebSocket with MJPEG endpoints (analyze, upload, stream, progress)"
```

---

### Task 5: Frontend Update

**Files:**
- Modify: `demo/index.html` — replace WebSocket client with `<img>` + progress poll

**Interfaces:**
- Consumes: MJPEG stream at `GET /v1/stream/{id}`, progress at `GET /v1/progress/{id}`

- [ ] **Step 1: Rewrite the JS**

Replace the `<script>` block in `demo/index.html`:

```html
<script>
async function start() {
  const url = document.getElementById("url").value.trim();
  if (!url) return;
  const btn = document.getElementById("start");
  btn.disabled = true;
  document.getElementById("status").textContent = "Downloading...";

  const res = await fetch("http://localhost:8000/v1/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  const data = await res.json();
  const sid = data.result.session_id;

  document.getElementById("img").src = `http://localhost:8000/v1/stream/${sid}`;
  document.getElementById("img").style.display = "block";
  document.getElementById("status").textContent = "Streaming...";

  const timer = setInterval(async () => {
    const p = await fetch(`http://localhost:8000/v1/progress/${sid}`);
    const pd = await p.json();
    document.getElementById("stats").textContent =
      `Frame ${pd.result.processed}/${pd.result.total}`;
  }, 1000);
}
</script>
```

- [ ] **Step 2: Update the body**

Replace the `<canvas>` with:

```html
<img id="img" style="display:none; width:100%; border-radius:8px;">
```

- [ ] **Step 3: Commit**

```bash
git add demo/index.html
git commit -m "feat(demo): switch frontend to MJPEG img tag with progress polling"
```

---

### Task 6: Cleanup old WebSocket code

**Files:**
- Modify: `README.md` — update endpoints table
- Verify: no leftover WebSocket references

- [ ] **Step 1: Grep for WebSocket leftovers**

Run: `grep -rn "websocket\|WebSocket\|ws://" src/ demo/ tests/`
Expected: no results (or only in comments/docs to be cleaned)

- [ ] **Step 2: Update README endpoints table**

Replace the endpoints section with:

```
| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/analyze` | YouTube URL → session_id |
| POST | `/v1/upload` | File upload → session_id |
| GET | `/v1/stream/{id}` | MJPEG video stream |
| GET | `/v1/progress/{id}` | `{processed, total}` |
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README endpoints for MJPEG streaming"
```
