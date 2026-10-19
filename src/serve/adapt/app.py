"""FastAPI application - MJPEG streaming adapter for the football tracker."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import cv2
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.config import load_config
from src.logger import get_logger
from src.serve.adapt.producer import FrameProducer
from src.serve.adapt.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzeResult,
    ProgressResponse,
    ProgressResult,
    UploadResponse,
)
from src.serve.adapt.session import Session
from src.serve.adapt.streaming import mjpeg_generator
from src.serve.pipeline.pipeline import Pipeline
from src.serve.pipeline.video_reader import VideoReader

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "default.yaml"
MAX_UPLOAD_BYTES = 2 * 1024**3
SESSION_TTL_S = 30 * 60
MAX_CONCURRENT_SESSIONS = int(os.environ.get("FT_MAX_SESSIONS", "1"))
CORS_ORIGINS = os.environ.get(
    "FT_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
).split(",")
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}

_sessions: dict[str, Session] = {}
_slots = threading.BoundedSemaphore(MAX_CONCURRENT_SESSIONS)
_log = get_logger(__name__)


def _rm_dir(video_path: str) -> None:
    shutil.rmtree(Path(video_path).parent, ignore_errors=True)


def _purge_expired() -> None:
    """Drop sessions idle past the TTL (never streamed, or finished long ago)."""
    now = time.monotonic()
    for sid, s in list(_sessions.items()):
        streaming = s.active_client and not s.producer.done.is_set()
        if not streaming and now - s.last_active > SESSION_TTL_S:
            _sessions.pop(sid, None)
            _rm_dir(s.video_path)


def _probe_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    finally:
        cap.release()


async def _build_session(video_path: str) -> Session:
    """Validate the video and register a Session; cleans up and 400s if invalid."""
    total = await run_in_threadpool(_probe_frames, video_path)
    if total <= 0:
        _rm_dir(video_path)
        raise HTTPException(400, detail="Not a readable video")
    _purge_expired()
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
    try:
        config = load_config(CONFIG_PATH)
        config.pipeline.display = False
        config.calibration.enabled = False
        pipeline = Pipeline(config)
        reader = VideoReader(session.video_path)
        pipeline.reset(reader.fps)
        for frame in reader:
            if session.producer.stop_flag.is_set():
                break
            annotated, _ = pipeline.process_frame(frame.image, frame.index, frame.timestamp)
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            session.producer.put(buf.tobytes())
    except Exception as e:  # noqa: BLE001 — any failure must end the stream and be reported
        _log.exception("pipeline_failed", session_id=session.session_id)
        session.error = f"{type(e).__name__}: {e}"
    finally:
        session.producer.mark_done()


def _check_youtube_url(url: str) -> None:
    u = urlparse(url)
    if u.scheme not in ("http", "https") or u.hostname not in YOUTUBE_HOSTS:
        raise HTTPException(400, detail="Only http(s) YouTube URLs are supported")


def create_app() -> FastAPI:
    """Build the FastAPI app with MJPEG streaming endpoints."""
    app = FastAPI(title="Football Tracker API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/v1/analyze")
    async def analyze(req: AnalyzeRequest):
        """Download a YouTube video and create a session."""
        _check_youtube_url(req.url)
        tmp_dir = tempfile.mkdtemp(prefix="football_")
        video_path = Path(tmp_dir) / "input.mp4"
        try:
            await run_in_threadpool(
                subprocess.run,
                ["yt-dlp", "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
                 "--merge-output-format", "mp4", "-o", str(video_path),
                 "--no-playlist", "--", req.url],
                check=True, capture_output=True, timeout=120,
            )
        except subprocess.CalledProcessError as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise HTTPException(400, detail=f"Download failed: {e.stderr.decode(errors='replace')}")
        except subprocess.TimeoutExpired:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise HTTPException(400, detail="Download timed out")
        except FileNotFoundError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise HTTPException(503, detail="yt-dlp is not installed on the server")
        if not video_path.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise HTTPException(400, detail="Download produced no mp4 file")
        session = await _build_session(str(video_path))
        return JSONResponse(AnalyzeResponse(result=AnalyzeResult(
            session_id=session.session_id, total_frames=session.total_frames,
        )).model_dump())

    @app.post("/v1/upload")
    async def upload(file: UploadFile):
        """Accept a video file upload and create a session."""
        tmp_dir = tempfile.mkdtemp(prefix="football_")
        suffix = Path(file.filename or "").suffix.lower()
        video_path = Path(tmp_dir) / f"input{suffix if suffix in VIDEO_SUFFIXES else '.mp4'}"
        size = 0
        with video_path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out.close()
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    raise HTTPException(413, detail="Upload too large")
                out.write(chunk)
        session = await _build_session(str(video_path))
        return JSONResponse(UploadResponse(result=AnalyzeResult(
            session_id=session.session_id, total_frames=session.total_frames,
        )).model_dump())

    @app.get("/v1/stream/{session_id}")
    def stream(session_id: str):
        """Stream annotated frames as MJPEG.  One client per session.

        Returns 429 when MAX_CONCURRENT_SESSIONS streams are already running.
        """
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(404, detail="Session not found")
        if not _slots.acquire(blocking=False):
            raise HTTPException(429, detail="Server busy, try again later")
        if not session.claim():
            _slots.release()
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
                _rm_dir(session.video_path)
                session.last_active = time.monotonic()
                _slots.release()

        return StreamingResponse(
            _gen(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/v1/progress/{session_id}")
    def progress(session_id: str):
        """Return processing progress for a session."""
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(404, detail="Session not found")
        return JSONResponse(ProgressResponse(result=ProgressResult(
            processed=session.producer.processed,
            total=session.total_frames,
            done=session.producer.done.is_set(),
            error=session.error,
        )).model_dump())

    return app


app = create_app()
