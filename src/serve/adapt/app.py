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
    """Build the FastAPI app with MJPEG streaming endpoints."""
    app = FastAPI(title="Football Tracker API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/v1/analyze")
    async def analyze(req: AnalyzeRequest):
        """Download a YouTube video and create a session."""
        tmp_dir = tempfile.mkdtemp(prefix="football_")
        video_path = Path(tmp_dir) / "input.mp4"
        loop = asyncio.get_event_loop()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                await loop.run_in_executor(
                    pool,
                    lambda: subprocess.run(
                        ["yt-dlp", "-o", str(video_path), "--no-playlist", req.url],
                        check=True, capture_output=True, timeout=120,
                    ),
                )
        except subprocess.CalledProcessError as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise HTTPException(400, detail=f"Download failed: {e.stderr.decode()}")
        session = _build_session(str(video_path))
        return JSONResponse(AnalyzeResponse(result=AnalyzeResult(
            session_id=session.session_id, total_frames=session.total_frames,
        )).model_dump())

    @app.post("/v1/upload")
    async def upload(file: UploadFile):
        """Accept a video file upload and create a session."""
        tmp_dir = tempfile.mkdtemp(prefix="football_")
        video_path = Path(tmp_dir) / file.filename
        video_path.write_bytes(await file.read())
        session = _build_session(str(video_path))
        return JSONResponse(UploadResponse(result=AnalyzeResult(
            session_id=session.session_id, total_frames=session.total_frames,
        )).model_dump())

    @app.get("/v1/stream/{session_id}")
    def stream(session_id: str):
        """Stream annotated frames as MJPEG.  One client per session."""
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
        """Return processing progress for a session."""
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(404, detail="Session not found")
        return JSONResponse(ProgressResponse(result=ProgressResult(
            processed=session.producer.processed,
            total=session.total_frames,
        )).model_dump())

    return app


app = create_app()
