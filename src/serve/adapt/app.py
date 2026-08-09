"""FastAPI application — HTTP + WebSocket adapter for the football tracker."""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import load_config
from src.serve.pipeline.pipeline import Pipeline
from src.serve.pipeline.video_reader import VideoReader
from src.serve.artifacts.serializer import Serializer
from src.serve.adapt.schemas import (
    AnalyzeRequest, AnalyzeResponse, AnalyzeResult,
    ErrorDetail, ErrorResponse,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "default.yaml"

_sessions: dict[str, dict] = {}
_serializer = Serializer()


def create_app() -> FastAPI:
    """Build the FastAPI app with pipeline routes.

    Returns:
        A FastAPI application instance.
    """
    app = FastAPI(title="Football Tracker API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/v1/analyze")
    async def analyze_video(req: AnalyzeRequest):
        """Download a YouTube video and return a session for streaming."""
        url = req.url

        tmp_dir = tempfile.mkdtemp(prefix="football_")
        video_path = Path(tmp_dir) / "input.mp4"

        # Download with yt-dlp.
        try:
            subprocess.run(
                ["yt-dlp", "-o", str(video_path), "--no-playlist", url],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except subprocess.CalledProcessError as e:
            raise HTTPException(400, detail=f"Download failed: {e.stderr.decode()}")

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        session_id = str(uuid.uuid4())[:8]
        _sessions[session_id] = {
            "video_path": str(video_path),
            "tmp_dir": tmp_dir,
            "queue": asyncio.Queue(),
            "done": False,
            "results": [],
        }

        return JSONResponse(AnalyzeResponse(
            result=AnalyzeResult(
                session_id=session_id,
                total_frames=total_frames,
            ),
        ).model_dump())

    @app.websocket("/v1/stream/{session_id}")
    async def stream_frames(ws: WebSocket, session_id: str):
        """Stream annotated frames as JPEG bytes over WebSocket.

        Each message is a binary JPEG frame.  A final JSON text message
        contains the complete per-frame data.
        """
        await ws.accept()

        session = _sessions.get(session_id)
        if session is None:
            await ws.send_json(ErrorResponse(
                error=ErrorDetail(code="UNKNOWN_SESSION", message="Session not found"),
            ).model_dump())
            await ws.close()
            return

        # Start pipeline in a background thread .
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        def _run_pipeline():
            config = load_config(CONFIG_PATH)
            config.pipeline.display = False
            config.calibration.enabled = False

            pipeline = Pipeline(config)
            reader = VideoReader(session["video_path"])

            for frame in reader:
                annotated, processed = pipeline.process_frame(
                    frame.image, frame.index,
                )
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                loop.call_soon_threadsafe(
                    session["queue"].put_nowait, ("frame", buf.tobytes()),
                )
                session["results"].append(processed)

            loop.call_soon_threadsafe(
                session["queue"].put_nowait, ("done", None),
            )

        executor.submit(_run_pipeline)

        # Stream frames to the WebSocket client.
        try:
            while True:
                msg_type, payload = await asyncio.wait_for(
                    session["queue"].get(), timeout=300,
                )
                if msg_type == "frame":
                    await ws.send_bytes(payload)
                elif msg_type == "done":
                    # Send final JSON results.
                    items = _serializer.serialize_all(session["results"])
                    await ws.send_json(items)
                    break
        except (asyncio.TimeoutError, WebSocketDisconnect):
            pass
        finally:
            await ws.close()
            _sessions.pop(session_id, None)
            executor.shutdown(wait=False)

    return app


app = create_app()
