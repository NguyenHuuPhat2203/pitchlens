"""API request/response DTOs — Pydantic models"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """POST /v1/analyze request body."""

    url: str = Field(..., description="YouTube video URL to analyze")


class AnalyzeResult(BaseModel):
    """Result payload inside AnalyzeResponse."""

    session_id: str
    total_frames: int


class AnalyzeResponse(BaseModel):
    """POST /v1/analyze success response."""

    status: str = "success"
    result: AnalyzeResult


class ErrorDetail(BaseModel):
    """Error payload inside ErrorResponse."""

    code: str
    message: str = ""


class ErrorResponse(BaseModel):
    """Standard error response"""

    status: str = "error"
    error: ErrorDetail
