"""Tests for pipeline stage timing reports."""

import json

from src.serve.artifacts.benchmark import StageTimer
from src.serve.artifacts.output_writer import OutputWriter


def test_stage_timer_reports_median_and_p95() -> None:
    timer = StageTimer(clock=iter([0.0, 0.010, 0.020, 0.050]).__next__)
    timer.measure("detect", lambda: None)
    timer.measure("detect", lambda: None)
    report = timer.report()
    assert report["detect"]["count"] == 2
    assert report["detect"]["median_ms"] == 20.0
    assert report["detect"]["p95_ms"] == 29.0


def test_write_benchmark_persists_stage_report(tmp_path) -> None:
    output = tmp_path / "benchmark.json"
    OutputWriter().write_benchmark(
        {"track": {"count": 1}, "total_ms": 4.0}, str(output)
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert "track" in report
    assert report["total_ms"] == 4.0
