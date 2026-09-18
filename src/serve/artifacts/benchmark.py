"""Stage timing collection for offline pipeline runs."""

from __future__ import annotations

import time
from collections.abc import Callable
from statistics import median
from typing import Any

import numpy as np


class StageTimer:
    """Collect elapsed samples and summarize them in milliseconds."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._samples: dict[str, list[float]] = {}

    def measure(self, name: str, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """Run ``fn`` and record its elapsed time under ``name``."""
        start = self._clock()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (self._clock() - start) * 1000.0
            self._samples.setdefault(name, []).append(elapsed_ms)

    def add(self, name: str, elapsed_ms: float) -> None:
        """Add an already measured sample."""
        self._samples.setdefault(name, []).append(float(elapsed_ms))

    def report(self) -> dict[str, dict[str, float] | float]:
        """Return per-stage count, median, p95, and total milliseconds."""
        report: dict[str, dict[str, float] | float] = {}
        total_ms = 0.0
        for name, samples in self._samples.items():
            total_ms += sum(samples)
            report[name] = {
                "count": len(samples),
                "median_ms": round(float(median(samples)), 6),
                "p95_ms": round(float(np.percentile(samples, 95)), 6),
            }
        report["total_ms"] = round(total_ms, 6)
        return report
