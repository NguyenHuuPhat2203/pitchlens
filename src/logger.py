"""Shared structured logger — JSON lines to stdout"""

from __future__ import annotations

import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
)


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Return a structured logger for *name*.

    Args:
        name: Typically ``__name__`` from the calling module.

    Returns:
        A bound structlog logger.
    """
    return structlog.get_logger(name)
