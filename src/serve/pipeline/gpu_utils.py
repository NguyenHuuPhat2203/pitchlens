"""Safe GPU inference helpers — OOM handling per ML rules §05."""

from __future__ import annotations

import gc

import torch


def safe_predict(model, *args, **kwargs):
    """Call model with CUDA OOM recovery.

    On OOM: clear cache, retry on CPU.  Returns the model output
    (or None if both GPU and CPU fail).

    Args:
        model: A YOLO model instance (or any callable returning results).
        *args: Positional args forwarded to the model.
        **kwargs: Keyword args forwarded to the model.

    Returns:
        Model results, or None on unrecoverable error.
    """
    try:
        return model(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        _free_gpu_memory()
        try:
            # Retry on CPU.
            kwargs_cpu = {**kwargs, "device": "cpu"}
            return model(*args, **kwargs_cpu)
        except Exception:
            return None


def _free_gpu_memory() -> None:
    """Release available GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
