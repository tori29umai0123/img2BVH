"""GPU memory profiling helpers, safe to call on CPU-only systems.

All functions return ``None`` / empty dicts / no-op when CUDA is unavailable
so callers can sprinkle them throughout the pipeline without conditional
guards. Numbers are reported in megabytes (``1024 * 1024`` bytes).

Typical usage from the inference worker:

    vram.reset_peak()
    instances = segmentation.extract_instances(...)
    log(vram.report("after SAM3"))

    vram.reset_peak()
    poses = pose.estimate_poses(...)
    log(vram.report("after SAM3DBody"))

The final ``snapshot()`` is also surfaced in the API ``complete`` /
``segment_complete`` events so the WebUI can show a peak VRAM badge.
"""
from __future__ import annotations


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def cuda_available() -> bool:
    t = _torch()
    return bool(t and t.cuda.is_available())


def device_info() -> dict:
    """Return ``{name, total_mb}`` or ``{}`` if no CUDA device."""
    if not cuda_available():
        return {}
    t = _torch()
    props = t.cuda.get_device_properties(0)
    return {
        "name": t.cuda.get_device_name(0),
        "total_mb": props.total_memory / (1024 * 1024),
    }


def reset_peak() -> None:
    if cuda_available():
        _torch().cuda.reset_peak_memory_stats()


def snapshot() -> dict:
    """Return ``{current_mb, peak_mb, reserved_mb}`` or ``{}`` if no CUDA."""
    if not cuda_available():
        return {}
    t = _torch()
    return {
        "current_mb": t.cuda.memory_allocated() / (1024 * 1024),
        "peak_mb": t.cuda.max_memory_allocated() / (1024 * 1024),
        "reserved_mb": t.cuda.memory_reserved() / (1024 * 1024),
    }


def report(label: str) -> str:
    """Format a one-line report suitable for the WebUI log panel.

    Numbers are MB; ``current`` is the live allocator footprint, ``peak`` is
    the high-water mark since the last :func:`reset_peak` call (or process
    start), ``reserved`` is what PyTorch holds in its caching allocator
    (≥ current; typically what nvidia-smi shows for the process).
    """
    if not cuda_available():
        return f"[vram] {label}: cuda not available (cpu run)"
    s = snapshot()
    return (
        f"[vram] {label}: current={s['current_mb']:.0f}MB "
        f"peak={s['peak_mb']:.0f}MB reserved={s['reserved_mb']:.0f}MB"
    )
