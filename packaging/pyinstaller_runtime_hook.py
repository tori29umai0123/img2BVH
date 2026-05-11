"""PyInstaller runtime hook for image2BVH.

Runs before any image2bvh import. Purpose:

1. **Triton JIT cache** — Triton derives its on-disk cache location from
   ``$TRITON_CACHE_DIR`` if set; otherwise it falls back to a path under
   ``$HOME/.triton``. On a frozen --onedir build, falling back to ``$HOME``
   is fine but means every install gets a fresh cache. Pin it to a
   stable location next to the EXE so the JIT compile only happens once
   per machine for a given image2BVH version.

2. **HuggingFace cache** — ``huggingface_hub.snapshot_download`` writes to
   ``$HF_HOME`` (or ``$XDG_CACHE_HOME/huggingface``). For missing-model
   downloads (e.g. a user re-running ``ensure_sam3()`` after deleting a
   weight file) we want those downloads under the install dir, not in
   ``%USERPROFILE%`` where the user can't easily find them.

3. **CUDA DLL search path** — torch's cu130 wheel ships ``cudart64_130.dll``,
   ``cublas64_130.dll``, ``cudnn_*.dll`` etc. under ``torch/lib/``.
   PyInstaller bundles them under ``_internal/torch/lib/`` and adds that
   to the DLL search path automatically. We additionally hint via
   ``CUDA_PATH`` so any vendor library that walks env vars (rare, but
   ``cupy`` and a few others do) finds the bundled DLLs instead of a
   stale system CUDA install.

This hook does NOT decide CUDA-vs-CPU; that's handled at runtime by
``torch.cuda.is_available()``. PyTorch's bundled DLLs probe the user's
NVIDIA driver and gracefully fall back when no compatible GPU is found.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _setup_frozen_paths() -> None:
    if not getattr(sys, "frozen", False):
        return

    # ``sys.executable`` → ``<install_root>\image2bvh.exe``
    install_root = Path(sys.executable).resolve().parent

    # Windowed PyInstaller builds (console=False) launched without a parent
    # console get sys.stdout / sys.stderr == None. Anything that calls
    # ``.isatty()`` (uvicorn's ColourizedFormatter, click's echo, etc.)
    # then dies with AttributeError before the app can boot. Redirect
    # both to a rotating log file next to the EXE so users (and we) can
    # actually read the traceback when something goes wrong.
    if sys.stdout is None or sys.stderr is None:
        log_path = install_root / "image2bvh.log"
        try:
            log_fh = open(log_path, "a", encoding="utf-8", errors="replace", buffering=1)
        except OSError:
            # Install dir not writable for some reason — fall back to /dev/null
            # equivalent so at least .isatty() works.
            log_fh = open(os.devnull, "w", encoding="utf-8", errors="replace")
        if sys.stdout is None:
            sys.stdout = log_fh
        if sys.stderr is None:
            sys.stderr = log_fh

    # Triton: keep the JIT cache persistent so the user doesn't pay the
    # ~5-15s sm_xx codegen tax on every launch. The install dir is user-
    # writable thanks to Inno Setup's ``{localappdata}\Programs\`` target,
    # so we can keep the cache there (vs. AppData) and uninstall cleans it
    # up automatically.
    os.environ.setdefault(
        "TRITON_CACHE_DIR",
        str(install_root / "triton-cache"),
    )

    # HuggingFace: only matters if the user deletes a bundled weight file
    # and triggers a re-download via ``ensure_sam3`` / ``ensure_sam3dbody``.
    # Bundled snapshots under ``runtime/models/`` already satisfy the
    # ``paths.sam3_ready`` checks, so this path is rarely hit.
    os.environ.setdefault(
        "HF_HOME",
        str(install_root / "hf-cache"),
    )

    # CUDA: hint to any library that consults ``CUDA_PATH``. The actual
    # DLL discovery for torch's own use goes through the OS loader and
    # PyInstaller's automatic ``_internal\torch\lib`` add-to-PATH step.
    torch_lib = install_root / "_internal" / "torch" / "lib"
    if torch_lib.is_dir():
        os.environ.setdefault("CUDA_PATH", str(torch_lib))


_setup_frozen_paths()
