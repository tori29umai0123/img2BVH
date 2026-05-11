"""Filesystem layout for image2BVH.

Persistent runtime artefacts (downloaded SAM3 / SAM 3D Body weights)
live under ``<project_root>/runtime``. BVH outputs are *transient* and
live under ``<project_root>/tmp`` — cleared before every inference run
and served back through the WebUI for download.

Note: earlier versions also auto-downloaded Blender Portable into
``runtime/blender/`` and shelled out to it for BVH/FBX export. That
path was removed; BVH is now emitted by the pure-Python writer in
``image2bvh/bvh_writer.py`` and FBX export is no longer supported. An
existing ``runtime/blender/`` directory left over from a previous
install can be deleted by hand — nothing in the project references it.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# When frozen by PyInstaller (--onedir), ``__file__`` resolves to a path
# inside ``_internal/`` which is the *bundle* directory (read-only at
# install location ``%LOCALAPPDATA%\Programs\image2BVH\``). Anchor on
# ``sys.executable``'s parent instead so ``runtime/``, ``tmp/`` and
# ``config.ini`` all live next to the EXE — and that directory IS
# writable because the Inno Setup installer drops us under
# ``{localappdata}\Programs\`` (no admin needed).
if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUNTIME_DIR = PROJECT_ROOT / "runtime"
MODELS_DIR = RUNTIME_DIR / "models"
SAM3_DIR = MODELS_DIR / "sam3"
SAM3DBODY_DIR = MODELS_DIR / "sam3dbody"

# Per-run output directory at project root. Wiped at the start of each run.
TMP_DIR = PROJECT_ROOT / "tmp"

VENDOR_DIR = PROJECT_ROOT / "image2bvh" / "vendor"


def ensure_dirs() -> None:
    for p in (RUNTIME_DIR, MODELS_DIR, SAM3_DIR, SAM3DBODY_DIR, TMP_DIR):
        p.mkdir(parents=True, exist_ok=True)


def reset_tmp() -> Path:
    """Wipe and recreate ``tmp/``. Called at the start of each inference run."""
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, ignore_errors=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    return TMP_DIR


def sam3dbody_ckpt_path() -> Path:
    return SAM3DBODY_DIR / "model.ckpt"


def sam3dbody_mhr_path() -> Path:
    return SAM3DBODY_DIR / "assets" / "mhr_model.pt"


def sam3dbody_ready() -> bool:
    return sam3dbody_ckpt_path().is_file() and sam3dbody_mhr_path().is_file()


def sam3_ready() -> bool:
    return (SAM3_DIR / "config.json").is_file()
