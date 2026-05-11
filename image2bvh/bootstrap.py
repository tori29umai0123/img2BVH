"""First-run setup: download SAM3 and SAM 3D Body weights.

HuggingFace repo IDs are read from ``config.ini`` so the user can track
upstream renames or swap in a fork without touching the source.

Each downloader is idempotent — re-running after a successful download
is a no-op. Designed to be invoked from the API's ``/api/run`` handler
so the first inference call transparently downloads what is missing.

Earlier versions also auto-downloaded Blender Portable for BVH/FBX
export. That path has been removed: BVH is now emitted by the pure-
Python writer in ``image2bvh/bvh_writer.py`` and FBX export is no
longer supported. ``runtime/blender/`` left over from a previous
install can be deleted by hand.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from . import config, paths

log = logging.getLogger(__name__)

ProgressCb = Callable[[str], None]


def _emit(progress: ProgressCb | None, msg: str) -> None:
    log.info(msg)
    if progress is not None:
        try:
            progress(msg)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Hugging Face downloads
# --------------------------------------------------------------------------- #


def ensure_sam3(progress: ProgressCb | None = None) -> Path:
    paths.ensure_dirs()
    target = paths.SAM3_DIR
    if (target / "config.json").is_file():
        _emit(progress, f"[SAM3] already present at {target}")
        return target

    repo_id = config.get("models", "sam3_repo_id")
    _emit(progress, f"[SAM3] downloading {repo_id} → {target}")
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.md", "*.gitattributes", "*.bin"],
        )
        if not (target / "config.json").is_file():
            snapshot_download(repo_id=repo_id, local_dir=str(target), local_dir_use_symlinks=False)
    except GatedRepoError as exc:
        # Meta gates SAM 3 with manual approval. Surface clear guidance.
        raise RuntimeError(_gated_message(repo_id)) from exc
    except RepositoryNotFoundError as exc:
        raise RuntimeError(
            f"[SAM3] HuggingFace repo {repo_id!r} not found. "
            f"Edit [models] sam3_repo_id in config.ini if upstream renamed it. "
            f"Underlying error: {exc}"
        ) from exc
    except HfHubHTTPError as exc:
        # 401 typically means missing / invalid HF_TOKEN against a gated repo.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise RuntimeError(_gated_message(repo_id)) from exc
        raise

    _emit(progress, "[SAM3] download complete")
    return target


def _gated_message(repo_id: str) -> str:
    return (
        f"[SAM3] HuggingFace repo {repo_id!r} is GATED (manual approval by Meta).\n\n"
        f"  1) Open https://huggingface.co/{repo_id} and click 'Request access'.\n"
        f"     Fill in the required fields (name, date of birth, country, affiliation,\n"
        f"     job title). Approval is not instant — it can take hours to days.\n\n"
        f"  2) Once approved, create an access token at:\n"
        f"     https://huggingface.co/settings/tokens  (read scope is enough)\n\n"
        f"  3) Make the token visible to image2BVH by either:\n"
        f"        - setting the environment variable  HF_TOKEN=<your_token>\n"
        f"          before launching run.bat / run.sh, OR\n"
        f"        - running once:  uv run --no-dev huggingface-cli login\n\n"
        f"  4) Retry. Already-downloaded files under runtime/models/sam3/ are kept.\n"
    )


def ensure_sam3dbody(progress: ProgressCb | None = None) -> Path:
    paths.ensure_dirs()
    target = paths.SAM3DBODY_DIR
    if paths.sam3dbody_ready():
        _emit(progress, f"[SAM3DBody] already present at {target}")
        return target

    repo_id = config.get("models", "sam3dbody_repo_id")
    _emit(progress, f"[SAM3DBody] downloading {repo_id} → {target}")
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=str(target), local_dir_use_symlinks=False)
    if not paths.sam3dbody_ready():
        raise RuntimeError(
            f"[SAM3DBody] download finished but expected files missing under {target}.\n"
            f"  required: model.ckpt and assets/mhr_model.pt"
        )
    _emit(progress, "[SAM3DBody] download complete")
    return target


# --------------------------------------------------------------------------- #
# Status / all-in-one
# --------------------------------------------------------------------------- #


def status() -> dict:
    return {
        "sam3": paths.sam3_ready(),
        "sam3dbody": paths.sam3dbody_ready(),
    }


def is_ready() -> bool:
    s = status()
    return bool(s["sam3"] and s["sam3dbody"])


def ensure_missing(progress: ProgressCb | None = None) -> dict:
    """Download only what is currently missing. Returns the same shape as
    :func:`status` after downloads complete."""
    paths.ensure_dirs()
    if not paths.sam3_ready():
        ensure_sam3(progress)
    if not paths.sam3dbody_ready():
        ensure_sam3dbody(progress)
    return status()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ensure_missing(progress=lambda m: print(m))
