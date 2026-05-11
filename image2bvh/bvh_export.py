"""Multi-person BVH export.

Takes a list of :class:`image2bvh.pose.PersonPose` and writes one BVH
file per detected idx via the pure-Python writer in
:mod:`image2bvh.bvh_writer`. Earlier versions optionally shelled out to
Blender Portable for an alternative export path; that backend has been
removed.
"""
from __future__ import annotations

import logging
from pathlib import Path

from . import bvh_writer, paths
from .pose import PersonPose

log = logging.getLogger(__name__)


def _safe_filename(prefix: str, idx: int, ext: str = ".bvh") -> str:
    safe_prefix = "".join(c if c.isalnum() or c in "._-" else "_" for c in prefix).strip("._-")
    if not safe_prefix:
        safe_prefix = "person"
    return f"{safe_prefix}_{idx:02d}{ext}"


def export_poses(
    poses: list[PersonPose],
    *,
    output_dir: Path | None = None,
    filename_prefix: str = "person",
) -> list[Path]:
    """Write one ``.bvh`` per pose; return the list of written paths.

    The destination directory is wiped before writing — outputs from the
    previous run do not leak into this one. The default destination is
    :data:`image2bvh.paths.TMP_DIR` (``<project_root>/tmp``).
    """
    if not poses:
        return []

    if output_dir is None:
        out_dir = paths.reset_tmp()
    else:
        out_dir = output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for pose in poses:
        out_path = (out_dir / _safe_filename(filename_prefix, pose.idx, ".bvh")).resolve()
        log.info("Exporting bvh for #%d → %s", pose.idx, out_path)
        bvh_text = bvh_writer.write_bvh(
            pose.joint_names,
            list(pose.joint_parents),
            pose.posed_joint_coords,
        )
        out_path.write_text(bvh_text, encoding="utf-8")
        written.append(out_path)
    return written
