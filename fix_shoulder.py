"""Post-process a single-frame BVH to apply the scapulohumeral rhythm:
transfer ALPHA fraction of each UpperArm's local rotation onto its
parent Shoulder, preserving the upper arm's *world* rotation. The
visual effect is that the clavicle now elevates/protracts with arm
motion (anatomically, ~1/3 of overhead reach belongs to the
scapulothoracic joint, 2/3 to the glenohumeral joint).

OFFSETs (HIERARCHY) are untouched; only the Shoulder and UpperArm
Euler XYZ rotations in the MOTION frame are rewritten.

Usage:
    python fix_shoulder.py <input.bvh> [output.bvh] [--alpha 0.333]
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np


_DEFAULT_ALPHA = 1.0 / 3.0


def _rx(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _ry(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rz(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def euler_xyz_to_mat(rx: float, ry: float, rz: float) -> np.ndarray:
    return _rx(math.radians(rx)) @ _ry(math.radians(ry)) @ _rz(math.radians(rz))


def mat_to_euler_xyz_deg(R: np.ndarray) -> tuple[float, float, float]:
    sb = max(-1.0, min(1.0, float(R[0, 2])))
    if abs(sb) < 0.99999:
        beta = math.asin(sb)
        alpha = math.atan2(-float(R[1, 2]), float(R[2, 2]))
        gamma = math.atan2(-float(R[0, 1]), float(R[0, 0]))
    else:
        beta = math.copysign(math.pi / 2.0, sb)
        alpha = math.atan2(float(R[2, 1]), float(R[1, 1]))
        gamma = 0.0
    return math.degrees(alpha), math.degrees(beta), math.degrees(gamma)


def rotation_power(R: np.ndarray, alpha: float) -> np.ndarray:
    """R**alpha for a 3x3 rotation matrix via axis-angle scaling."""
    cos_a = max(-1.0, min(1.0, 0.5 * (float(np.trace(R)) - 1.0)))
    angle = math.acos(cos_a)
    if angle < 1e-9:
        return np.eye(3)
    sin_a = math.sin(angle)
    if sin_a < 1e-9:
        M = (R + np.eye(3)) * 0.5
        col = max(range(3), key=lambda j: float(np.linalg.norm(M[:, j])))
        v = M[:, col]
        nv = float(np.linalg.norm(v))
        axis = v / nv if nv > 1e-9 else np.array([1.0, 0.0, 0.0])
    else:
        axis = np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
        ) / (2.0 * sin_a)
    new_angle = angle * alpha
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + math.sin(new_angle) * K + (1 - math.cos(new_angle)) * (K @ K)


_NAME_RE = re.compile(r"^\s*(ROOT|JOINT)\s+(\S+)")
_CHAN_RE = re.compile(r"CHANNELS\s+(\d+)\s+(.*)")


def parse_hierarchy(lines: list[str]) -> tuple[list[str], list[int]]:
    names: list[str] = []
    chan_counts: list[int] = []
    pending: str | None = None
    for line in lines:
        if line.strip().startswith("MOTION"):
            break
        m = _NAME_RE.match(line)
        if m:
            pending = m.group(2)
            continue
        m = _CHAN_RE.search(line)
        if m and pending is not None:
            names.append(pending)
            chan_counts.append(int(m.group(1)))
            pending = None
    return names, chan_counts


def fix_bvh(text: str, alpha: float) -> str:
    lines = text.splitlines()
    motion_idx = next(i for i, l in enumerate(lines) if l.strip() == "MOTION")
    names, chan_counts = parse_hierarchy(lines[:motion_idx])

    offsets: list[int] = []
    cursor = 0
    for c in chan_counts:
        offsets.append(cursor)
        cursor += c

    frame_idx = None
    for i in range(motion_idx + 1, len(lines)):
        s = lines[i].strip()
        if not s or s.startswith("Frames:") or s.startswith("Frame Time:"):
            continue
        frame_idx = i
        break
    if frame_idx is None:
        raise RuntimeError("no motion frame")

    values = [float(x) for x in lines[frame_idx].split()]

    for shoulder, upperarm in (
        ("LeftShoulder", "LeftUpperArm"),
        ("RightShoulder", "RightUpperArm"),
    ):
        if shoulder not in names or upperarm not in names:
            continue
        si = names.index(shoulder)
        ai = names.index(upperarm)
        s_off = offsets[si] + chan_counts[si] - 3
        a_off = offsets[ai] + chan_counts[ai] - 3
        sx, sy, sz = values[s_off], values[s_off + 1], values[s_off + 2]
        ax, ay, az = values[a_off], values[a_off + 1], values[a_off + 2]
        Rs = euler_xyz_to_mat(sx, sy, sz)
        Ra = euler_xyz_to_mat(ax, ay, az)
        Ra_alpha = rotation_power(Ra, alpha)
        Rs_new = Rs @ Ra_alpha
        # composite (Rs @ Ra) preserved via Ra_new = Ra_alpha.T @ Ra
        Ra_new = Ra_alpha.T @ Ra
        nsx, nsy, nsz = mat_to_euler_xyz_deg(Rs_new)
        nax, nay, naz = mat_to_euler_xyz_deg(Ra_new)
        values[s_off], values[s_off + 1], values[s_off + 2] = nsx, nsy, nsz
        values[a_off], values[a_off + 1], values[a_off + 2] = nax, nay, naz
        print(
            f"  {shoulder}: ({sx:.2f},{sy:.2f},{sz:.2f}) -> "
            f"({nsx:.2f},{nsy:.2f},{nsz:.2f})"
        )
        print(
            f"  {upperarm}: ({ax:.2f},{ay:.2f},{az:.2f}) -> "
            f"({nax:.2f},{nay:.2f},{naz:.2f})"
        )

    lines[frame_idx] = " ".join(f"{v:.6f}" for v in values)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def main() -> None:
    args = sys.argv[1:]
    alpha = _DEFAULT_ALPHA
    if "--alpha" in args:
        i = args.index("--alpha")
        alpha = float(args[i + 1])
        del args[i : i + 2]
    if not args:
        print(__doc__)
        sys.exit(1)
    src = Path(args[0])
    dst = Path(args[1]) if len(args) >= 2 else src
    text = src.read_text(encoding="utf-8")
    print(f"alpha = {alpha:.4f}")
    fixed = fix_bvh(text, alpha)
    dst.write_text(fixed, encoding="utf-8")
    print(f"wrote: {dst}")


if __name__ == "__main__":
    main()
