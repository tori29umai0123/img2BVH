"""SAM 3D Body — per-instance pose estimation.

Wraps the vendored ``sam_3d_body`` package to (a) load the model once,
(b) run pose inference for each detected person from
:mod:`image2bvh.segmentation`, and (c) compute rest- vs. posed-skeleton
joint coordinates and rotations needed by the BVH builder.

Joint subsetting and humanoid name remapping are kept in
:mod:`image2bvh.bvh_export` so this module stays focused on inference.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from threading import Lock
from typing import Iterable

import numpy as np

from . import paths
from .segmentation import Instance

log = logging.getLogger(__name__)

# Default forward-lean correction chain. Tuned interactively to fight the
# combination of (a) SAM 3D Body's tendency to over-fit the camera by
# leaning subjects forward, and (b) Clip Studio Paint's tendency to
# auto-flatten an obviously-rotated rig on import. We split the bend
# across the three thoracic-region joints (spine_01..spine_03) and then
# add a small finishing tilt at neck_01 + head so the gaze ends up
# roughly horizontal even with the rest of the spine pulled back.
#
# Each entry is (joint_idx, base_angle_in_radians). Strength linearly
# scales each joint's rotation and propagates to all descendants;
# strength=0 disables the correction. After the chain runs the pose is
# re-grounded (pelvis X/Z restored + min-Y aligned to the floor) so the
# legs stay vertical and the feet don't sink/float.
_LEAN_CHAIN_DEFAULT: tuple[tuple[int, float], ...] = (
    (35,  math.radians(0.0)),    # spine_01
    (36,  math.radians(20.0)),   # spine_02
    (37,  math.radians(20.0)),   # spine_03
    (113, math.radians(10.0)),   # head
)


def _subtree_indices(parents: np.ndarray, root: int) -> list[int]:
    """Return ``root`` plus every descendant of it under the parents tree."""
    num_joints = int(parents.shape[0])
    children: dict[int, list[int]] = {}
    for j in range(num_joints):
        p = int(parents[j])
        if p >= 0:
            children.setdefault(p, []).append(j)
    out: list[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(children.get(node, ()))
    return out


def _rotx(theta: float) -> np.ndarray:
    """Right-handed rotation matrix around the X axis."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [[1.0, 0.0, 0.0],
         [0.0,   c,  -s],
         [0.0,   s,   c]],
        dtype=np.float32,
    )


def apply_lean_correction(
    posed_joint_rots: np.ndarray,
    posed_joint_coords: np.ndarray,
    parents: np.ndarray,
    strength: float,
    *,
    chain: tuple[tuple[int, float], ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bend the spine→neck→head chain backwards to counter the model's
    forward-lean bias. Implementation ported from the reference repo's
    ``apply_pose_lean_correction_rig`` in
    ``ComfyUI-SAM3DBody_utills/nodes/processing/process.py``.

    Args:
        posed_joint_rots: ``[J, 3, 3]`` per-joint rotation matrices.
        posed_joint_coords: ``[J, 3]`` per-joint world positions.
        parents: ``[J]`` parent index per joint (-1 = root).
        strength: 0.0 → no correction; 1.0 → upstream's "full" preset.
        chain: optional override for the default ``(joint, base_angle)``
            list. ``None`` uses :data:`_LEAN_CHAIN_DEFAULT`.

    Returns:
        ``(updated_rots, updated_coords)`` — fresh arrays. Inputs are
        not mutated.
    """
    rots = posed_joint_rots.astype(np.float32, copy=True)
    coords = posed_joint_coords.astype(np.float32, copy=True)
    if strength is None:
        return rots, coords
    try:
        s = float(strength)
    except (TypeError, ValueError):
        return rots, coords
    if not math.isfinite(s) or s <= 1e-6 or parents is None:
        return rots, coords

    active_chain = chain if chain is not None else _LEAN_CHAIN_DEFAULT
    num_joints = int(parents.shape[0])

    # Snapshot pelvis X/Z and the lowest-Y point BEFORE the chain runs so
    # we can re-ground after: legs stay rotated by the chain (none of
    # spine_01..head reach the legs), but the cumulative spine bend can
    # tip the whole upper body forward enough that the pelvis appears
    # to drift. Pelvis is conventionally joint index 1 in MHR.
    if num_joints > 1:
        pelvis_xz_before = (float(coords[1, 0]), float(coords[1, 2]))
        min_y_before = float(coords[:, 1].min())
    else:
        pelvis_xz_before = None
        min_y_before = None

    for joint_id, base_angle in active_chain:
        if joint_id >= num_joints:
            continue
        theta = s * float(base_angle)
        if abs(theta) < 1e-8:
            continue
        subtree = _subtree_indices(parents, joint_id)
        if not subtree:
            continue

        pivot = coords[joint_id].copy()
        # Negative X-axis rotation tilts the subtree backwards in MHR's
        # (X right, Y up, Z forward) frame.
        corr = _rotx(-theta)
        for k in subtree:
            off = coords[k] - pivot
            coords[k] = pivot + corr @ off
            rots[k] = corr @ rots[k]

    # Counter-rotate the arm chains so the spine bend doesn't propagate
    # an unrealistic shoulder rise onto the arms. Without this, every
    # chain entry whose subtree includes the clavicles (35..37 in MHR)
    # rigidly rotates the arms together with the spine, so the shoulder
    # elevates and the BVH-encoded palm direction tilts with it. CSP
    # then tries to keep the palm roll consistent with where the wrist
    # *should* be pointing in world space and ends up twisting the
    # wrist. We apply the inverse of the cumulative spine rotation to
    # each clavicle subtree, pivoted at the post-bend clavicle position
    # — clavicle attaches to spine_03 (geometric integrity preserved),
    # but the arm direction in world frame stays the same as before
    # the bend.
    clavicle_indices = (74, 38)  # clavicle_l, clavicle_r in MHR
    if num_joints > max(clavicle_indices):
        arm_total_theta = 0.0
        probe = clavicle_indices[1]  # any arm joint works as a probe
        for joint_id, base_angle in active_chain:
            if joint_id >= num_joints:
                continue
            if probe in _subtree_indices(parents, joint_id):
                arm_total_theta += s * float(base_angle)
        if abs(arm_total_theta) > 1e-8:
            counter = _rotx(arm_total_theta)  # +theta cancels the chain's -theta
            for clav_idx in clavicle_indices:
                sub = _subtree_indices(parents, clav_idx)
                if not sub:
                    continue
                pivot = coords[clav_idx].copy()
                for k in sub:
                    off = coords[k] - pivot
                    coords[k] = pivot + counter @ off
                    rots[k] = counter @ rots[k]

    # Re-grounding: shift all joints uniformly so pelvis X/Z matches what
    # it was before the chain and the lowest joint sits back on the
    # floor. Uniform translation doesn't disturb any parent→child offset
    # encoded into the BVH HIERARCHY — it only affects the root world
    # position written to MOTION + the visual ground level in previews.
    if pelvis_xz_before is not None:
        dx = pelvis_xz_before[0] - float(coords[1, 0])
        dz = pelvis_xz_before[1] - float(coords[1, 2])
        dy = (min_y_before - float(coords[:, 1].min())) if min_y_before is not None else 0.0
        if abs(dx) > 1e-9 or abs(dy) > 1e-9 or abs(dz) > 1e-9:
            coords[:, 0] += dx
            coords[:, 1] += dy
            coords[:, 2] += dz

    return rots, coords


_CONVENTION_CHECKED = False


def _check_mhr_convention(  # noqa: PLR0915 — diagnostic, kept for future debugging
    rest_rots: np.ndarray,
    rest_coords: np.ndarray,
    posed_rots: np.ndarray,
    posed_coords: np.ndarray,
    parents: np.ndarray,
) -> None:
    """One-shot diagnostic: log MHR's rotation/position convention so we
    can decide how to write a proper BVH.

    Verifies:
      1. ``rest_rots[i]`` is identity for all joints (joint local frames
         are world-aligned in rest pose).
      2. SMPL kinematic equation holds for direct parent-child pairs:
         ``posed_pos[c] - posed_pos[p] == posed_rots[p] @ (rest_pos[c] -
         rest_pos[p])``. If so, ``posed_rots`` is the cumulative world
         rotation matrix per joint (G[i]).
    """
    global _CONVENTION_CHECKED
    if _CONVENTION_CHECKED:
        return
    _CONVENTION_CHECKED = True

    log.info("=== MHR rotation convention check ===")

    # 1. Sample rest_rots: which joints have non-identity rest frames?
    log.info("--- rest_rots[i] deviation from identity ---")
    for idx in (1, 2, 3, 18, 34, 35, 36, 37, 38, 39, 74, 75, 110, 113):
        if idx < rest_rots.shape[0]:
            R = rest_rots[idx]
            dev = float(np.linalg.norm(R - np.eye(3)))
            tag = " (NOT identity)" if dev > 1e-3 else ""
            log.info("  rest_rots[%d]: dev=%.4f%s", idx, dev, tag)

    # 2. Test multiple hypotheses for the SMPL-X kinematic equation.
    # H1: posed_diff = G_posed[p] @ rest_diff                    (rest frames = world)
    # H2: posed_diff = G_posed[p] @ G_rest[p].T @ rest_diff      (rest frames bone-aligned)
    # H3: posed_diff = G_rest[p] @ G_posed[p] @ G_rest[p].T @ rest_diff
    # H4: posed_diff = (G_posed[p] @ G_rest[p].T) @ (G_rest[c].T @ rest_diff_local? unlikely)
    # H5: posed_diff = G_posed[c] @ G_rest[c].T @ rest_diff
    log.info("--- kinematic equation hypotheses ---")
    log.info("  H1=G_posed[p]·rest, H2=G_posed[p]·G_rest[p]ᵀ·rest, H3=G_rest[p]·G_posed[p]·G_rest[p]ᵀ·rest, H5=G_posed[c]·G_rest[c]ᵀ·rest")
    test_pairs = (
        (34, 35), (35, 36), (36, 37), (37, 38), (38, 39),
        (37, 74), (74, 75), (37, 110), (110, 113),
        (1, 2), (2, 3), (1, 18),
    )
    for p_idx, c_idx in test_pairs:
        if c_idx >= parents.shape[0]:
            continue
        if int(parents[c_idx]) != p_idx:
            continue
        rest_diff = rest_coords[c_idx] - rest_coords[p_idx]
        posed_diff = posed_coords[c_idx] - posed_coords[p_idx]
        rest_len = float(np.linalg.norm(rest_diff))
        if rest_len < 1e-6:
            continue
        Gp = posed_rots[p_idx]
        Gc = posed_rots[c_idx]
        Rp = rest_rots[p_idx]
        Rc = rest_rots[c_idx]
        h1 = Gp @ rest_diff
        h2 = Gp @ Rp.T @ rest_diff
        h3 = Rp @ Gp @ Rp.T @ rest_diff
        h5 = Gc @ Rc.T @ rest_diff
        e1 = float(np.linalg.norm(h1 - posed_diff))
        e2 = float(np.linalg.norm(h2 - posed_diff))
        e3 = float(np.linalg.norm(h3 - posed_diff))
        e5 = float(np.linalg.norm(h5 - posed_diff))
        log.info(
            "  [%d->%d] len=%.3f H1=%.4f H2=%.4f H3=%.4f H5=%.4f",
            p_idx, c_idx, rest_len, e1, e2, e3, e5,
        )

    log.info("=====================================")


def apply_lean_correction_to_pose(p: "PersonPose", strength: float) -> "PersonPose":
    """Return a copy of ``p`` with the forward-lean correction baked into
    its posed coords / rotations. ``strength == 0`` short-circuits and
    returns the input unchanged."""
    if not strength or strength <= 1e-6:
        return p
    parents_arr = np.asarray(p.joint_parents, dtype=np.int32)
    new_rots, new_coords = apply_lean_correction(
        p.posed_joint_rots, p.posed_joint_coords, parents_arr, strength,
    )
    return replace(p, posed_joint_coords=new_coords, posed_joint_rots=new_rots)


# MHR joint index → human-readable bone name. Verbatim from the reference
# repo's ``export_rigged.py``; the BVH builder relies on these names being
# stable so that the humanoid bone subset still resolves correctly.
KNOWN_JOINT_NAMES: dict[int, str] = {
    1:   "pelvis",
    2:   "thigh_l",   3:  "calf_l",   4:  "foot_l",
    18:  "thigh_r",  19:  "calf_r",  20:  "foot_r",
    35:  "spine_01", 36:  "spine_02", 37: "spine_03",
    38:  "clavicle_r", 39: "upperarm_r", 40: "lowerarm_r", 42: "hand_r",
    74:  "clavicle_l", 75: "upperarm_l", 76: "lowerarm_l", 78: "hand_l",
    110: "neck_01",  113: "head",
}


@dataclass
class PersonPose:
    idx: int
    bbox_xyxy: np.ndarray
    score: float
    joint_names: list[str]
    joint_parents: list[int]
    rest_joint_coords: np.ndarray  # [J, 3]
    rest_joint_rots: np.ndarray    # [J, 3, 3]
    posed_joint_coords: np.ndarray
    posed_joint_rots: np.ndarray


_LOCK = Lock()
_LOADED: dict | None = None


def _resolve_device(device_pref: str = "auto") -> str:
    import torch

    if device_pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but not available")
    return device_pref


def load_model(device_pref: str = "auto") -> dict:
    """Load SAM 3D Body once and cache it."""
    global _LOADED
    with _LOCK:
        if _LOADED is not None:
            return _LOADED

        from .vendor.sam_3d_body import load_sam_3d_body

        device = _resolve_device(device_pref)
        ckpt = str(paths.sam3dbody_ckpt_path())
        mhr = str(paths.sam3dbody_mhr_path())
        log.info("Loading SAM 3D Body (device=%s, ckpt=%s)", device, ckpt)
        model, model_cfg, _ = load_sam_3d_body(checkpoint_path=ckpt, device=device, mhr_path=mhr)
        _LOADED = {"model": model, "model_cfg": model_cfg, "device": device, "mhr_head": model.head_pose}
        return _LOADED


def _to_batched_tensor(value, device, width: int):
    import torch

    if value is None:
        return torch.zeros((1, width), dtype=torch.float32, device=device)
    if isinstance(value, torch.Tensor):
        t = value.to(device=device, dtype=torch.float32)
    else:
        t = torch.tensor(np.asarray(value), dtype=torch.float32, device=device)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t


def _unpack_batched(tensor_tuple) -> tuple[np.ndarray, np.ndarray]:
    """Pick (joint_rots [J,3,3], joint_coords [J,3]) out of mhr_forward's
    return tuple by tensor shape — same heuristic the reference uses."""
    rots = coords = None
    for t in tensor_tuple:
        if t is None:
            continue
        if hasattr(t, "ndim"):
            if t.ndim == 4 and t.shape[-1] == 3 and t.shape[-2] == 3:
                rots = t
            elif t.ndim == 3 and t.shape[-1] == 3 and t.shape[-2] != 3:
                coords = t
    r = rots.detach().cpu().numpy() if rots is not None else None
    c = coords.detach().cpu().numpy() if coords is not None else None
    if r is not None and r.ndim == 4:
        r = r[0]
    if c is not None and c.ndim == 3:
        c = c[0]
    return r, c


def _joint_parents_from_head(mhr_head) -> np.ndarray:
    """Extract joint parent indices from MHR. Mirrors the lookup used in
    the reference repo: scan ``mhr_head.mhr.named_buffers()`` for a buffer
    whose name contains ``joint_parents``."""
    try:
        bufs = dict(mhr_head.mhr.named_buffers())
        for k, v in bufs.items():
            if "joint_parents" in k.lower():
                return v.detach().cpu().numpy().astype(np.int32)
    except Exception as exc:
        log.warning("[pose] mhr.named_buffers lookup failed: %s", exc)
    # Last-ditch fallback: a few common attribute names.
    for attr in ("joint_parents", "parents", "kintree_table"):
        if hasattr(mhr_head, attr):
            v = getattr(mhr_head, attr)
            arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            if attr == "kintree_table" and arr.ndim == 2:
                return arr[0].astype(np.int32)
            return arr.astype(np.int32)
    raise AttributeError("MHR head exposes no joint_parents buffer")


def estimate_poses(
    image_rgb: np.ndarray,
    instances: list[Instance],
    *,
    bbox_threshold: float = 0.5,
    inference_type: str = "full",
    device_pref: str = "auto",
    hand_overrides: dict[int, dict] | None = None,
    pose_adjust: float = 0.0,
) -> list[PersonPose]:
    """Run pose inference per instance and assemble PersonPose records.

    Args:
        hand_overrides: optional ``{person_idx: override_dict}`` map. Each
            ``override_dict`` may contain any of:
                ``lhand_rgb`` — H×W×3 uint8 RGB array (left hand crop)
                ``rhand_rgb`` — H×W×3 uint8 RGB array (right hand crop)
                ``lhand_flip``, ``rhand_flip`` — bool, extra horizontal flip
                                                 applied before model input
            For matching person idx, the hand-only decoder is re-run on
            those crops and its output replaces the corresponding 54-dim
            slot of ``hand_pose_params``.
        pose_adjust: forward-lean correction strength in [0.0, 1.0].
            0.0 keeps the raw model output; 1.0 applies the full upstream
            preset (≈ 20° backwards bend at spine_01 plus small trims at
            neck_01 / head). Useful because SAM 3D Body tends to estimate
            standing subjects as slightly leaning forward — particularly
            for illustration / line-art inputs. Photographic inputs are
            usually fine at 0.0; illustrations often look right around 0.5.
    """
    if not instances:
        return []

    import cv2
    import torch

    loaded = load_model(device_pref)
    sam_3d_model = loaded["model"]
    model_cfg = loaded["model_cfg"]
    device = torch.device(loaded["device"])
    mhr_head = loaded["mhr_head"]

    # Estimator instance is cheap to construct; reuse across persons but
    # not across images (the reference reuses across frames in animation).
    from .vendor.sam_3d_body import SAM3DBodyEstimator

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=sam_3d_model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )

    if image_rgb.dtype != np.uint8:
        image_rgb = image_rgb.astype(np.uint8)
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Compute rest skeleton ONCE — shape/scale params are zeros.
    parents = _joint_parents_from_head(mhr_head).astype(np.int32)
    num_joints = parents.shape[0]
    names_full = [KNOWN_JOINT_NAMES.get(i, f"joint_{i:03d}") for i in range(num_joints)]

    shape_params = torch.zeros((1, mhr_head.num_shape_comps), dtype=torch.float32, device=device)
    scale_params = torch.zeros((1, mhr_head.num_scale_comps), dtype=torch.float32, device=device)
    expr_params = torch.zeros((1, mhr_head.num_face_comps), dtype=torch.float32, device=device)
    zeros3 = torch.zeros((1, 3), dtype=torch.float32, device=device)
    body_zero = torch.zeros((1, 133), dtype=torch.float32, device=device)
    hand_zero = torch.zeros((1, 108), dtype=torch.float32, device=device)
    global_trans = torch.zeros((1, 3), dtype=torch.float32, device=device)

    with torch.no_grad():
        rest_out = mhr_head.mhr_forward(
            global_trans=global_trans,
            global_rot=zeros3,
            body_pose_params=body_zero,
            hand_pose_params=hand_zero,
            scale_params=scale_params,
            shape_params=shape_params,
            expr_params=expr_params,
            return_joint_rotations=True,
            return_joint_coords=True,
        )
    rest_rots, rest_coords = _unpack_batched(rest_out[1:])
    rest_rots = rest_rots.astype(np.float32)
    rest_coords = rest_coords.astype(np.float32)

    # Run body inference for ALL instances in a single batched call.
    #
    # The previous version called process_one_image once per person. That
    # path takes the batch_size==1 branch in run_inference's keypoint-prompt
    # logic (sam3d_body.py around line 1448) — a different code path than
    # the multi-person branch — and the model also stashes per-call state
    # (`self.hand_batch_idx`, `self.body_batch_idx`, `self._max_num_person`,
    # `self._person_valid`) on the singleton. Calling N times in a row left
    # the previous person's state visible to the next, manifesting as the
    # observed swap when one person was skipped (the surviving 2 persons
    # then ran through a different prompt branch / state pair than when all
    # 3 were processed). Passing the full batch once routes through the
    # multi-person branch consistently and keeps person→output indexing
    # one-to-one.
    bboxes_all = np.stack([inst.bbox_xyxy for inst in instances]).astype(np.float32)
    masks_all = np.stack([inst.mask for inst in instances]).astype(np.uint8)
    try:
        outputs = estimator.process_one_image(
            img_bgr,
            bboxes=bboxes_all,
            masks=masks_all,
            bbox_thr=bbox_threshold,
            use_mask=True,
            inference_type=inference_type,
        )
    except Exception as exc:
        log.warning("[pose] batched estimation failed (%d persons): %s", len(instances), exc)
        return []

    if not outputs or len(outputs) != len(instances):
        log.warning(
            "[pose] estimator returned %d outputs for %d input persons",
            0 if not outputs else len(outputs), len(instances),
        )
        return []

    poses: list[PersonPose] = []
    for inst, raw in zip(instances, outputs):
        # Optional per-person hand override: re-run the hand decoder on
        # user-supplied crops and splice the result into hand_pose_params.
        hand_pose_arr = raw.get("hand_pose_params")
        ov = (hand_overrides or {}).get(inst.idx)
        if ov:
            from . import hand_inference

            lhand_params = None
            rhand_params = None
            l_rgb = ov.get("lhand_rgb")
            r_rgb = ov.get("rhand_rgb")
            if l_rgb is not None:
                l_uint8 = hand_inference.hand_rgb_to_uint8(l_rgb)
                if l_uint8 is not None:
                    try:
                        lhand_params = hand_inference.run_hand_only_inference(
                            estimator,
                            l_uint8,
                            is_left=True,
                            user_flip=bool(ov.get("lhand_flip", False)),
                        )
                    except Exception as exc:
                        log.warning("[pose] left hand override failed for #%d: %s", inst.idx, exc)
            if r_rgb is not None:
                r_uint8 = hand_inference.hand_rgb_to_uint8(r_rgb)
                if r_uint8 is not None:
                    try:
                        rhand_params = hand_inference.run_hand_only_inference(
                            estimator,
                            r_uint8,
                            is_left=False,
                            user_flip=bool(ov.get("rhand_flip", False)),
                        )
                    except Exception as exc:
                        log.warning("[pose] right hand override failed for #%d: %s", inst.idx, exc)
            if lhand_params is not None or rhand_params is not None:
                hand_pose_arr = hand_inference.splice_hand_into_params(
                    hand_pose_arr,
                    lhand_params=lhand_params,
                    rhand_params=rhand_params,
                )

        global_rot_t = _to_batched_tensor(raw.get("global_rot"), device, width=3)
        body_pose_t = _to_batched_tensor(raw.get("body_pose_params"), device, width=133)
        hand_pose_t = _to_batched_tensor(hand_pose_arr, device, width=108)

        with torch.no_grad():
            posed_out = mhr_head.mhr_forward(
                global_trans=global_trans,
                global_rot=global_rot_t,
                body_pose_params=body_pose_t,
                hand_pose_params=hand_pose_t,
                scale_params=scale_params,
                shape_params=shape_params,
                expr_params=expr_params,
                return_joint_rotations=True,
                return_joint_coords=True,
            )
        posed_rots, posed_coords = _unpack_batched(posed_out[1:])
        posed_rots = posed_rots.astype(np.float32)
        posed_coords = posed_coords.astype(np.float32)

        # Forward-lean correction (counter SAM 3D Body's slight forward
        # bias). No-op when pose_adjust is 0.
        if pose_adjust:
            posed_rots, posed_coords = apply_lean_correction(
                posed_rots, posed_coords, parents, pose_adjust,
            )

        poses.append(
            PersonPose(
                idx=inst.idx,
                bbox_xyxy=inst.bbox_xyxy.astype(np.float32),
                score=float(inst.score),
                joint_names=names_full,
                joint_parents=parents.tolist(),
                rest_joint_coords=rest_coords,
                rest_joint_rots=rest_rots,
                posed_joint_coords=posed_coords,
                posed_joint_rots=posed_rots,
            )
        )

    return poses
