"""Pure-Python BVH writer.

Emits a single-frame BVH where:

- HIERARCHY OFFSETs are the **REST** parent→child diffs (anatomically
  natural rest skeleton).
- MOTION carries one frame with the root's world position + the root's
  rotation, and every other joint's local rotation as intrinsic XYZ
  Euler angles in degrees.

MHR rotation convention (verified empirically via the diagnostic in
``pose._check_mhr_convention``): ``posed_joint_rots[i]`` is the joint's
cumulative world rotation ``G_posed[i]`` and ``rest_joint_rots[i]`` is
the world rotation ``G_rest[i]`` that takes the joint's rest local
frame to world. Most joints in MHR have **bone-aligned** local frames,
so ``G_rest[i] != identity`` for limbs (only pelvis has identity rest).

Kinematic equation (verified): ``posed_pos[c] - posed_pos[p] =
G_posed[p] @ G_rest[p].T @ (rest_pos[c] - rest_pos[p])``.

Encoding strategy: define a per-joint BVH world rotation
``G_BVH[i] = G_posed[i] @ G_rest[i].T``. Then BVH evaluation
``BVH_pos[c] = BVH_pos[p] + G_BVH[p] @ offset[c]`` reproduces the
MHR posed position when ``offset[c] = rest_pos[c] - rest_pos[p]``
(world rest diff). At rest pose ``G_posed = G_rest``, so
``G_BVH = identity`` and all MOTION rotations are zero — the BVH
displays rest pose as expected.

Coordinate convention: MHR's coords are written through to BVH as-is
(X right, Y up, Z = character forward).
"""
from __future__ import annotations

import math
from io import StringIO

import numpy as np

# Subset of MHR bone names → standard humanoid names.
#
# We deliberately diverge from the reference Blender exporter on the
# spine: that exporter mapped pelvis/joint_034/spine_01/spine_02 to
# Hips/Spine/Chest/UpperChest and dropped spine_03 entirely. With
# spine_03 missing, MHR's clavicles (which are children of spine_03)
# are reparented to UpperChest (= spine_02), so the BVH "Shoulder" bone
# spans both spine_03 and the actual clavicle in a single OFFSET. CSP
# then renders that combined OFFSET as a long, forward-leaning
# shoulder bone with zero rotation — the upper arm comes out from a
# non-rotated "shoulder", which made the armpit collapse against the
# torso and the shoulder itself look stuck.
#
# Fix: keep the standard 4-bone humanoid spine (Hips / Spine / Chest /
# UpperChest) but shift the mapping up by one so spine_03 occupies
# UpperChest. joint_034 is the loser: it merges into the pelvis→Spine
# OFFSET (~13 cm, anatomically reasonable for a single lower-spine
# bone). Now the clavicles attach directly to UpperChest in the BVH,
# matching MHR's actual parentage, and the shoulder OFFSET is just one
# bone long.
HUMANOID_MAP: dict[str, str] = {
    "pelvis": "Hips",
    "spine_01": "Spine",
    "spine_02": "Chest",
    "spine_03": "UpperChest",
    "neck_01": "Neck",
    "head": "Head",
    "clavicle_l": "LeftShoulder",
    "upperarm_l": "LeftUpperArm",
    "lowerarm_l": "LeftLowerArm",
    "hand_l": "LeftHand",
    "clavicle_r": "RightShoulder",
    "upperarm_r": "RightUpperArm",
    "lowerarm_r": "RightLowerArm",
    "hand_r": "RightHand",
    "thigh_l": "LeftUpperLeg",
    "calf_l": "LeftLowerLeg",
    "foot_l": "LeftFoot",
    "thigh_r": "RightUpperLeg",
    "calf_r": "RightLowerLeg",
    "foot_r": "RightFoot",
    # MHR / MANO model the thumb with FIVE skeletal joints
    # (hand_r -> 060 -> 061 -> 062 -> 063 -> 064) because the
    # carpometacarpal joint articulates for opposability — other fingers
    # only have FOUR joints in their chain. Mapping the humanoid
    # Proximal/Intermediate/Distal labels to joint_060/061/062 covers only
    # the carpal/metacarpal base and clips the actual proximal phalanx, IP
    # and distal phalanx (joints 062..064 / 098..100), so the BVH thumb
    # ends up as a stub whose orientation never reflects user input
    # (it sits roughly in the rest "thumbs-up" direction regardless).
    # Shift labels by one joint so Prox/Int/Dist cover the real phalanx
    # bones; joint_060 / joint_096 are dropped from the humanoid subset
    # but their offset is preserved in hand_r -> joint_061 via reparenting.
    "joint_061": "RightThumbProximal",
    "joint_062": "RightThumbIntermediate",
    "joint_063": "RightThumbDistal",
    "joint_056": "RightIndexProximal",
    "joint_057": "RightIndexIntermediate",
    "joint_058": "RightIndexDistal",
    "joint_052": "RightMiddleProximal",
    "joint_053": "RightMiddleIntermediate",
    "joint_054": "RightMiddleDistal",
    "joint_048": "RightRingProximal",
    "joint_049": "RightRingIntermediate",
    "joint_050": "RightRingDistal",
    "joint_044": "RightLittleProximal",
    "joint_045": "RightLittleIntermediate",
    "joint_046": "RightLittleDistal",
    "joint_097": "LeftThumbProximal",
    "joint_098": "LeftThumbIntermediate",
    "joint_099": "LeftThumbDistal",
    "joint_092": "LeftIndexProximal",
    "joint_093": "LeftIndexIntermediate",
    "joint_094": "LeftIndexDistal",
    "joint_088": "LeftMiddleProximal",
    "joint_089": "LeftMiddleIntermediate",
    "joint_090": "LeftMiddleDistal",
    "joint_084": "LeftRingProximal",
    "joint_085": "LeftRingIntermediate",
    "joint_086": "LeftRingDistal",
    "joint_080": "LeftLittleProximal",
    "joint_081": "LeftLittleIntermediate",
    "joint_082": "LeftLittleDistal",
}

_DEFAULT_LEAF = 0.03


def _subset_humanoid(
    names: list[str],
    parents: list[int],
    posed_coords: np.ndarray,
    rest_coords: np.ndarray,
    posed_rots: np.ndarray,
    rest_rots: np.ndarray,
) -> tuple[
    list[str], list[int],
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    dict[int, np.ndarray],
]:
    """Subset to humanoid joints, re-index parents, slice all per-joint
    arrays in lock-step.

    Returns ``(new_names, new_parents, new_posed_coords, new_rest_coords,
    new_posed_rots, new_rest_rots, dropped_tip_offset_local)``.
    ``dropped_tip_offset_local`` maps a kept joint's NEW index to the
    End Site OFFSET (in that joint's rest local frame) derived from
    pruned descendants — used by the End Site emitter so leaf joints
    (e.g. foot) whose anatomical tip lives further down the chain
    still display the correct bone direction.
    """
    kept = [i for i, n in enumerate(names) if n in HUMANOID_MAP]
    if not kept:
        raise RuntimeError("No humanoid-compatible joints in pose package.")

    old_to_new = {old: new for new, old in enumerate(kept)}
    new_parents: list[int] = []
    for old_i in kept:
        p = parents[old_i]
        while p >= 0 and p not in old_to_new:
            p = parents[p]
        new_parents.append(old_to_new[p] if p >= 0 else -1)

    new_names = [HUMANOID_MAP[names[i]] for i in kept]

    # Only use a deeper-descendant world position as the End Site
    # target for joints whose anatomical tip really IS a forward
    # extension (toes -> toe tip). Joints like Head have many
    # face-joint children in MHR (eye/ear/jaw); the "first dropped
    # child" there is arbitrary and points sideways, breaking the
    # head bone direction.
    _TIP_TARGET_HUMANOID_NAMES = {"LeftFoot", "RightFoot"}
    orig_children: dict[int, list[int]] = {}
    for j, p in enumerate(parents):
        if p >= 0:
            orig_children.setdefault(p, []).append(j)
    dropped_tip_offset_local: dict[int, np.ndarray] = {}
    kept_set = set(kept)
    for new_idx, old_idx in enumerate(kept):
        if HUMANOID_MAP[names[old_idx]] not in _TIP_TARGET_HUMANOID_NAMES:
            continue
        # Walk down through dropped descendants until either we run
        # out, OR the next hop would be a true leaf. MHR's foot chain
        # ends at a 5th joint (e.g. joint_008) that sits past the toe
        # tip — likely a ground-contact reference, not anatomy.
        # Stopping at the second-to-last joint puts us at the real toe
        # tip (joint_007 / joint_023).
        cur = old_idx
        while True:
            non_kept_kids = [
                k for k in orig_children.get(cur, []) if k not in kept_set
            ]
            if not non_kept_kids:
                break
            next_cur = non_kept_kids[0]
            if not orig_children.get(next_cur):
                break
            cur = next_cur
        if cur != old_idx:
            # End Site OFFSET = the posed world diff (toe tip - ankle)
            # converted into foot_l's *posed* local frame. This is
            # POSE-DEPENDENT, which is unusual for BVH OFFSETs (they
            # are normally rest geometry), but it works for our
            # single-frame BVH and avoids touching foot_l's MOTION
            # rotation. With this OFFSET, BVH evaluation gives:
            #   End Site world = posed_J4 + G_posed[foot_l] @ OFFSET
            #     = posed_J4 + G_posed[foot_l] @ G_posed[foot_l].T @ (posed_J7 - posed_J4)
            #     = posed_J7
            # — the actual toe tip in the input pose. Since this only
            # touches OFFSETs (not rotations), the writer's existing
            # G_BVH = G_posed convention stays intact for foot_l, and
            # both Blender and Clip Studio Paint compute the same
            # posed bone direction.
            posed_diff = posed_coords[cur] - posed_coords[old_idx]
            dropped_tip_offset_local[new_idx] = (
                posed_rots[old_idx].T @ posed_diff
            ).astype(np.float64, copy=True)

    return (
        new_names,
        new_parents,
        posed_coords[kept].astype(np.float64, copy=True),
        rest_coords[kept].astype(np.float64, copy=True),
        posed_rots[kept].astype(np.float64, copy=True),
        rest_rots[kept].astype(np.float64, copy=True),
        dropped_tip_offset_local,
    )


def _matrix_to_euler_xyz_deg(R: np.ndarray) -> tuple[float, float, float]:
    """Extract intrinsic XYZ Euler angles in degrees from a 3x3 rotation
    matrix. Convention matches BVH ``CHANNELS Xrotation Yrotation
    Zrotation``: ``R = Rx(α) @ Ry(β) @ Rz(γ)`` for column vectors.
    """
    sb = float(R[0, 2])
    sb = max(-1.0, min(1.0, sb))
    if abs(sb) < 0.99999:
        beta = math.asin(sb)
        alpha = math.atan2(-float(R[1, 2]), float(R[2, 2]))
        gamma = math.atan2(-float(R[0, 1]), float(R[0, 0]))
    else:
        beta = math.copysign(math.pi / 2.0, sb)
        alpha = math.atan2(float(R[2, 1]), float(R[1, 1]))
        gamma = 0.0
    return math.degrees(alpha), math.degrees(beta), math.degrees(gamma)


def _rotation_power(R: np.ndarray, alpha: float) -> np.ndarray:
    """Return ``R**alpha`` for a 3x3 rotation matrix via axis-angle scaling.
    ``alpha == 0`` -> identity, ``alpha == 1`` -> R, with continuous
    interpolation along the same rotation axis in between.
    """
    cos_a = max(-1.0, min(1.0, 0.5 * (float(np.trace(R)) - 1.0)))
    angle = math.acos(cos_a)
    if angle < 1e-9:
        return np.eye(3, dtype=np.float64)
    sin_a = math.sin(angle)
    if sin_a < 1e-9:
        # angle ≈ π: derive axis from (R + I)/2 column with largest norm
        M = (R + np.eye(3, dtype=np.float64)) * 0.5
        col = max(range(3), key=lambda j: float(np.linalg.norm(M[:, j])))
        v = M[:, col]
        nv = float(np.linalg.norm(v))
        axis = v / nv if nv > 1e-9 else np.array([1.0, 0.0, 0.0])
    else:
        axis = np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=np.float64,
        ) / (2.0 * sin_a)
    new_angle = angle * alpha
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(new_angle) * K
        + (1.0 - math.cos(new_angle)) * (K @ K)
    )


# Scapulohumeral rhythm: when the upper arm rotates relative to the
# clavicle, anatomically about 1/3 of that motion belongs to the
# clavicle (scapulothoracic) and 2/3 to the glenohumeral joint. Pose
# estimators tend to put 100% of the motion into the upper arm, leaving
# the clavicle pinned in its rest direction. This constant transfers
# `alpha` of the upper arm's local rotation to the shoulder, in the
# same rotation axis, preserving every downstream world orientation.
_SCAPULAR_RHYTHM_ALPHA = 1.0 / 3.0


def write_bvh(
    joint_names: list[str],
    joint_parents: list[int],
    posed_joint_coords: np.ndarray,
    *,
    rest_joint_coords: np.ndarray,
    rest_joint_rots: np.ndarray,
    posed_joint_rots: np.ndarray,
    frame_time: float = 1.0 / 30.0,
) -> str:
    """Return a one-frame BVH that encodes the rest skeleton in
    HIERARCHY and the joint rotations in MOTION.

    Args:
        joint_names: per-joint name list (MHR convention).
        joint_parents: per-joint parent index (-1 for the root).
        posed_joint_coords: ``[J, 3]`` posed world positions, used for
            the root MOTION position channels.
        rest_joint_coords: ``[J, 3]`` rest world positions, used for
            HIERARCHY parent→child OFFSETs.
        rest_joint_rots: ``[J, 3, 3]`` rest world rotation per joint
            (``G_rest[i]`` in the convention check). Limbs in MHR have
            bone-aligned local frames so this is non-identity.
        posed_joint_rots: ``[J, 3, 3]`` posed world rotation per joint
            (``G_posed[i]``).
        frame_time: BVH frame time in seconds (1/30 by default).
    """
    # MHR's foot chain is foot_l -> joint_005 -> joint_006 -> joint_007.
    # foot_l itself only carries small ankle articulation; the actual
    # foot direction (ankle plantarflexion + toe extension) lives at
    # the ball joint (joint_006 / joint_022). Clip Studio Paint
    # ignores any toe-area joint we expose (LeftToes, LeftToeBase,
    # lefttoebase_bb_ — none survive its retargeter), so the foot
    # mesh in CSP only ever responds to LeftFoot's MOTION rotation.
    # Lift the ball joint's posed world rotation onto foot_l so the
    # BVH MOTION at LeftFoot carries the entire foot direction.
    posed_joint_rots = posed_joint_rots.astype(np.float64, copy=True)
    for foot_name in ("foot_l", "foot_r"):
        if foot_name not in joint_names:
            continue
        foot_idx = joint_names.index(foot_name)
        heel_anchor = next(
            (i for i, p in enumerate(joint_parents) if p == foot_idx), -1,
        )
        ball = next(
            (i for i, p in enumerate(joint_parents) if p == heel_anchor), -1,
        ) if heel_anchor >= 0 else -1
        if ball >= 0:
            posed_joint_rots[foot_idx] = posed_joint_rots[ball]

    (
        names, parents, p_coords, r_coords, p_rots, r_rots,
        dropped_tip_offset_local,
    ) = _subset_humanoid(
        joint_names, joint_parents,
        posed_joint_coords, rest_joint_coords,
        posed_joint_rots, rest_joint_rots,
    )
    n = len(names)

    children: dict[int, list[int]] = {}
    for i, p in enumerate(parents):
        if p >= 0:
            children.setdefault(p, []).append(i)

    roots = [i for i, p in enumerate(parents) if p < 0]
    if len(roots) != 1:
        raise RuntimeError(f"Expected 1 root joint, got {len(roots)}: {[names[i] for i in roots]}")
    root = roots[0]
    root_pos = p_coords[root].copy()

    # Use the joint's posed world rotation directly as the BVH world
    # rotation: G_BVH[i] = G_posed[i]. This preserves MHR's bone-aligned
    # local frame in BVH (an alternative would be G_posed[i] @ G_rest[i].T
    # to make the BVH rest pose world-aligned, but that produces an
    # unnatural bone roll downstream — CSP expects each joint's frame to
    # match the rest skeleton's local axes).
    G_BVH = p_rots  # already a copy from _subset_humanoid

    # Scapulohumeral rhythm correction: transfer ALPHA fraction of each
    # upper arm's local rotation onto its parent shoulder. The upper
    # arm's posed *world* rotation is left untouched, so the local
    # rotation re-derived later as G_BVH[shoulder_new].T @ G_BVH[arm]
    # automatically becomes the residual (1-ALPHA) share. Net result:
    # the shoulder visibly elevates / protracts with arm motion (the
    # clavicle no longer looks pinned), without changing the world pose
    # of the arm, hand, or fingers.
    for shoulder_name, arm_name in (
        ("LeftShoulder", "LeftUpperArm"),
        ("RightShoulder", "RightUpperArm"),
    ):
        if shoulder_name not in names or arm_name not in names:
            continue
        si = names.index(shoulder_name)
        ai = names.index(arm_name)
        if parents[ai] != si:
            continue
        R_arm_local = G_BVH[si].T @ G_BVH[ai]
        G_BVH[si] = G_BVH[si] @ _rotation_power(R_arm_local, _SCAPULAR_RHYTHM_ALPHA)

    # Pre-bake the root world rotation into root-child HIERARCHY OFFSETs
    # so the BVH's ROOT MOTION rotation channel becomes identity. Clip
    # Studio Paint's "Reset Model Rotation" zeroes the ROOT rotation;
    # without this bake, that operation strips the inferred pelvis tilt
    # and visibly tips the body. With the bake the loaded pose IS the
    # post-reset state, so reset is a no-op.
    #
    # Math: BVH evaluates world_pos[c] = world_pos[p] + world_rot[p] @
    # offset[c]. For root's children we currently rely on world_rot[root]
    # = R_p (pelvis world rotation) so that root_pos + R_p @ rest_diff
    # = posed_pos[c]. Setting world_rot[root] = identity means we have
    # to push R_p into offset[c] instead: new offset = R_p @ original
    # offset. Grandchildren and below are unaffected — their world
    # rotations and OFFSETs are unchanged because root's children's
    # local rotation now equals their full G_posed (no parent-relative
    # subtraction since the new parent rotation is identity), so the
    # chain G_posed[parent] propagates downstream normally.
    R_root_orig = G_BVH[root].copy()
    G_BVH[root] = np.eye(3, dtype=np.float64)
    root_child_offset_premul: dict[int, np.ndarray] = {
        c: R_root_orig for c in children.get(root, [])
    }

    local_rots = np.empty((n, 3, 3), dtype=np.float64)
    for i in range(n):
        if parents[i] < 0:
            local_rots[i] = G_BVH[i]
        else:
            local_rots[i] = G_BVH[parents[i]].T @ G_BVH[i]

    eulers_deg = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        eulers_deg[i] = _matrix_to_euler_xyz_deg(local_rots[i])

    buf = StringIO()
    buf.write("HIERARCHY\n")
    joint_order: list[int] = []

    def _write_joint(idx: int, depth: int, is_root: bool) -> None:
        joint_order.append(idx)
        indent = "  " * depth
        if is_root:
            buf.write(f"{indent}ROOT {names[idx]}\n")
        else:
            buf.write(f"{indent}JOINT {names[idx]}\n")
        buf.write(f"{indent}{{\n")
        if is_root:
            # Root OFFSET stays zero; root world position goes into
            # the per-frame MOTION position channels.
            buf.write(f"{indent}  OFFSET 0.000000 0.000000 0.000000\n")
            buf.write(
                f"{indent}  CHANNELS 6 Xposition Yposition Zposition "
                f"Xrotation Yrotation Zrotation\n"
            )
        else:
            # OFFSET in PARENT's rest local frame (= G_rest[parent].T
            # @ rest world diff). Combined with G_BVH[parent] =
            # G_posed[parent], BVH evaluation reproduces the SMPL-X
            # kinematic equation posed_diff = G_posed[p] @ G_rest[p].T
            # @ rest_world_diff (verified empirically).
            world_diff = r_coords[idx] - r_coords[parents[idx]]
            offset = r_rots[parents[idx]].T @ world_diff
            if idx in root_child_offset_premul:
                # Root's children: their parent's world rotation is now
                # identity (root rotation was baked out), so the OFFSET
                # carries the full posed bone direction directly.
                offset = root_child_offset_premul[idx] @ offset
            buf.write(
                f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n"
            )
            buf.write(f"{indent}  CHANNELS 3 Xrotation Yrotation Zrotation\n")

        kids = children.get(idx, [])
        if kids:
            for c in kids:
                _write_joint(c, depth + 1, is_root=False)
        else:
            if idx in dropped_tip_offset_local:
                # Pre-computed End Site OFFSET in idx's rest local
                # frame — the *bone vector* derived from the deepest
                # meaningful hop in the pruned chain (e.g. ball ->
                # toe tip for foot, length ~6 cm).
                tip = dropped_tip_offset_local[idx]
            elif parents[idx] >= 0:
                # Fallback: continue this joint's bone direction (parent
                # -> idx) for a default 3 cm leaf.
                world_axis = r_coords[idx] - r_coords[parents[idx]]
                length = float(np.linalg.norm(world_axis))
                if length > 1e-6:
                    axis = r_rots[parents[idx]].T @ (world_axis / length)
                    tip = axis * _DEFAULT_LEAF
                else:
                    tip = np.array([0.0, _DEFAULT_LEAF, 0.0])
            else:
                tip = np.array([0.0, _DEFAULT_LEAF, 0.0])
            buf.write(f"{indent}  End Site\n")
            buf.write(f"{indent}  {{\n")
            buf.write(
                f"{indent}    OFFSET {tip[0]:.6f} {tip[1]:.6f} {tip[2]:.6f}\n"
            )
            buf.write(f"{indent}  }}\n")
        buf.write(f"{indent}}}\n")

    _write_joint(root, depth=0, is_root=True)

    buf.write("MOTION\n")
    buf.write("Frames: 1\n")
    buf.write(f"Frame Time: {frame_time:.7f}\n")

    motion: list[str] = []
    for i in joint_order:
        rx_v, ry_v, rz_v = eulers_deg[i]
        if i == root:
            motion.extend([
                f"{root_pos[0]:.6f}", f"{root_pos[1]:.6f}", f"{root_pos[2]:.6f}",
                f"{rx_v:.6f}", f"{ry_v:.6f}", f"{rz_v:.6f}",
            ])
        else:
            motion.extend([f"{rx_v:.6f}", f"{ry_v:.6f}", f"{rz_v:.6f}"])
    buf.write(" ".join(motion) + "\n")

    return buf.getvalue()
