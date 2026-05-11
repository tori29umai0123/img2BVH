"""Gradio WebUI entry point for the local image2BVH project.

Mirrors the ZeroGPU demo's two-stage layout (mask → person picker →
run → per-person sliders + 3D plots) but talks to the local model
cache under ``runtime/models/`` instead of HuggingFace Hub at request
time. Replaces the previous FastAPI + custom HTML/CSS/JS frontend.

Layout:

    [左] 入力画像 + 人物検出
    [右] 検出オーバーレイ + 推論する人物選択（色付き）
         + 実行ボタン + ステータス + BVH 一覧
         + 検出人数分の「前のめり補正スライダー + 3D プレビュー」

Pose inference runs on the GPU once per 実行 click; subsequent slider
movements only re-run the CPU-side lean correction + BVH writer, so
there is no GPU re-cost when iterating on the lean correction.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from . import bootstrap, bvh_writer, paths, pose, segmentation

log = logging.getLogger("image2bvh.gradio")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# UI translation table. The default language is Japanese; English is
# selectable from the language dropdown in the top-right corner. Keys are
# component IDs (matching the order they're emitted by ``_switch_lang``);
# format-able strings use ``{...}`` placeholders that are filled at the
# call site via ``T(key, lang, **kwargs)``.
# ---------------------------------------------------------------------------
TRANSLATIONS: dict[str, dict[str, str]] = {
    "ja": {
        "input_image": "入力画像",
        "detect_btn": "人物検出",
        "overlay": "検出オーバーレイ",
        "picker_header": "**推論する人物を選択**",
        "lhand_img": "左手画像 (任意)",
        "rhand_img": "右手画像 (任意)",
        "lhand_flip": "左手を反転",
        "rhand_flip": "右手を反転",
        "run_btn": "ポーズ生成",
        "files_out": "BVH（クリックで個別 DL）",
        "lean_label": "前のめり補正",
        "lean_info": "この値が出力 BVH の前のめり度に焼き込まれます (release で再生成)。",
        "lean_label_fmt": "person #{idx} — 前のめり補正",
    },
    "en": {
        "input_image": "Input image",
        "detect_btn": "Detect persons",
        "overlay": "Detection overlay",
        "picker_header": "**Persons to estimate**",
        "lhand_img": "Left hand image (optional)",
        "rhand_img": "Right hand image (optional)",
        "lhand_flip": "Flip left",
        "rhand_flip": "Flip right",
        "run_btn": "Generate pose",
        "files_out": "BVH (click to download)",
        "lean_label": "Forward-lean correction",
        "lean_info": "This value is baked into the exported BVH (regenerates on slider release).",
        "lean_label_fmt": "person #{idx} — Forward-lean correction",
    },
}
DEFAULT_LANG = "ja"


def T(key: str, lang: str | None = None, **kwargs) -> str:
    """Look up a translation by key; fall back to ``DEFAULT_LANG`` if the
    locale has no entry. ``kwargs`` are passed to ``str.format`` so
    callers can interpolate per-instance values."""
    table = TRANSLATIONS.get(lang or DEFAULT_LANG, TRANSLATIONS[DEFAULT_LANG])
    s = table.get(key, key)
    if kwargs:
        s = s.format(**kwargs)
    return s

# ---------------------------------------------------------------------------
# MHR rest skeleton cache. Computed once via the local SAM 3D Body model and
# saved to runtime/mhr_rest.json so subsequent launches load instantly
# without paying the model-load tax on the 3D preview.
# ---------------------------------------------------------------------------
_REST_CACHE_PATH = paths.RUNTIME_DIR / "mhr_rest.json"


def _bake_rest_skeleton() -> dict:
    """Compute MHR rest joint positions once and persist to JSON."""
    import torch  # noqa: PLC0415 — heavy import deferred to first launch

    # Make sure the SAM 3D Body weights are present so we can run the
    # head module's forward kinematics.
    if not paths.sam3dbody_ready():
        bootstrap.ensure_sam3dbody()

    loaded = pose.load_model("auto")
    mhr_head = loaded["mhr_head"]
    device = torch.device(loaded["device"])

    parents = pose._joint_parents_from_head(mhr_head).astype(np.int32)
    num_joints = parents.shape[0]
    names = [pose.KNOWN_JOINT_NAMES.get(i, f"joint_{i:03d}") for i in range(num_joints)]

    shape_p = torch.zeros((1, mhr_head.num_shape_comps), dtype=torch.float32, device=device)
    scale_p = torch.zeros((1, mhr_head.num_scale_comps), dtype=torch.float32, device=device)
    expr_p = torch.zeros((1, mhr_head.num_face_comps), dtype=torch.float32, device=device)
    zeros3 = torch.zeros((1, 3), dtype=torch.float32, device=device)
    body_zero = torch.zeros((1, 133), dtype=torch.float32, device=device)
    hand_zero = torch.zeros((1, 108), dtype=torch.float32, device=device)
    global_trans = torch.zeros((1, 3), dtype=torch.float32, device=device)

    with torch.no_grad():
        rest_out = mhr_head.mhr_forward(
            global_trans=global_trans, global_rot=zeros3,
            body_pose_params=body_zero, hand_pose_params=hand_zero,
            scale_params=scale_p, shape_params=shape_p,
            expr_params=expr_p,
            return_joint_rotations=True, return_joint_coords=True,
        )
    _, rest_coords = pose._unpack_batched(rest_out[1:])
    rest_coords = rest_coords.astype(np.float32)

    lean_chain = [
        [int(j), math.degrees(float(angle))]
        for j, angle in pose._LEAN_CHAIN_DEFAULT
    ]
    payload = {
        "joint_names": names,
        "joint_parents": parents.tolist(),
        "rest_coords": rest_coords.tolist(),
        "lean_chain": lean_chain,
    }
    _REST_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REST_CACHE_PATH.write_text(json.dumps(payload), encoding="utf-8")
    log.info("Baked MHR rest skeleton → %s", _REST_CACHE_PATH)
    return payload


def _load_or_bake_rest() -> dict:
    if _REST_CACHE_PATH.is_file():
        cached = json.loads(_REST_CACHE_PATH.read_text(encoding="utf-8"))
        # If the lean chain in source code drifted, force a re-bake so the
        # preview applies the same chain the BVH writer will.
        cached_chain = [tuple(x) for x in cached.get("lean_chain", [])]
        live_chain = [
            (int(j), round(math.degrees(float(angle)), 6))
            for j, angle in pose._LEAN_CHAIN_DEFAULT
        ]
        cached_chain_norm = [(int(j), round(float(deg), 6)) for j, deg in cached_chain]
        if cached_chain_norm == live_chain:
            return cached
        log.info("MHR rest cache lean_chain differs from source — re-baking")
    return _bake_rest_skeleton()


_REST = _load_or_bake_rest()
_REST_COORDS: list[list[float]] = _REST["rest_coords"]
_PARENTS: list[int] = _REST["joint_parents"]
_NAMES: list[str] = _REST["joint_names"]
_LEAN_CHAIN_RAD: list[tuple[int, float]] = [
    (int(j), math.radians(float(deg))) for j, deg in _REST["lean_chain"]
]
_NUM_JOINTS = len(_REST_COORDS)


def _name_to_idx(name: str) -> int:
    try:
        return _NAMES.index(name)
    except ValueError:
        return -1


_CLAVICLE_L_IDX = _name_to_idx("clavicle_l")
_CLAVICLE_R_IDX = _name_to_idx("clavicle_r")


# ---------------------------------------------------------------------------
# Lean correction (positions only, JS-equivalent of pose.apply_lean_correction)
# ---------------------------------------------------------------------------
def _subtree(parents: list[int], root: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for i, p in enumerate(parents):
        if p >= 0:
            children.setdefault(p, []).append(i)
    out: list[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(children.get(node, ()))
    return out


def _apply_lean(coords: list[list[float]], strength: float) -> list[list[float]]:
    out = [list(c) for c in coords]
    if not strength or strength <= 1e-6:
        return out
    if len(out) > 1:
        anchor_pelvis_xz = (out[1][0], out[1][2])
        anchor_min_y = min(c[1] for c in out)
    else:
        anchor_pelvis_xz = (0.0, 0.0)
        anchor_min_y = 0.0
    # Stage 1: spine chain (pelvis-anchored). Re-ground after so the legs
    # stay vertical and feet stay on the floor.
    for joint_id, base_rad in _LEAN_CHAIN_RAD:
        if joint_id >= len(out):
            continue
        theta = -float(strength) * float(base_rad)
        if abs(theta) < 1e-8:
            continue
        c, s = math.cos(theta), math.sin(theta)
        sub = _subtree(_PARENTS, joint_id)
        px, py, pz = out[joint_id]
        for k in sub:
            dx = out[k][0] - px
            dy = out[k][1] - py
            dz = out[k][2] - pz
            out[k] = [px + dx, py + (c * dy - s * dz), pz + (s * dy + c * dz)]
    # Counter-rotate the arms by the cumulative spine rotation so the
    # shoulder doesn't appear to elevate (mirrors pose.apply_lean_correction).
    if _CLAVICLE_R_IDX >= 0 and _CLAVICLE_R_IDX < len(out):
        arm_total_theta = 0.0
        for joint_id, base_rad in _LEAN_CHAIN_RAD:
            if joint_id >= len(out):
                continue
            if _CLAVICLE_R_IDX in _subtree(_PARENTS, joint_id):
                arm_total_theta += -float(strength) * float(base_rad)
        if abs(arm_total_theta) > 1e-8:
            cc, ss = math.cos(-arm_total_theta), math.sin(-arm_total_theta)
            for clav_idx in (_CLAVICLE_L_IDX, _CLAVICLE_R_IDX):
                if clav_idx < 0 or clav_idx >= len(out):
                    continue
                sub = _subtree(_PARENTS, clav_idx)
                px, py, pz = out[clav_idx]
                for k in sub:
                    dx = out[k][0] - px
                    dy = out[k][1] - py
                    dz = out[k][2] - pz
                    out[k] = [px + dx, py + (cc * dy - ss * dz), pz + (ss * dy + cc * dz)]
    if len(out) > 1:
        dx = anchor_pelvis_xz[0] - out[1][0]
        dz = anchor_pelvis_xz[1] - out[1][2]
        dy = anchor_min_y - min(c[1] for c in out)
        if abs(dx) > 1e-9 or abs(dy) > 1e-9 or abs(dz) > 1e-9:
            for k in range(len(out)):
                out[k][0] += dx
                out[k][1] += dy
                out[k][2] += dz
    return out


def _make_figure(coords: list[list[float]], title: str) -> go.Figure:
    line_x: list[float | None] = []
    line_y: list[float | None] = []
    line_z: list[float | None] = []
    for i, p in enumerate(_PARENTS):
        if p < 0:
            continue
        line_x += [coords[p][0], coords[i][0], None]
        line_y += [coords[p][1], coords[i][1], None]
        line_z += [coords[p][2], coords[i][2], None]
    fig = go.Figure(
        [
            go.Scatter3d(
                x=line_x, y=line_y, z=line_z, mode="lines",
                line=dict(color="#000000", width=3),
                name="bones", hoverinfo="skip",
            ),
            go.Scatter3d(
                x=[c[0] for c in coords],
                y=[c[1] for c in coords],
                z=[c[2] for c in coords],
                mode="markers",
                marker=dict(size=1, color="#ffffff", line=dict(color="#000000", width=1)),
                name="joints", hoverinfo="skip",
            ),
        ]
    )
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            bgcolor="#ffffff",
            camera=dict(eye=dict(x=0.0, y=1.0, z=2.5), up=dict(x=0, y=1, z=0)),
        ),
        paper_bgcolor="#ffffff",
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=0),
        title=dict(text=title, x=0.5, font=dict(size=11, color="#444444")),
        height=240,
        uirevision="preview",
    )
    return fig


def make_preview_for(pose_dict, pose_adjust):
    strength = float(pose_adjust or 0.0)
    if pose_dict:
        title = f"person #{pose_dict['idx']} — pose_adjust={strength:.2f}"
        base = pose_dict["joint_coords"]
    else:
        title = f"rest pose — pose_adjust={strength:.2f}"
        base = _REST_COORDS
    return _make_figure(_apply_lean(base, strength), title)


# ---------------------------------------------------------------------------
# Slot UI helpers (fixed N_SLOTS; visible/value updated per state change)
# ---------------------------------------------------------------------------
N_SLOTS = 8
_INITIAL_STRENGTH = 0.0


def _picker_updates(instances) -> list:
    """Return [grp, cb, md, lhand_img, rhand_img] × N_SLOTS for the
    picker section. Hand image fields are reset (cleared) on every
    mask regeneration so the override state doesn't bleed across
    uploads. Per-image flip is a button that mutates the image in
    place, so we don't carry a separate flip-flag state."""
    out: list = []
    n = len(instances) if instances else 0
    for i in range(N_SLOTS):
        if i < n:
            inst = instances[i]
            r, g, b = segmentation.INSTANCE_PALETTE[
                int(inst.idx) % len(segmentation.INSTANCE_PALETTE)
            ]
            md = (
                f'<span style="color:rgb({r},{g},{b}); font-weight:600">'
                f'person #{inst.idx}  (score {inst.score:.2f})</span>'
            )
            out.extend([
                gr.update(visible=True),
                gr.update(value=True),
                gr.update(value=md),
                gr.update(value=None),    # lhand_img
                gr.update(value=None),    # rhand_img
            ])
        else:
            out.extend([
                gr.update(visible=False),
                gr.update(value=False),
                gr.update(value=""),
                gr.update(value=None),
                gr.update(value=None),
            ])
    return out


def _flip_image_lr(img):
    """Mirror an uploaded hand crop horizontally. Bound to each picker
    slot's 反転 button so the user can fix orientation in place
    without an extra checkbox + flag state."""
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim < 2:
        return img
    return np.ascontiguousarray(arr[:, ::-1])


def _empty_person_updates(lang: str = DEFAULT_LANG) -> list:
    out: list = []
    for _ in range(N_SLOTS):
        out.extend([
            gr.update(visible=False),
            gr.update(value=_INITIAL_STRENGTH, label=T("lean_label", lang)),
            gr.update(value=None),
        ])
    return out


def _person_slot_updates(state_poses, strengths, lang: str = DEFAULT_LANG) -> list:
    out: list = []
    n = len(state_poses) if state_poses else 0
    for i in range(N_SLOTS):
        if i < n:
            person = state_poses[i]
            s = float((strengths or {}).get(int(person["idx"]), _INITIAL_STRENGTH))
            out.extend([
                gr.update(visible=True),
                gr.update(
                    value=s,
                    label=T("lean_label_fmt", lang, idx=person["idx"]),
                ),
                gr.update(value=make_preview_for(person, s)),
            ])
        else:
            out.extend([
                gr.update(visible=False),
                gr.update(value=_INITIAL_STRENGTH, label=T("lean_label", lang)),
                gr.update(value=None),
            ])
    return out


# ---------------------------------------------------------------------------
# BVH writers (re-used by initial run + slider release)
# ---------------------------------------------------------------------------
def _write_bvhs(poses_raw, strengths_by_idx, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths_out: list[str] = []
    for p in poses_raw:
        s = float(strengths_by_idx.get(int(p.idx), _INITIAL_STRENGTH))
        corrected = pose.apply_lean_correction_to_pose(p, s) if s > 1e-6 else p
        bvh_path = out_dir / f"person_{p.idx:02d}.bvh"
        bvh_path.write_text(
            bvh_writer.write_bvh(
                corrected.joint_names,
                corrected.joint_parents,
                corrected.posed_joint_coords,
                rest_joint_coords=corrected.rest_joint_coords,
                rest_joint_rots=corrected.rest_joint_rots,
                posed_joint_rots=corrected.posed_joint_rots,
            ),
            encoding="utf-8",
        )
        paths_out.append(str(bvh_path))
    return paths_out


def _new_output_dir() -> Path:
    """Per-request scratch directory under ``tmp/`` (wiped each request).

    Differs from the demo (which used ``/tmp`` ephemeral) — local builds
    write to the project's ``tmp/`` so users can also locate the files
    on disk if they prefer.
    """
    base = paths.reset_tmp()
    sub = base / "out"
    sub.mkdir(parents=True, exist_ok=True)
    return sub


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------
def step1_segment(image, lang: str = DEFAULT_LANG):
    if image is None:
        return (
            None, None, None, None, [],
            *_picker_updates([]),
            *_empty_person_updates(lang),
        )

    # Make sure SAM 3 weights are present (idempotent).
    if not paths.sam3_ready():
        bootstrap.ensure_sam3()

    rgb = np.asarray(image.convert("RGB"))
    instances = segmentation.extract_instances(rgb, "person", threshold=0.5)
    if not instances:
        return (
            None, None, None, None, [],
            *_picker_updates([]),
            *_empty_person_updates(lang),
        )

    overlay = segmentation.overlay_preview(rgb, instances)
    selected = [int(inst.idx) for inst in instances[:N_SLOTS]]
    return (
        overlay,
        instances,
        rgb,
        None,
        selected,
        *_picker_updates(instances[:N_SLOTS]),
        *_empty_person_updates(lang),
    )


def _toggle_picked_slot(slot_id: int, checked: bool, current_picked, instances):
    if not instances or slot_id >= len(instances):
        return current_picked or []
    cur = list(current_picked or [])
    pid = int(instances[slot_id].idx)
    if checked and pid not in cur:
        cur.append(pid)
    elif not checked and pid in cur:
        cur.remove(pid)
    return sorted(cur)


def step2_run(state_instances, state_rgb, picked, lang, *hand_inputs):
    """``hand_inputs`` is laid out as ``[*lhand_imgs, *rhand_imgs]`` —
    each of length ``N_SLOTS`` and ordered to match the picker slots.
    Slots whose person isn't selected (or whose image upload is empty)
    are skipped; the rest become entries in ``hand_overrides``. Each
    uploaded crop is used as-is — the per-slot 反転 button mutates
    the image in place when the user clicks it, so there's no
    separate flip flag to read here. ``lang`` (``ja``/``en``) drives
    the localized slider labels emitted by ``_person_slot_updates``."""
    def _err():
        return (
            None,
            None, None, None,
            *_empty_person_updates(lang),
        )
    if state_instances is None or state_rgb is None:
        return _err()
    if not picked:
        return _err()

    # Make sure SAM 3D Body weights are present (idempotent).
    if not paths.sam3dbody_ready():
        bootstrap.ensure_sam3dbody()

    keep_idx = {int(x) for x in picked}
    keep = [inst for inst in state_instances if int(inst.idx) in keep_idx]
    if not keep:
        return _err()

    n = N_SLOTS
    lhand_imgs = hand_inputs[0:n]
    rhand_imgs = hand_inputs[n:2 * n]
    hand_overrides: dict[int, dict] = {}
    for slot_id, inst in enumerate(state_instances[:n]):
        inst_idx = int(inst.idx)
        if inst_idx not in keep_idx:
            continue
        ov: dict = {}
        if lhand_imgs[slot_id] is not None:
            ov["lhand_rgb"] = lhand_imgs[slot_id]
        if rhand_imgs[slot_id] is not None:
            ov["rhand_rgb"] = rhand_imgs[slot_id]
        if ov:
            hand_overrides[inst_idx] = ov

    poses_raw = pose.estimate_poses(
        state_rgb, keep,
        pose_adjust=0.0,
        hand_overrides=hand_overrides or None,
    )
    if not poses_raw:
        return _err()

    strengths = {int(p.idx): _INITIAL_STRENGTH for p in poses_raw}
    out_dir = _new_output_dir()
    bvh_paths = _write_bvhs(poses_raw, strengths, out_dir)

    poses_for_preview = [
        {"idx": int(p.idx), "joint_coords": p.posed_joint_coords.astype(float).tolist()}
        for p in poses_raw
    ]
    return (
        bvh_paths,
        poses_for_preview, list(poses_raw), strengths,
        *_person_slot_updates(poses_for_preview[:N_SLOTS], strengths, lang),
    )


def regen_after_slider(person_idx, slider_value, strengths, poses_full):
    if not poses_full:
        return None, strengths or {}
    new_strengths = dict(strengths or {})
    new_strengths[int(person_idx)] = float(slider_value)
    out_dir = _new_output_dir()
    return _write_bvhs(poses_full, new_strengths, out_dir), new_strengths


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="image2BVH") as demo:
        state_instances = gr.State(value=None)
        state_rgb = gr.State(value=None)
        state_poses = gr.State(value=None)
        state_poses_full = gr.State(value=None)
        state_strengths = gr.State(value=None)
        state_picked = gr.State(value=[])
        state_lang = gr.State(value=DEFAULT_LANG)

        picker_grps: list = []
        picker_cbs: list = []
        picker_mds: list = []
        picker_lhand_imgs: list = []
        picker_lhand_flip_btns: list = []
        picker_rhand_imgs: list = []
        picker_rhand_flip_btns: list = []
        person_groups: list = []
        person_sliders: list = []
        person_plots: list = []

        # Top-right language picker. Right-aligned by giving it scale=0
        # in a row whose first column is a flexible spacer.
        with gr.Row():
            with gr.Column(scale=10):
                pass
            with gr.Column(scale=0, min_width=180):
                lang_dd = gr.Dropdown(
                    choices=[("日本語", "ja"), ("English", "en")],
                    value=DEFAULT_LANG,
                    show_label=False, container=False,
                    interactive=True,
                )

        with gr.Row():
            with gr.Column():
                image_in = gr.Image(
                    label=T("input_image"), type="pil", height=320,
                )
                segment_btn = gr.Button(T("detect_btn"), variant="secondary")

            with gr.Column():
                overlay_out = gr.Image(
                    label=T("overlay"), interactive=False, height=320,
                )

                picker_header_md = gr.Markdown(T("picker_header"))
                for i in range(N_SLOTS):
                    with gr.Group(visible=False) as picker_grp:
                        with gr.Row():
                            # Wrapping each child in a Column lets us
                            # set scale on the column itself: gr.Markdown
                            # in Gradio 6.x doesn't accept `scale=`,
                            # which is why putting scale=0 directly on
                            # the Checkbox alone isn't enough — the Row
                            # falls back to equal partition between
                            # siblings without explicit scales.
                            with gr.Column(scale=0, min_width=36):
                                cb = gr.Checkbox(
                                    value=False, container=False, show_label=False,
                                )
                            with gr.Column(scale=10):
                                md = gr.Markdown("", container=False)
                        # Optional per-person hand image overrides.
                        # Each takes precedence over the body-decoder's
                        # hand pose only if a non-empty image is
                        # uploaded; absent uploads fall back to the
                        # default body+hand inference. The 反転 button
                        # mirrors its image in place.
                        with gr.Row():
                            with gr.Column(scale=1):
                                lhand_img = gr.Image(
                                    label=T("lhand_img"),
                                    type="numpy",
                                    height=240,
                                    sources=["upload", "clipboard"],
                                )
                                lhand_flip_btn = gr.Button(
                                    T("lhand_flip"), size="sm",
                                )
                            with gr.Column(scale=1):
                                rhand_img = gr.Image(
                                    label=T("rhand_img"),
                                    type="numpy",
                                    height=240,
                                    sources=["upload", "clipboard"],
                                )
                                rhand_flip_btn = gr.Button(
                                    T("rhand_flip"), size="sm",
                                )
                    picker_grps.append(picker_grp)
                    picker_cbs.append(cb)
                    picker_mds.append(md)
                    picker_lhand_imgs.append(lhand_img)
                    picker_lhand_flip_btns.append(lhand_flip_btn)
                    picker_rhand_imgs.append(rhand_img)
                    picker_rhand_flip_btns.append(rhand_flip_btn)

                run_btn = gr.Button(T("run_btn"), variant="primary")
                files_out = gr.Files(label=T("files_out"), interactive=False)

                for i in range(N_SLOTS):
                    with gr.Group(visible=False) as group:
                        slider = gr.Slider(
                            0.0, 1.0, value=_INITIAL_STRENGTH, step=0.05,
                            label=T("lean_label"),
                            info=T("lean_info"),
                        )
                        plot = gr.Plot(value=None, show_label=False)
                    person_groups.append(group)
                    person_sliders.append(slider)
                    person_plots.append(plot)

        # --- wire pickers ---
        for slot_id in range(N_SLOTS):
            picker_cbs[slot_id].change(
                fn=lambda checked, current, instances, _slot=slot_id: _toggle_picked_slot(
                    _slot, checked, current, instances,
                ),
                inputs=[picker_cbs[slot_id], state_picked, state_instances],
                outputs=[state_picked],
            )

        # --- wire per-person sliders ---
        def _make_change(slot_id):
            def fn(slider_value, sp):
                if not sp or slot_id >= len(sp):
                    return None
                return make_preview_for(sp[slot_id], slider_value)
            return fn

        def _make_release(slot_id):
            def fn(slider_value, strengths, poses_full):
                if not poses_full or slot_id >= len(poses_full):
                    return None, strengths or {}
                return regen_after_slider(
                    int(poses_full[slot_id].idx),
                    slider_value, strengths, poses_full,
                )
            return fn

        for slot_id in range(N_SLOTS):
            person_sliders[slot_id].change(
                fn=_make_change(slot_id),
                inputs=[person_sliders[slot_id], state_poses],
                outputs=[person_plots[slot_id]],
            )
            person_sliders[slot_id].release(
                fn=_make_release(slot_id),
                inputs=[person_sliders[slot_id], state_strengths, state_poses_full],
                outputs=[files_out, state_strengths],
            )

        picker_components = [
            c for grp, cb, md, li, ri in zip(
                picker_grps, picker_cbs, picker_mds,
                picker_lhand_imgs, picker_rhand_imgs,
            )
            for c in (grp, cb, md, li, ri)
        ]
        person_components = [c for grp, sl, pl in zip(person_groups, person_sliders, person_plots)
                             for c in (grp, sl, pl)]

        # --- wire flip buttons (mirror the corresponding image in place) ---
        for slot_id in range(N_SLOTS):
            picker_lhand_flip_btns[slot_id].click(
                fn=_flip_image_lr,
                inputs=[picker_lhand_imgs[slot_id]],
                outputs=[picker_lhand_imgs[slot_id]],
            )
            picker_rhand_flip_btns[slot_id].click(
                fn=_flip_image_lr,
                inputs=[picker_rhand_imgs[slot_id]],
                outputs=[picker_rhand_imgs[slot_id]],
            )

        segment_btn.click(
            fn=step1_segment,
            inputs=[image_in, state_lang],
            outputs=[
                overlay_out, state_instances, state_rgb, state_poses,
                state_picked,
                *picker_components,
                *person_components,
            ],
        )

        run_btn.click(
            fn=step2_run,
            inputs=[
                state_instances, state_rgb, state_picked, state_lang,
                *picker_lhand_imgs, *picker_rhand_imgs,
            ],
            outputs=[
                files_out, state_poses, state_poses_full, state_strengths,
                *person_components,
            ],
        )

        # --- language switch: relocalize every static label + per-slot
        #     slider label (preserving current values / visibility) ---
        def _switch_lang(lang, sp, strengths):
            n = len(sp) if sp else 0
            slider_updates: list = []
            for i in range(N_SLOTS):
                if i < n:
                    person = sp[i]
                    slider_updates.append(gr.update(
                        label=T("lean_label_fmt", lang, idx=person["idx"]),
                        info=T("lean_info", lang),
                    ))
                else:
                    slider_updates.append(gr.update(
                        label=T("lean_label", lang),
                        info=T("lean_info", lang),
                    ))
            return [
                gr.update(label=T("input_image", lang)),     # image_in
                gr.update(value=T("detect_btn", lang)),      # segment_btn
                gr.update(label=T("overlay", lang)),         # overlay_out
                gr.update(value=T("picker_header", lang)),   # picker_header_md
                *[gr.update(label=T("lhand_img", lang)) for _ in range(N_SLOTS)],
                *[gr.update(value=T("lhand_flip", lang)) for _ in range(N_SLOTS)],
                *[gr.update(label=T("rhand_img", lang)) for _ in range(N_SLOTS)],
                *[gr.update(value=T("rhand_flip", lang)) for _ in range(N_SLOTS)],
                gr.update(value=T("run_btn", lang)),         # run_btn
                gr.update(label=T("files_out", lang)),       # files_out
                *slider_updates,                              # per-slot sliders
                lang,                                         # state_lang
            ]

        lang_dd.change(
            fn=_switch_lang,
            inputs=[lang_dd, state_poses, state_strengths],
            outputs=[
                image_in, segment_btn, overlay_out, picker_header_md,
                *picker_lhand_imgs, *picker_lhand_flip_btns,
                *picker_rhand_imgs, *picker_rhand_flip_btns,
                run_btn, files_out,
                *person_sliders,
                state_lang,
            ],
        )

    return demo


def _pick_free_port(host: str, start: int, tries: int = 50) -> int:
    """Walk forward from ``start`` until a TCP port is bind-able on ``host``.

    Lets a second instance launch on 7861 etc. instead of dying with
    ``OSError: Cannot find empty port`` when the previous run is still up.
    """
    import socket
    for offset in range(tries):
        port = start + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(f"No free port in {start}..{start + tries - 1} on {host}")


def main() -> None:
    """Launch the Gradio app on the first free port at/above 7860."""
    paths.ensure_dirs()
    demo = build_ui()
    port = _pick_free_port("127.0.0.1", 7860)
    demo.queue().launch(server_name="127.0.0.1", server_port=port, inbrowser=True)


if __name__ == "__main__":
    main()
