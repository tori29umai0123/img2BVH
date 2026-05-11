"""Hand-only inference helpers.

Port of the reference repo's ``editor_app/hand_inference.py``. The full
SAM 3D Body inference path runs both body and hand decoders. These helpers
let us re-run *only* the hand decoder on a user-supplied cropped hand
image and splice the resulting 54-dim params into a body's
``hand_pose_params`` (108-dim, 54 left + 54 right).

The model was trained on right-hand crops, so a left-hand crop is mirrored
horizontally before inference; the per-user "flip" toggle adds an extra
mirror on top (useful when an upload is already mirrored, e.g. selfie
camera, or when the user wants to re-purpose a right-hand photo as a
left-hand reference).
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def hand_rgb_to_uint8(image_like) -> np.ndarray | None:
    """Coerce PIL / numpy / torch hand image input into H×W×3 uint8 RGB.
    Returns ``None`` for absent / 1×1 placeholders / unrecognised input."""
    if image_like is None:
        return None

    arr = None
    try:
        import torch

        if isinstance(image_like, torch.Tensor):
            t = image_like
            if t.dim() == 4:
                t = t[0]
            arr = t.detach().cpu().numpy()
        elif isinstance(image_like, np.ndarray):
            arr = image_like[0] if image_like.ndim == 4 else image_like
        else:
            from PIL import Image as _PILImage
            if isinstance(image_like, _PILImage.Image):
                arr = np.asarray(image_like.convert("RGB"))
    except Exception as exc:
        log.warning("hand image normalize failed: %s", exc)
        return None

    if arr is None:
        return None
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        return None
    if arr.shape[0] < 4 or arr.shape[1] < 4:
        return None
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def run_hand_only_inference(
    estimator,
    hand_rgb_uint8: np.ndarray,
    *,
    is_left: bool,
    user_flip: bool = False,
) -> np.ndarray:
    """Run the hand decoder on a cropped hand image and return a 54-dim
    pose vector for the requested side.

    Args:
        estimator: a ``SAM3DBodyEstimator``
        hand_rgb_uint8: H×W×3 uint8 RGB array
        is_left: ``True`` for left hand. Adds an internal horizontal flip
            so the model sees a "right" hand (it was trained on right
            hands), then we read the right slot of its output and use
            that as the left-hand pose.
        user_flip: extra horizontal flip applied *before* the
            ``is_left`` mirror. Use when the upload is already mirrored
            relative to the natural orientation of that hand.
    """
    import torch

    from .vendor.sam_3d_body.data.utils.prepare_batch import prepare_batch
    from .vendor.sam_3d_body.utils import recursive_to

    img = hand_rgb_uint8
    if user_flip:
        img = np.ascontiguousarray(img[:, ::-1])
    if is_left:
        img = np.ascontiguousarray(img[:, ::-1])

    h, w = img.shape[:2]
    bbox = np.array([[0, 0, w, h]], dtype=np.float32)
    with torch.no_grad():
        batch = prepare_batch(img, estimator.transform_hand, bbox)
        batch = recursive_to(batch, estimator.device)
        estimator.model._initialize_batch(batch)
        pose_output = estimator.model.forward_step(batch, decoder_type="hand")
    hand = pose_output["mhr_hand"]["hand"]  # (B, 108) — 54 left + 54 right
    return hand[0, 54:].detach().cpu().numpy().astype(np.float32)


def splice_hand_into_params(
    hand_pose_params,
    *,
    lhand_params: np.ndarray | None = None,
    rhand_params: np.ndarray | None = None,
) -> np.ndarray:
    """Splice user-provided hand vectors into a (108,) hand_pose_params.

    Returns a fresh float32 array — never mutates the input.
    """
    if hand_pose_params is None:
        out = np.zeros((108,), dtype=np.float32)
    else:
        out = np.asarray(hand_pose_params, dtype=np.float32).reshape(-1).copy()
        if out.size != 108:
            fixed = np.zeros((108,), dtype=np.float32)
            fixed[: min(108, out.size)] = out[: min(108, out.size)]
            out = fixed
    if lhand_params is not None:
        out[:54] = np.asarray(lhand_params, dtype=np.float32).reshape(-1)[:54]
    if rhand_params is not None:
        out[54:] = np.asarray(rhand_params, dtype=np.float32).reshape(-1)[:54]
    return out
