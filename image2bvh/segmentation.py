"""SAM 3 keyword-based multi-instance segmentation.

Wraps `facebook/sam3` (text-prompt Promptable Concept Segmentation) so that
calling :func:`extract_instances` with an RGB image and a keyword (default
``"person"``) returns one entry per detected instance — each with a binary
mask, an xyxy bbox, a score, and an idx.

The model is loaded from the local snapshot under ``runtime/models/sam3``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import Iterable

import numpy as np
from PIL import Image

from . import paths

log = logging.getLogger(__name__)


@dataclass
class Instance:
    idx: int
    mask: np.ndarray  # uint8 [H, W], values 0/1
    bbox_xyxy: np.ndarray  # float32 [4]
    score: float


_MODEL_LOCK = Lock()
_MODEL = None
_PROCESSOR = None
_DEVICE: str | None = None


def _load(device_pref: str = "auto"):
    """Lazy-load Sam3Model + Sam3Processor from the local snapshot."""
    global _MODEL, _PROCESSOR, _DEVICE
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL, _PROCESSOR, _DEVICE

        import torch
        from transformers import Sam3Model, Sam3Processor

        if device_pref == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_pref

        local_dir = str(paths.SAM3_DIR)
        log.info("Loading SAM3 from %s on %s", local_dir, device)
        processor = Sam3Processor.from_pretrained(local_dir)
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = Sam3Model.from_pretrained(local_dir, torch_dtype=dtype)
        model.to(device)
        model.eval()

        _MODEL, _PROCESSOR, _DEVICE = model, processor, device
        return model, processor, device


def extract_instances(
    image: Image.Image | np.ndarray,
    keyword: str = "person",
    *,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_area_ratio: float = 0.005,
    device_pref: str = "auto",
) -> list[Instance]:
    """Run SAM3 with a text prompt and return per-instance masks.

    Args:
        image: PIL.Image or numpy RGB uint8 [H, W, 3].
        keyword: short noun phrase (e.g. ``"person"``, ``"person in red"``).
        threshold: instance presence score cutoff.
        mask_threshold: per-pixel mask binarization threshold.
        min_area_ratio: drop masks whose foreground area is < this fraction
            of the full image area (filters spurious tiny detections).
    """
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        pil = Image.fromarray(image)
    else:
        pil = image
    pil = pil.convert("RGB")

    import torch

    model, processor, device = _load(device_pref)

    inputs = processor(images=pil, text=keyword, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = inputs.get("original_sizes")
    if target_sizes is not None:
        target_sizes = target_sizes.tolist()
    else:
        # Fallback to PIL size (W, H) → (H, W) target size
        target_sizes = [(pil.size[1], pil.size[0])]

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=target_sizes,
    )[0]

    masks = results.get("masks")
    scores = results.get("scores")
    if masks is None or len(masks) == 0:
        return []

    masks_np = masks.detach().cpu().numpy().astype(np.uint8)
    scores_np = scores.detach().cpu().numpy().astype(np.float32) if scores is not None else None

    H, W = pil.size[1], pil.size[0]
    image_area = float(H * W)
    min_area = max(1.0, min_area_ratio * image_area)

    # Always derive the bbox from the predicted mask, never from SAM3's
    # pred_boxes head. The two are predicted by separate model heads and
    # their per-query outputs can drift apart spatially — when two persons
    # stand close together, query Q's mask might land on person A while
    # query Q's pred_boxes might land closer to person B's centre. Since
    # the mask is what we hand to SAM 3D Body for the per-person crop, we
    # need the bbox (used for left-to-right sorting and the UI overlay) to
    # come from the same source. Otherwise the sort order disagrees with
    # the pose-input order and person N's BVH ends up with person M's pose.
    candidates: list[Instance] = []
    for i in range(masks_np.shape[0]):
        m = masks_np[i]
        if m.ndim == 3:  # transformer outputs may include leading channel
            m = m[0]
        m = (m > 0).astype(np.uint8)
        area = int(m.sum())
        if area < min_area:
            continue
        bbox = _bbox_from_mask(m)
        score = float(scores_np[i]) if scores_np is not None and i < len(scores_np) else 0.0
        candidates.append(Instance(idx=-1, mask=m, bbox_xyxy=bbox, score=score))

    # Reading-order ordering: group instances into "rows" by vertical
    # bbox overlap, sort each row left-to-right by cx, then sort rows
    # top-to-bottom by min y. This handles two common arrangements:
    #
    #   - Group photo (everyone at roughly the same height): all bboxes
    #     overlap vertically → one row → pure left-to-right cx sort.
    #
    #   - Manga / comic panels (persons stacked vertically with little
    #     or no y overlap): each panel becomes its own row → upper
    #     person gets the smaller idx, matching how a human reads the
    #     page.
    #
    # The previous flat (cx, cy) sort failed the manga case: a person
    # higher up but with a larger cx ended up after a person lower down
    # with a smaller cx, so when the user skipped idx=0 they perceived
    # the surviving idx=1 / idx=2 as "swapped" relative to reading order.
    candidates = _sort_reading_order(candidates)
    for new_idx, inst in enumerate(candidates):
        inst.idx = new_idx
    return candidates


def _sort_reading_order(insts: list[Instance]) -> list[Instance]:
    """Return ``insts`` sorted by row (top-to-bottom) then column (left-to-right).

    Two bboxes belong to the same row iff their y-ranges overlap.
    Greedy single-pass clustering: scan instances by y_top ascending and
    extend the current row as long as the next bbox still overlaps the
    row's running y-extent. Otherwise start a new row.
    """
    if len(insts) <= 1:
        return list(insts)

    # Sort by y_top so the greedy row-merge sees bboxes from the top.
    by_top = sorted(insts, key=lambda i: float(i.bbox_xyxy[1]))

    rows: list[list[Instance]] = []
    row_extents: list[tuple[float, float]] = []  # (y_min, y_max) per row
    for inst in by_top:
        y1 = float(inst.bbox_xyxy[1])
        y2 = float(inst.bbox_xyxy[3])
        placed = False
        for r_idx, (rmin, rmax) in enumerate(row_extents):
            if y1 < rmax and y2 > rmin:
                rows[r_idx].append(inst)
                row_extents[r_idx] = (min(rmin, y1), max(rmax, y2))
                placed = True
                break
        if not placed:
            rows.append([inst])
            row_extents.append((y1, y2))

    # Sort each row left-to-right by cx.
    for row in rows:
        row.sort(key=lambda i: round((i.bbox_xyxy[0] + i.bbox_xyxy[2]) * 0.5, 1))
    # Sort rows top-to-bottom by their y_top.
    order = sorted(range(len(rows)), key=lambda k: row_extents[k][0])
    return [inst for k in order for inst in rows[k]]


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return np.array([0.0, 0.0, float(mask.shape[1]), float(mask.shape[0])], dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


# Per-instance colour palette. Reused by the WebUI to colour-match each
# person's row tag with the corresponding mask blob in the overlay.
INSTANCE_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 80, 80),
    (80, 200, 120),
    (80, 130, 255),
    (255, 200, 80),
    (200, 80, 255),
    (80, 220, 220),
    (255, 120, 200),
    (180, 220, 80),
)


def instance_color(idx: int) -> tuple[int, int, int]:
    """Return the (R, G, B) colour assigned to a given instance idx."""
    return INSTANCE_PALETTE[idx % len(INSTANCE_PALETTE)]


def overlay_preview(image: np.ndarray, instances: Iterable[Instance]) -> np.ndarray:
    """Build an RGB preview image with each instance drawn in a unique colour
    plus an idx label at the bbox top-left."""
    import cv2

    base = image.copy()
    if base.dtype != np.uint8:
        base = base.astype(np.uint8)
    overlay = base.copy()
    palette = np.array(INSTANCE_PALETTE, dtype=np.uint8)
    for inst in instances:
        colour = palette[inst.idx % len(palette)].tolist()
        m = inst.mask.astype(bool)
        overlay[m] = (0.5 * np.array(colour) + 0.5 * overlay[m]).astype(np.uint8)
        x1, y1, x2, y2 = inst.bbox_xyxy.astype(int).tolist()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            overlay,
            f"#{inst.idx} {inst.score:.2f}",
            (x1 + 4, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
            cv2.LINE_AA,
        )
    return overlay
