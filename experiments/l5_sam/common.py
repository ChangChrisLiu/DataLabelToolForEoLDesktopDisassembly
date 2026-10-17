"""The evaluation protocol of L5: windows, prompts and the two quality numbers.

The window and the metrics are sam_compare's (:mod:`experiments.sam_compare.common`)
copied verbatim rather than imported, because that package's ``_env`` re-points
every cache variable on import.  The prompts are the app's
(``tda.ui.canvas.sam_tools``): the first prompt of a part is sent with
``multimask=True`` and the app shows the candidate SAM scores highest; a later
click refines the shown mask (``mask_input``) with ``multimask=False``.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

#: First prompts, gated.
GATED_PROMPTS = ("box", "point", "point_box")
#: Every prompt evaluated (``refine`` follows ``point_box``).
PROMPTS = ("box", "point", "point_box", "box_loose", "refine")

MIN_WINDOW = 512
MAX_WINDOW = 1024
WINDOW_CONTEXT = 1.4
BOUNDARY_TOL = 2
BOX_JITTER = 2


@dataclass(frozen=True)
class Window:
    x0: int
    y0: int
    x1: int
    y1: int
    scale: float

    @property
    def side(self) -> int:
        return self.x1 - self.x0

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1

    @property
    def oversize(self) -> bool:
        return self.scale < 1.0


def crop_window(bbox, frame_hw) -> Window:
    """sam_compare's production-like viewport around ``bbox`` (see its docstring)."""
    h, w = frame_hw
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    longest = max(x1 - x0, y1 - y0)
    side = int(round(longest * WINDOW_CONTEXT))
    scale = 1.0
    if side > MAX_WINDOW:
        side = min(int(round(longest * 1.1)), min(h, w))
        scale = MAX_WINDOW / float(side)
    side = max(MIN_WINDOW, side)
    side = min(side, min(h, w))
    wx0 = int(round(cx - side / 2.0))
    wy0 = int(round(cy - side / 2.0))
    wx0 = max(0, min(wx0, w - side))
    wy0 = max(0, min(wy0, h - side))
    return Window(wx0, wy0, wx0 + side, wy0 + side, scale)


def crop_image(frame: np.ndarray, win: Window) -> np.ndarray:
    patch = frame[win.y0:win.y1, win.x0:win.x1]
    if not win.oversize:
        return np.ascontiguousarray(patch)
    return np.ascontiguousarray(
        cv2.resize(patch, (MAX_WINDOW, MAX_WINDOW), interpolation=cv2.INTER_AREA))


def paste_mask(pred: np.ndarray, win: Window) -> np.ndarray:
    """Model-crop mask back to native window resolution (identity when scale=1)."""
    if pred.shape == (win.side, win.side):
        return pred
    return cv2.resize(pred.astype(np.uint8), (win.side, win.side),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def to_crop(xy, win: Window) -> tuple[float, float]:
    return (xy[0] - win.x0) * win.scale, (xy[1] - win.y0) * win.scale


def seed_of(desktop: int, step: int, instance: str) -> int:
    return zlib.crc32(f"{desktop}:{step}:{instance}".encode("utf-8"))


def jittered_box(bbox, seed: int, jitter: int = BOX_JITTER):
    """The GT bbox (exclusive x1/y1) with each edge moved by an integer U[-j, +j]."""
    rng = np.random.default_rng(seed)
    d = rng.integers(-jitter, jitter + 1, size=4)
    x0, y0, x1, y1 = (bbox[0] + d[0], bbox[1] + d[1], bbox[2] + d[2], bbox[3] + d[3])
    if x1 - x0 < 2:
        x0, x1 = bbox[0], bbox[2]
    if y1 - y0 < 2:
        y0, y1 = bbox[1], bbox[3]
    return float(x0), float(y0), float(x1), float(y1)


def loose_box(bbox):
    """The difference-map case: each edge pushed out by max(3, 15 % of the side)."""
    x0, y0, x1, y1 = bbox
    px = max(3.0, 0.15 * (x1 - x0))
    py = max(3.0, 0.15 * (y1 - y0))
    return float(x0 - px), float(y0 - py), float(x1 + px), float(y1 + py)


def box_in_crop(box, win: Window) -> tuple[float, float, float, float]:
    """A frame-coordinate box clipped to the window, in crop coordinates (as the app)."""
    bx0 = (min(max(box[0], win.x0), win.x1) - win.x0) * win.scale
    by0 = (min(max(box[1], win.y0), win.y1) - win.y0) * win.scale
    bx1 = (min(max(box[2], win.x0), win.x1) - win.x0) * win.scale
    by1 = (min(max(box[3], win.y0), win.y1) - win.y0) * win.scale
    return bx0, by0, bx1, by1


def corrective_click(pred: np.ndarray, gt: np.ndarray) -> Optional[tuple[int, int, int]]:
    """The standard click simulator: the deepest pixel of the larger error.

    ``(x, y, label)`` in the arrays' coordinates; label 1 in a miss (gt and not
    pred), 0 in an extra (pred and not gt); ``None`` when the masks agree.
    """
    best = None
    for err, label in ((gt & ~pred, 1), (pred & ~gt, 0)):
        if not err.any():
            continue
        ys, xs = np.nonzero(err)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = np.pad(err[y0:y1, x0:x1].astype(np.uint8), 1)
        dist = cv2.distanceTransform(sub, cv2.DIST_L2, 3)
        idx = int(np.argmax(dist))
        yy, xx = divmod(idx, dist.shape[1])
        d = float(dist[yy, xx])
        if best is None or d > best[0]:
            best = (d, int(x0 + xx - 1), int(y0 + yy - 1), label)
    if best is None:
        return None
    return best[1], best[2], best[3]


# --------------------------------------------------------------------------- #
# metrics (sam_compare.common, verbatim)
# --------------------------------------------------------------------------- #
def mask_bbox(mask: np.ndarray):
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _roi(a, b, pad):
    h, w = a.shape
    bb = mask_bbox(a | b)
    if bb is None:
        return slice(0, 0), slice(0, 0)
    x0, y0, x1, y1 = bb
    return (slice(max(0, y0 - pad), min(h, y1 + pad)), slice(max(0, x0 - pad), min(w, x1 + pad)))


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = int(np.count_nonzero(pred & gt))
    union = int(np.count_nonzero(pred | gt))
    return inter / union if union else 1.0


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    eroded = cv2.erode(m, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)
    return (m - eroded).astype(bool)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol: int = BOUNDARY_TOL) -> float:
    rows, cols = _roi(pred, gt, tol + 3)
    p, g = pred[rows, cols], gt[rows, cols]
    if p.size == 0:
        return 1.0
    pb, gb = _boundary(p), _boundary(g)
    n_p, n_g = int(pb.sum()), int(gb.sum())
    if n_p == 0 and n_g == 0:
        return 1.0
    if n_p == 0 or n_g == 0:
        return 0.0
    k = 2 * tol + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    gb_d = cv2.dilate(gb.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                      borderValue=0).astype(bool)
    pb_d = cv2.dilate(pb.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                      borderValue=0).astype(bool)
    precision = float((pb & gb_d).sum()) / n_p
    recall = float((gb & pb_d).sum()) / n_g
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
