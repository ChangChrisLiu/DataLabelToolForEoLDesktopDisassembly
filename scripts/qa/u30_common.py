"""U30-QA: what the extraction, the analysis, the sheets and the SAM triage share.

Read-only helpers: load one desktop's extraction (``u30_extract.py``), the
scanner frame the annotator saw (the local cache's chosen burst image), and a
change map between two frames of the fixed scanner camera.
"""
from __future__ import annotations

import os
import pickle
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

from tda.core import masks as M

QA_ROOT = os.environ.get("U30QA_ROOT", r"D:\DataSet\.cache\tmp\u30qa")
CACHE_DIR = os.environ.get("U30QA_CACHE", r"D:\DataSet\cache")
DESKTOPS = list(range(2, 18)) + list(range(19, 33))
#: after the 10th/90th-percentile brightness match, a 5x5-blurred |difference|
#: above this many grey levels is a change (scripts/step_changes.py's rule)
CHANGE_GRAY = 25.0
FLOOR = ("chassis", "motherboard")


def load(desktop: int) -> dict:
    with open(os.path.join(QA_ROOT, "data", f"D{desktop:02d}.pkl"), "rb") as fh:
        return pickle.load(fh)


def image_path(res: dict, step: int) -> str:
    fr = res["frames"].get(step) or {}
    path = fr.get("cache_path")
    if path and os.path.isfile(path):
        return path
    return os.path.join(CACHE_DIR, "scan", f"D{res['desktop']:02d}", f"s{step:03d}.png")


@lru_cache(maxsize=8)
def _read(path: str, gray: bool) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    return img


def gray(res: dict, step: int) -> Optional[np.ndarray]:
    return _read(image_path(res, step), True)


def color(res: dict, step: int) -> Optional[np.ndarray]:
    """BGR, as OpenCV reads it."""
    return _read(image_path(res, step), False)


def decode(rle) -> Optional[np.ndarray]:
    if not rle:
        return None
    return M.decode_rle(rle)


def _region(roi, shape):
    if roi:
        x0, y0, x1, y1 = (int(v) for v in roi)
        return max(0, y0), min(shape[0], y1), max(0, x0), min(shape[1], x1)
    return 0, shape[0], 0, shape[1]


def matched(x: np.ndarray, ref: np.ndarray, roi) -> np.ndarray:
    """``x`` mapped so its ROI's 10th/90th percentiles are ``ref``'s."""
    y0, y1, x0, x1 = _region(roi, x.shape)
    x10, x90 = np.percentile(x[y0:y1, x0:x1], (10, 90))
    r10, r90 = np.percentile(ref[y0:y1, x0:x1], (10, 90))
    gain = (r90 - r10) / max(float(x90 - x10), 1.0)
    return (x - x10) * gain + r10


#: log-ratio change threshold after the illumination field is removed (~25 %)
CHANGE_LOG = 0.22
#: ... and a change must also be this many grey levels (dark noise)
CHANGE_ABS = 18.0


def _poly_terms(xx, yy, deg=3):
    return np.stack([xx ** i * yy ** j for i in range(deg + 1) for j in range(deg + 1 - i)],
                    axis=-1)


def illumination_ratio(fa: np.ndarray, fb: np.ndarray, roi) -> np.ndarray:
    """``log(b+10) - log(a+10)`` with a smooth (cubic) illumination field removed.

    The scanner's lighting drifts between shots in broad gradients (one end of
    the board washed out); a global brightness match leaves that as a 'change'
    over half the frame. A cubic in x, y is fitted robustly (three rounds,
    dropping residuals beyond 2.5 MAD) to the log ratio on a 1/8 grid inside the
    ROI and subtracted; a removed part is local and survives.
    """
    la, lb = np.log(fa + 10.0), np.log(fb + 10.0)
    lr = lb - la
    h, w = lr.shape
    y0, y1, x0, x1 = _region(roi, lr.shape)
    small = cv2.resize(lr[y0:y1, x0:x1], ((x1 - x0) // 8, (y1 - y0) // 8),
                       interpolation=cv2.INTER_AREA)
    sy, sx = np.mgrid[0:small.shape[0], 0:small.shape[1]].astype(np.float64)
    nx, ny = sx / max(1, small.shape[1] - 1), sy / max(1, small.shape[0] - 1)
    terms = _poly_terms(nx.ravel(), ny.ravel())
    z = small.ravel().astype(np.float64)
    keep = np.ones_like(z, bool)
    coef = None
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(terms[keep], z[keep], rcond=None)
        res = z - terms @ coef
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-6
        keep = np.abs(res - np.median(res[keep])) < 2.5 * 1.4826 * mad
    # the field on a 1/8 grid of the whole frame, then up to full size (smooth anyway)
    gh, gw = max(2, h // 8), max(2, w // 8)
    gy, gx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    px, py = (gx + 0.5) * (w / gw), (gy + 0.5) * (h / gh)
    fx = np.clip((px - x0) / max(1.0, (x1 - x0) - 1.0), 0, 1)
    fy = np.clip((py - y0) / max(1.0, (y1 - y0) - 1.0), 0, 1)
    field = np.zeros((gh, gw), np.float32)
    k = 0
    for i in range(4):
        for j in range(4 - i):
            field += float(coef[k]) * fx ** i * fy ** j
            k += 1
    field = cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR)
    return lr - field


#: grey level from which a pixel is treated as blown out (no evidence either way)
SATURATED = 235
#: local normalised cross-correlation below which the texture changed
NCC_CHANGED = 0.6
NCC_WIN = 11
FLAT_STD = 4.0


def _local_ncc(fa: np.ndarray, fb: np.ndarray):
    k = (NCC_WIN, NCC_WIN)
    ma, mb = cv2.blur(fa, k), cv2.blur(fb, k)
    va = np.maximum(cv2.blur(fa * fa, k) - ma * ma, 0)
    vb = np.maximum(cv2.blur(fb * fb, k) - mb * mb, 0)
    cov = cv2.blur(fa * fb, k) - ma * mb
    ncc = cov / np.sqrt(va * vb + 1e-3)
    return ncc, np.sqrt(va), np.sqrt(vb)


def change_maps(a: np.ndarray, b: np.ndarray, roi) -> tuple[np.ndarray, np.ndarray]:
    """``(changed, valid)`` between two grey frames of the fixed camera (bool, bool).

    The scanner's light and white balance drift between shots -- in broad
    gradients, and non-linearly where a surface is near white -- so a plain
    difference reads half the board as changed. A pixel changed when, with the
    smooth illumination field removed (:func:`illumination_ratio`), the 5x5
    log ratio is beyond :data:`CHANGE_LOG` and the difference beyond
    :data:`CHANGE_ABS` grey levels, **and** the local texture changed too
    (11x11 normalised cross-correlation < :data:`NCC_CHANGED`) unless both
    patches are flat. ``valid`` is False where either frame is blown out
    (>= :data:`SATURATED`, dilated 5 px): no evidence there either way.
    """
    fa, fb = a.astype(np.float32), b.astype(np.float32)
    hp = illumination_ratio(fa, fb, roi)
    field = (np.log(fb + 10.0) - np.log(fa + 10.0)) - hp
    fb_lit = (fb + 10.0) * np.exp(-field) - 10.0
    absd = cv2.blur(np.abs(fb_lit - fa), (5, 5))
    tone = (np.abs(cv2.blur(hp, (5, 5))) > CHANGE_LOG) & (absd > CHANGE_ABS)
    ncc, sa, sb = _local_ncc(fa, fb_lit)
    flat = (sa < FLAT_STD) & (sb < FLAT_STD)
    # dark on dark: a flat black drive over a black connector with pins has the
    # same tone but not the same texture -- a 3x local contrast ratio is a change
    texture = (np.abs(np.log((sa + 2.0) / (sb + 2.0))) > np.log(3.0)) & \
        (np.maximum(sa, sb) > 2 * FLAT_STD)
    texture = cv2.blur(texture.astype(np.float32), (5, 5)) > 0.5
    changed = (tone & ((ncc < NCC_CHANGED) | flat)) | texture
    sat = ((a >= SATURATED) | (b >= SATURATED)).astype(np.uint8)
    valid = ~cv2.dilate(sat, np.ones((5, 5), np.uint8)).astype(bool)
    return changed & valid, valid


def change_map(a: np.ndarray, b: np.ndarray, roi) -> np.ndarray:
    """:func:`change_maps`' ``changed`` alone."""
    return change_maps(a, b, roi)[0]


def frac(changed: np.ndarray, valid: np.ndarray, where: np.ndarray):
    """Changed fraction of the valid pixels of ``where``; None when < 30 % is valid."""
    w = where & valid
    n = int(w.sum())
    if n < max(10, 0.3 * int(where.sum())):
        return None
    return float(changed[w].mean())


def bbox_of(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
