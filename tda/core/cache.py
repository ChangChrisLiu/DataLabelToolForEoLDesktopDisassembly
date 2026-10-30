"""Local image cache, scanner burst selection and chassis ROI suggestion (spec 2.4).

The annotator never reads the source drive while a person is waiting: the image
each step needs is copied once into ``<cache>/<view>/D<nn>/s<kkk>.<ext>``.  For
the scanner every step is a *burst* of 5-10 shots of the same scene, so the copy
step also has to decide which shot represents the step:

* the default is ``P_0`` (the first shot of the burst);
* a burst that starts at ``P_2`` (D49 ``RGB261``) has no ``P_0``, so the
  lowest-numbered shot that passes the exposure check is used
  (reason ``"p0_missing"``);
* a ``P_0`` that is unexposed, blown out, or far from the burst's pixel-wise
  median (a hand or tool reaching over the board) is replaced by the sharpest
  remaining shot, and the reason is recorded for human review.

Every decision, together with the per-shot metrics it was made from, is written
to ``<cache>/<view>/D<nn>/manifest.json`` so the choice can be reviewed and
overridden in the UI without re-reading the source drive.

Nothing here ever writes outside ``cache_dir``; the source roots are read-only.
Images are OpenCV BGR arrays and all coordinates are in *original* image pixels.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from functools import lru_cache
from typing import Any, Callable, Iterable, Optional

import cv2
import numpy as np

from tda.core.cache_roi_detect import (  # re-exported: suggest_roi's strategies
    OAK_MAX_AREA_FRAC,
    OAK_MAX_ASPECT,
    OAK_MIN_AREA_FRAC,
    OAK_MIN_ASPECT,
    ROI_MAX_AREA_FRAC,
    ROI_MAX_ASPECT,
    ROI_MIN_AREA_FRAC,
    ROI_MIN_ASPECT,
    box_plausibility,
    oak_chassis_box,
    oak_chassis_candidates,
    pad_box,
    scan_bed_box,
    scan_bed_candidates,
    scan_chassis_box,
    scan_chassis_candidates,
)
from tda.core.cache_thumbs import (DbRoiLookup, add_thumb_args, build_thumbs,  # re-exported
                                   run_thumb_cli, thumb_path)
from tda.core.index import DEFAULT_PATHS_PATH, REPO_ROOT, DesktopIndex, FrameFile, load_index
from tda.core.model import FrameKey
from tda.core.rawroot import resolve_raw

__all__ = [
    "BURST_VIEWS", "DbRoiLookup", "build_cache", "build_thumbs", "burst_metrics", "cache_path",
    "cached_image_path", "choose_scan_image", "configured_cache_dir", "full_frame",
    "has_no_photo", "measure_roi", "missing_copies", "oak_chassis_candidates",
    "prune_missing", "roi_bands", "scan_bed_candidates",
    "scan_chassis_candidates", "suggest_roi", "suggest_roi_over", "thumb_path",
    "OAK_ROI_PAD", "ROI_SAMPLE_FRAMES",
]

#: How many frames of a pose segment its ROI proposal is measured on: the
#: first, the middle and the last that have an image. Three is what a segment
#: of any length can give -- the shortest real one is two steps -- and it is the
#: point where the union stops growing: a fourth frame adds nothing the first,
#: middle and last have not already seen, because the machine does not move.
ROI_SAMPLE_FRAMES = 3

#: How an OAK segment's per-frame measurements become one rectangle:
#: ``"median"`` (per edge), ``"trim"`` (union without the outsized one) or
#: ``"union"``. See :func:`combine_boxes` for why the scanner and OAK differ
#: and ``task-B3-report.md`` for the by-eye numbers the choice was made on.
OAK_COMBINE = "median"

#: ``"trim"`` drops a box this much bigger than the median of the measurements.
OAK_OUTLIER_AREA = 1.5

#: The most an OAK box is grown on any one side, in pixels of the original.
#:
#: The margin exists for one measured thing: the machine leans out over the
#: tape square towards the camera and the detector may only look inside the
#: square, so the box clips the near rail by 100-200 px at full size. Twelve
#: per cent of a small box is about that; twelve per cent of a box that is
#: already 2,500 px wide is 300 px per side, which is not recovering a rail, it
#: is taking in bench -- and it made whether D24 and D29 oak1 ended at 46 % or
#: 70 % of the frame turn on which side of the area ceiling the padded box
#: happened to land.
OAK_PAD_MAX_PX = 200

#: OAK only: how much an accepted box is grown after the gate has judged it.
#: The machine leans out over the tape square towards the camera and the
#: detector may only look inside the square, so the box clips the near rail by
#: 100-200 px at full size; 12 % of the box covers that on every real frame
#: measured without reaching the bench's edge.
OAK_ROI_PAD = 0.12

# --- burst metrics -------------------------------------------------------
DOWNSCALE = 8  # metrics are computed on a 1/8 copy (1600^2 -> 200^2) for speed
SAT_LEVEL = 250  # a pixel counts as saturated when every channel is >= this

# --- burst selection (thresholds pinned by the user, see module docstring)
# Only a gross failure may override decision C10 ("use P_0"); the thresholds are
# absolute, because within a burst the scanner lamp drifts by a gray level or
# two and the white reference board saturates ~30% of every good frame (median
# 0.29, up to 0.58, measured over 2807 real bursts).
MEAN_MIN = 40.0  # mean gray below this -> unexposed (a lamp failure sits at ~28)
SAT_MAX = 0.75  # saturated fraction above this -> blown out; the board alone
#                 reaches 0.52-0.54 on the six brightest desktops, so a lower
#                 bar swapped good P_0 shots for equally bright siblings
DIST_MAX = 20.0  # mean abs difference to the burst median above this gray levels
#                  -> a gross scene difference, e.g. a hand or tool in the shot

# --- cache layout --------------------------------------------------------
VIEW_EXT = {"scan": "png", "rs": "png", "oak1": "jpg", "oak2": "jpg",
            # the components gallery (task U10, the scanner's U42): its camera's files
            "cmp_scan": "png", "cmp_oak1": "jpg", "cmp_oak2": "jpg", "cmp_rs": "png"}
#: The views whose frame is a scanner burst to choose from: the steps and,
#: since task U42, the scanner's all-components scans -- the same bursts of
#: ten taken by the same machine, so the same choice (``p0``, ``p0_dark`` ...).
BURST_VIEWS = frozenset({"scan", "cmp_scan"})
MANIFEST_NAME = "manifest.json"
MANIFEST_FLUSH_EVERY = 25  # flush mid-desktop so an interrupted run resumes
METRICS_VERSION = 1  # bump whenever burst_metrics changes what it measures (a new
#                      metric, another SAT_LEVEL or DOWNSCALE): records stamped
#                      with an older version are measured again instead of being
#                      re-decided from stale numbers.  A record written before the
#                      stamp existed counts as version 1.
SINGLE_REASON = "only"  # views without a burst have nothing to choose

_P_RE = re.compile(r"P_(\d+)", re.IGNORECASE)


def _norm(path: str) -> str:
    """Forward-slash form of ``path`` (the index and manifests use it)."""
    return str(path).replace("\\", "/")


def _source(path) -> str:
    """The file to open for a recorded source path, wherever its drive is today.

    The index and the manifests keep the path as it was recorded -- that is
    their provenance -- and :func:`tda.core.rawroot.resolve_raw` owns the
    question of which letter the raw drive has now. With the drive unplugged
    the read fails as ``OSError``, which :func:`build_cache` records per step
    rather than letting it end the run.
    """
    found = resolve_raw(path)
    if found is None:
        raise OSError(f"the raw data drive is not connected: {path}")
    return found


# ---------------------------------------------------------------------------
# burst metrics
# ---------------------------------------------------------------------------
def _downscale(img: np.ndarray) -> np.ndarray:
    """Area-average ``img`` down by :data:`DOWNSCALE` (never below 1 px)."""
    h, w = img.shape[:2]
    size = (max(1, w // DOWNSCALE), max(1, h // DOWNSCALE))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _p_index(path: str, fallback: int) -> int:
    """Burst number from a ``P_<k>.png`` name, else the position in the burst."""
    m = _P_RE.search(os.path.basename(str(path)))
    return int(m.group(1)) if m else fallback


def burst_metrics(paths: list[str]) -> list[dict]:
    """Per-image quality metrics for one scanner burst, in the given order.

    Each record holds ``path``, ``p_index`` (the ``k`` of ``P_k.png``), ``mean``
    and ``lap_var`` (mean gray and variance of the Laplacian of the 1/8 gray
    copy), ``sat_frac`` (fraction of pixels saturated in all three channels) and
    ``dist_to_median`` (mean absolute difference to the pixel-wise median of the
    downscaled burst - this is what a hand or tool in one shot shows up as).

    **Every number is measured on the 1/8 downscale**, never on the full frame,
    and the thresholds in :func:`choose_scan_image` are calibrated against that:
    area-averaging suppresses sensor noise and softens saturated edges, so
    ``lap_var`` and ``sat_frac`` in particular are not comparable to values taken
    at full resolution.  Changing :data:`DOWNSCALE`, :data:`SAT_LEVEL` or the set
    of metrics therefore means bumping :data:`METRICS_VERSION`, which makes cached
    steps measure themselves again.

    Raises ``ValueError`` on an empty burst and ``OSError`` on an image that
    cannot be decoded.  ``dist_to_median`` is left at 0 if the burst mixes image
    sizes, since no pixel-wise median exists then.
    """
    if not paths:
        raise ValueError("burst_metrics() needs at least one image path")

    grays: list[np.ndarray] = []
    out: list[dict] = []
    for i, path in enumerate(paths):
        img = cv2.imread(_source(path), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"cannot read image: {path}")
        small = _downscale(img)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grays.append(gray)
        out.append({
            "path": _norm(path),
            "p_index": _p_index(path, i),
            "mean": round(float(gray.mean()), 3),
            "sat_frac": round(float(np.all(small >= SAT_LEVEL, axis=2).mean()), 5),
            "lap_var": round(float(cv2.Laplacian(gray, cv2.CV_32F).var()), 2),
            "dist_to_median": 0.0,
        })

    if len({g.shape for g in grays}) == 1:
        median = np.median(np.stack(grays), axis=0)
        for rec, gray in zip(out, grays):
            rec["dist_to_median"] = round(float(np.abs(gray - median).mean()), 3)
    return out


# ---------------------------------------------------------------------------
# burst selection
# ---------------------------------------------------------------------------
def _reject_kind(rec: dict) -> Optional[str]:
    """Why this shot is unusable (``dark``/``saturated``/``outlier``), or None."""
    if float(rec.get("mean", 0.0)) < MEAN_MIN:
        return "dark"
    if float(rec.get("sat_frac", 0.0)) > SAT_MAX:
        return "saturated"
    if float(rec.get("dist_to_median", 0.0)) > DIST_MAX:
        return "outlier"
    return None


def choose_scan_image(metrics: list[dict]) -> tuple[int, str]:
    """Pick the shot that represents the step; return its *position* and why.

    ``P_0`` wins unless it fails :func:`_reject_kind`, in which case the sharpest
    (highest ``lap_var``) shot that passes the same checks is used.  Reasons are
    ``"p0"``, ``"p0_missing"`` (no ``P_0`` in the burst - lowest available number
    that passes), and ``"p0_dark"``/``"p0_saturated"``/``"p0_outlier"`` when
    ``P_0`` was replaced.  When *no* shot passes, the checks cannot say which
    shot is better, so the default wins anyway and the reason gets a ``"_kept"``
    suffix (``"p0_missing_fallback"`` for a burst without a ``P_0``); those are
    the steps to review.
    """
    if not metrics:
        raise ValueError("choose_scan_image() needs at least one metrics record")

    passing = [i for i, m in enumerate(metrics) if _reject_kind(m) is None]

    def p_index(i: int) -> int:
        return int(metrics[i].get("p_index", i))

    def sharpest(pool: Iterable[int]) -> int:
        return max(pool, key=lambda i: (float(metrics[i].get("lap_var", 0.0)), -p_index(i)))

    p0 = next((i for i in range(len(metrics)) if p_index(i) == 0), None)
    if p0 is None:  # e.g. D49 RGB261, whose burst on disk starts at P_2
        pool = passing or list(range(len(metrics)))
        lowest = min(pool, key=lambda i: (p_index(i), i))
        return lowest, "p0_missing" if passing else "p0_missing_fallback"

    if p0 in passing:
        return p0, "p0"

    reason = f"p0_{_reject_kind(metrics[p0])}"
    alternatives = [i for i in passing if i != p0]
    if alternatives:
        return sharpest(alternatives), reason
    return p0, f"{reason}_kept"  # nothing passed: the default still wins


# ---------------------------------------------------------------------------
# ROI suggestion
# ---------------------------------------------------------------------------
def full_frame(width: int, height: int) -> tuple[int, int, int, int]:
    """The whole image as a box -- what "no crop" means in ROI coordinates."""
    return (0, 0, int(width), int(height))


def suggest_roi(img: np.ndarray, view: str) -> tuple[int, int, int, int]:
    """Suggest the chassis ROI ``(x0, y0, x1, y1)`` in original image pixels.

    A scanner frame is measured twice by :mod:`tda.core.cache_roi_detect` -- as
    the darkest object on the board and as whatever is not the scan bed -- and
    the better answer wins.  The dark-object stage is asked first and kept
    whenever it is plausible: it is the stage the thresholds were calibrated on,
    it is tighter on the 55 machines it gets right, and preferring the
    higher-scoring box instead re-cropped most of the 66 real frames for no
    benefit.  The bed stage is the answer for a light or silver machine, where
    the dark one measures something that is not a chassis at all.

    An **OAK** frame is measured by :func:`tda.core.cache_roi_detect.oak_chassis_box`
    instead: the same bench seen from the side, where the tape square covers a
    third of the picture rather than framing it and the rest is floor, operator
    and robot rig.  It is one stage, not two -- "what on the board is not the
    board" finds a black Dell and a bare silver G4 alike -- and it is measured
    on a downscale.

    Any other view, and a frame where no strategy convinces, gets
    the **whole frame**: not a crop at all.  It used to be the central 70 % box,
    on the theory that some crop beats none -- and on all four tape-less real
    desktops that were looked at (D46, D47, D63, D66) it cut the machine in half,
    because a chassis that covers its own tape square is exactly a chassis that
    fills the frame.  A timeline picture of the whole bench is a nuisance; one
    that hides the part being removed is a wrong answer.  The suggestion is
    always confirmed by a human anyway (spec 2.4).
    """
    if img is None or getattr(img, "size", 0) == 0:
        raise ValueError("suggest_roi() needs a non-empty image")
    height, width = img.shape[:2]
    found = measure_roi(img, view)
    return found if found is not None else full_frame(width, height)


def measure_roi(img: np.ndarray, view: str) -> Optional[tuple[int, int, int, int]]:
    """:func:`suggest_roi` without the fallback: the box, or ``None``.

    The difference matters to :func:`suggest_roi_over`, which has to tell "this
    frame found nothing" (skip it) from "this frame says the chassis fills the
    picture" (a box that happens to be the whole frame, which the area gate
    would reject anyway). A view with no detector at all answers ``None`` here
    **without** converting the image first: a greyscale RealSense frame used to
    pay for a colour conversion on the way to a fallback that never looks at it.
    """
    if view != "scan" and not str(view).startswith("oak"):
        return None
    bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if view == "scan":
        for found in (scan_chassis_box(bgr), scan_bed_box(bgr)):
            if found is not None:
                return found[0]
        return None
    found = oak_chassis_box(bgr)
    return None if found is None else found[0]


def _box_area(box) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def combine_boxes(boxes: list, how: str) -> tuple[int, int, int, int]:
    """Several measurements of one segment's chassis, as one rectangle.

    ``"union"`` -- the smallest box holding all of them. Right where a
    measurement can only ever be too *small*, which is the scanner: its failure
    is clipping, so unioning three frames recovers what any one of them cut.

    ``"trim"`` -- the union of the boxes that are not far bigger than the
    others (:data:`OAK_OUTLIER_AREA`), i.e. the union with the one frame that
    grew into the bench left out.

    ``"median"`` -- each edge independently the median of the measurements.
    Two frames have no median, so they union; one is itself. This is the
    variant that survives a measurement being wrong in *either* direction,
    which on an OAK frame it can be: the detector sees the machine against a
    bench, and a shadow or an arm at the bench's edge grows the box as readily
    as a dark floor shrinks it.
    """
    if not boxes:
        raise ValueError("combine_boxes() needs at least one box")
    if how == "median" and len(boxes) >= 3:
        edges = [sorted(float(box[i]) for box in boxes) for i in range(4)]
        middle = len(boxes) // 2
        found = tuple(int(round(edge[middle])) for edge in edges)
        return found  # type: ignore[return-value]
    wanted = list(boxes)
    if how == "trim" and len(boxes) >= 2:
        areas = sorted(_box_area(box) for box in boxes)
        median = (areas[len(areas) // 2] if len(areas) % 2
                  else 0.5 * (areas[len(areas) // 2 - 1] + areas[len(areas) // 2]))
        kept = [box for box in boxes if _box_area(box) <= OAK_OUTLIER_AREA * median]
        wanted = kept or boxes
    return (min(b[0] for b in wanted), min(b[1] for b in wanted),
            max(b[2] for b in wanted), max(b[3] for b in wanted))


def roi_bands(view: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """``(area, aspect)`` bounds a whole-segment box is judged by, per view."""
    if str(view).startswith("oak"):
        return ((OAK_MIN_AREA_FRAC, OAK_MAX_AREA_FRAC),
                (OAK_MIN_ASPECT, OAK_MAX_ASPECT))
    return ((ROI_MIN_AREA_FRAC, ROI_MAX_AREA_FRAC),
            (ROI_MIN_ASPECT, ROI_MAX_ASPECT))


def suggest_roi_over(images: Iterable[np.ndarray],
                     view: str) -> tuple[int, int, int, int]:
    """The chassis ROI of a whole **pose segment**, from several of its frames.

    Inside a pose segment the machine does not move -- that is what a pose
    segment *is* (spec 2.5) -- so every frame of it is a measurement of the same
    rectangle, and the union of those measurements is a better one than any of
    them. It is strictly better in the direction that matters: a per-frame box
    can only be too *small*, because segmentation clips where contrast fails,
    and by the late steps the chassis is nearly empty and its dark floor reads
    as bench. Three measurements of the first, middle and last frame (the caller
    picks them, :data:`ROI_SAMPLE_FRAMES`) cover both ends of that.

    Measured on the real scanner frames of twelve desktops: a single reference
    frame held the whole machine on 8 of 24, the union on 20 of 24, and the two
    boxes that were plausible-looking and *wrong* -- D61 step 39 at 45 % of the
    chassis, D29 step 47 cutting its top quarter -- are right.

    Frames that found nothing are skipped, not counted against it; with nothing
    found at all the answer is the whole frame, i.e. "no proposal", exactly as
    before.

    How the measurements are combined is **per view** (:func:`combine_boxes`).
    The scanner unions them, because its failure is clipping and nothing else.
    An OAK frame is different and the real data says so: on D36 oak1 step 1
    measures ``(839, 264, 3355, 2445)`` and step 22 ``(992, 791, 3622, 3040)``
    of the same, unmoved machine, so one of the two has grown into the bench --
    the camera looks across the bench rather than down at a lit board, and a
    shadow or an arm at its edge inflates a box as readily as a dark floor
    shrinks it. Unioning that with a good box gave 2.6x the area of the right
    answer on D13 oak1. So OAK takes the **per-edge median** instead
    (:data:`OAK_COMBINE`), which needs a measurement to be wrong the same way on
    two frames out of three before it moves the answer.

    The combined box is then judged by the same area and aspect bounds one
    frame's box is judged by (:func:`roi_bands`).

    On an OAK view the accepted box is finally grown by :data:`OAK_ROI_PAD` of
    its own size, while that keeps it inside the area ceiling. The camera looks
    at the bench from the side and the machine leans out over the tape square
    towards it, and the detector may only look inside the square, so the box
    reliably clips the near rail; the gate has already judged the tight box, so
    this cannot let a bad one through.
    """
    boxes: list[tuple[int, int, int, int]] = []
    hw: Optional[tuple[int, int]] = None
    for img in images:
        if img is None or getattr(img, "size", 0) == 0:
            continue
        hw = (int(img.shape[0]), int(img.shape[1]))
        found = measure_roi(img, view)
        if found is not None:
            boxes.append(tuple(int(v) for v in found))  # type: ignore[arg-type]
    if hw is None:
        raise ValueError("suggest_roi_over() needs at least one non-empty image")
    height, width = hw
    whole = full_frame(width, height)
    if not boxes:
        return whole

    is_oak = str(view).startswith("oak")
    combined = combine_boxes(boxes, OAK_COMBINE if is_oak else "union")
    area_band, aspect_band = roi_bands(view)
    # rectangularity is not a question a combined box can answer -- there is no
    # one component behind it -- so the referee is asked about area and aspect
    if box_plausibility(combined, 1.0, width, height, 0.0,
                        area_band=area_band, aspect_band=aspect_band) is None:
        return whole
    if is_oak:
        padded = _pad_oak(combined, width, height)
        area = (padded[2] - padded[0]) * (padded[3] - padded[1])
        if area <= area_band[1] * width * height:
            combined = padded
        # ... and otherwise the tight box stands. On a box that already covers
        # most of the picture the margin only takes in more bench, and on the
        # real D36 and D34 oak1 it turned 68 % of the frame into 86 %.
    return combined


def _pad_oak(box, width: int, height: int) -> tuple[int, int, int, int]:
    """Grow ``box`` by :data:`OAK_ROI_PAD` of itself, at most :data:`OAK_PAD_MAX_PX`."""
    x0, y0, x1, y1 = (int(v) for v in box)
    grow_x = int(round(min(OAK_ROI_PAD * (x1 - x0), OAK_PAD_MAX_PX)))
    grow_y = int(round(min(OAK_ROI_PAD * (y1 - y0), OAK_PAD_MAX_PX)))
    return (max(0, x0 - grow_x), max(0, y0 - grow_y),
            min(width, x1 + grow_x), min(height, y1 + grow_y))


# ---------------------------------------------------------------------------
# the cache itself
# ---------------------------------------------------------------------------
def cache_path(cache_dir, key: FrameKey, ext: str) -> str:
    """Where the image of ``key`` lives in the cache, e.g. ``.../scan/D13/s042.png``."""
    root = _norm(cache_dir).rstrip("/")
    return f"{root}/{key.view}/D{key.desktop:02d}/s{key.step:03d}.{ext}"


def has_no_photo(row: Optional[dict]) -> bool:
    """Does this ``frame`` row say the frame has **no photo of its own**?

    True for a row flagged ``missing`` -- a step this view never photographed
    (spec 4.2, 缺帧处理) -- and for a row with no stored path; whatever its
    ``aux`` still names. Such a frame has no pixels at all: a file at its
    cache name (:func:`cached_image_path`), or a ``cache_path`` left in its
    ``aux``, is a copy of some *other* photo that once had this step number.
    D61's scanner step 26 is the case (task U38): it was ``RGB261`` until the
    index declared the step a missed shot and ``RGB261`` became step 27, and
    the copy cached under ``s026.png`` showed the annotator step 27's picture
    under step 26's number, and the annotator confirmed it.

    ``None`` (no row at all) is not such a row: nothing says the frame is
    missing, and the callers that pass one keep reading the cache as before.
    """
    if row is None:
        return False
    return bool(row.get("missing")) or not row.get("path")


def cached_image_path(cache_dir, key: FrameKey) -> Optional[str]:
    """The local copy of one frame, by the convention :func:`build_cache` writes.

    **The one answer to "where are this frame's pixels, locally".** The image
    cache the canvas draws from and the size measurement the compiler makes both
    ask it, so neither can look somewhere the other does not -- which is how
    ``frame_hw`` came to read 12 MP stills off the read-only source drive on the
    first compile of every view, and to fall back to the view's nominal size
    whenever that drive was detached.

    ``None`` when no cache directory is known; the file may or may not exist.
    """
    if not cache_dir:
        return None
    return cache_path(cache_dir, key, VIEW_EXT.get(key.view, "png"))


@lru_cache(maxsize=8)
def _read_paths(path: str, _stamp: tuple) -> tuple[Optional[str], Optional[str]]:
    """``(db_path, cache_dir)`` of one paths.yaml, memoised per file *version*.

    ``_stamp`` is the file's size and modification time: it takes no part in the
    read, it is what makes the memo expire. A plain cache keyed on the path
    alone would answer from a configuration the file no longer has -- for the
    lifetime of the process, which for the GUI is the working day.
    """
    import yaml

    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except (OSError, ValueError):
        return (None, None)
    return (cfg.get("db_path"), cfg.get("cache_dir"))


def _configured_paths(paths_path: str) -> tuple[Optional[str], Optional[str]]:
    """``(db_path, cache_dir)`` from a paths.yaml, or ``(None, None)``."""
    path = paths_path if (os.path.isabs(paths_path) or os.path.exists(paths_path)) \
        else os.path.join(REPO_ROOT, paths_path)
    try:
        stat = os.stat(path)
        stamp: tuple = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        return (None, None)
    return _read_paths(path, stamp)


def configured_cache_dir(db_path: Optional[str],
                         paths_path: str = DEFAULT_PATHS_PATH) -> Optional[str]:
    """The cache directory ``configs/paths.yaml`` names **for this database**.

    A cache belongs to the annotations it was built for. Answering with the
    configured directory for *any* database would point a test's temporary file
    -- or a probe copy under ``.cache/tmp`` -- at the machine's real cache,
    whose images would then be measured into it: the run's results would depend
    on what happens to be on that disk.

    So the match is on the database path itself, and every other database gets
    ``None``, which simply means "no local cache known" (see
    :func:`cached_image_path`).
    """
    if not db_path:
        return None
    configured_db, cache_dir = _configured_paths(str(paths_path))
    if not configured_db or not cache_dir:
        return None
    try:
        same = os.path.normcase(os.path.abspath(str(db_path))) == os.path.normcase(
            os.path.abspath(str(configured_db)))
    except (OSError, ValueError):  # pragma: no cover - defensive
        return None
    return str(cache_dir) if same else None


def _load_manifest(path: str) -> dict:
    """Read an existing manifest, tolerating a missing or corrupt file."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_manifest(path: str, manifest: dict) -> None:
    """Write the manifest atomically, steps in numeric order."""
    ordered = {k: manifest[k] for k in sorted(manifest, key=lambda s: int(s))}
    tmp = f"{path}.part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(ordered, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def _select_source(frame: FrameFile, view: str) -> dict:
    """Manifest record for one frame: which source file to copy, and why."""
    if view in BURST_VIEWS:
        burst = [str(p) for p in (frame.aux.get("burst") or [frame.path])]
        metrics = burst_metrics(burst)
        chosen, reason = choose_scan_image(metrics)
        return {"chosen": chosen, "reason": reason, "metrics": metrics,
                "src": metrics[chosen]["path"], "metrics_version": METRICS_VERSION}
    return {"chosen": 0, "reason": SINGLE_REASON, "metrics": [],
            "src": _norm(frame.path), "metrics_version": METRICS_VERSION}


def _metrics_stale(record: Optional[dict], view: str) -> bool:
    """True when ``record`` was measured by an older :func:`burst_metrics`."""
    if not record or view not in BURST_VIEWS:
        return False
    return int(record.get("metrics_version", 1)) != METRICS_VERSION


def _source_moved(record: Optional[dict], frame: FrameFile, view: str) -> bool:
    """Is the cached copy from another file than the frame's?

    A components-gallery shot re-pointed by ``load-index`` (task U10) keeps its
    frame key, so its cached copy must be replaced even though it has the
    right name. So does a step the collapse of the sheet's ``dupli`` rows
    renumbered (task U25): D18's scanner step 42 was ``RGB421`` and is
    ``RGB431``. The scanner chooses among a burst, so there the question is
    whether the recorded shot is one of the frame's burst at all (which shot
    of it wins is :func:`_redecide`'s).
    """
    if not record:
        return False
    if view in BURST_VIEWS:
        burst = {_norm(p) for p in (frame.aux.get("burst") or [frame.path]) if p}
        return bool(burst) and _norm(record.get("src") or "") not in burst
    if not frame.path:
        return False
    return record.get("src") != _norm(frame.path)


def _redecide(record: dict, view: str) -> Optional[dict]:
    """Re-run the choice on the metrics already in ``record``; None if unchanged.

    The metrics do not depend on the selection rule, so a cached step can be
    re-decided without touching the source drive again - that is how a re-run
    picks up a changed rule.
    """
    if view not in BURST_VIEWS or not record.get("metrics"):
        return None
    chosen, reason = choose_scan_image(record["metrics"])
    if chosen == record.get("chosen") and reason == record.get("reason"):
        return None
    updated = dict(record)
    updated["chosen"] = chosen
    updated["reason"] = reason
    updated["src"] = record["metrics"][chosen]["path"]
    return updated


def _up_to_date(src: str, dest: str) -> bool:
    """True when ``dest`` already holds a full copy of ``src`` (same byte size)."""
    try:
        return os.path.getsize(_source(src)) == os.path.getsize(dest)
    except OSError:
        return False


def _copy(src: str, dest: str) -> None:
    """Copy ``src`` to ``dest`` atomically (via a ``.part`` file in the cache)."""
    tmp = f"{dest}.part"
    shutil.copyfile(_source(src), tmp)
    os.replace(tmp, dest)


def build_cache(index: dict[int, DesktopIndex], cache_dir: str, views=("scan",),
                desktops=None, progress: Optional[Callable[[int, int, str], Any]] = None,
                force: bool = False, recompute: bool = False) -> dict:
    """Copy the image every step needs into the local cache; write the manifests.

    For each desktop in ``desktops`` (default: all of ``index``) and each view in
    ``views``, the chosen source image is copied to
    :func:`cache_path` and the decision is appended to
    ``<cache>/<view>/D<nn>/manifest.json`` as
    ``{step: {"chosen", "reason", "metrics", "src"}}``.

    Re-runs are cheap and idempotent: a step whose manifest record is present and
    whose cached file already has the source's byte size is not read again, only
    *re-decided* from the metrics in its record - so a re-run after a change to
    :func:`choose_scan_image` replaces the cached file (``rechosen``) or just the
    recorded reason (``relabelled``) without touching the source drive.  That
    shortcut only holds while the stored metrics still mean what they say, so a
    record stamped with an older :data:`METRICS_VERSION` is measured again;
    ``recompute=True`` measures every step again regardless (copying only what
    actually changed) and ``force=True`` additionally re-copies every file.  The
    manifest is flushed every :data:`MANIFEST_FLUSH_EVERY` copies as well as at
    the end of each desktop, so an interrupted run resumes instead of redoing the
    desktop it was in.  A step that cannot be read is recorded in ``failures``
    and does not stop the run.  ``progress(desktop, step, view)`` is called once
    per step, after it is handled; an exception it raises is not caught.

    Returns counters plus the manifests: ``copied``, ``skipped``, ``rechosen``,
    ``relabelled``, ``steps``, ``bytes_copied``, ``failures``, ``non_p0`` (every
    scanner step whose cached image is *not* ``P_0``), ``flagged`` (``P_0`` kept
    although the burst failed a check - review the burst, not the choice),
    ``manifests`` and ``elapsed_s``.  Both lists cover every step seen, whether
    written now or already cached.
    """
    started = time.perf_counter()
    stats: dict[str, Any] = {
        "copied": 0, "skipped": 0, "rechosen": 0, "relabelled": 0, "steps": 0,
        "bytes_copied": 0, "failures": [], "non_p0": [], "flagged": [],
        "manifests": {}, "elapsed_s": 0.0,
    }
    wanted = None if desktops is None else set(int(d) for d in desktops)

    for desktop in sorted(index):
        if wanted is not None and desktop not in wanted:
            continue
        frames = index[desktop].frames
        # a step the index says has no photo is never copied (task U38): a copy
        # under its number would be shown as its picture
        absent = set(index[desktop].missing)
        for view in views:
            keys = sorted((k for k in frames if k.view == view and k not in absent),
                          key=lambda k: k.step)
            if not keys:
                continue
            ext = VIEW_EXT.get(view, "png")
            view_dir = os.path.dirname(cache_path(cache_dir, keys[0], ext))
            os.makedirs(view_dir, exist_ok=True)
            manifest_path = f"{view_dir}/{MANIFEST_NAME}"
            manifest = _load_manifest(manifest_path)
            dirty = False

            for key in keys:
                stats["steps"] += 1
                dest = cache_path(cache_dir, key, ext)
                step = str(key.step)
                try:
                    record = manifest.get(step)
                    measure = (force or recompute or _metrics_stale(record, view)
                               or _source_moved(record, frames[key], view))
                    if record and not measure and _up_to_date(record.get("src", ""), dest):
                        again = _redecide(record, view)
                        if again is None:
                            stats["skipped"] += 1
                        elif again["src"] != record["src"]:  # another shot wins now
                            _copy(again["src"], dest)
                            manifest[step] = record = again
                            dirty = True
                            stats["copied"] += 1
                            stats["rechosen"] += 1
                            stats["bytes_copied"] += os.path.getsize(dest)
                        else:  # same image, new reason: only the record changes
                            manifest[step] = record = again
                            dirty = True
                            stats["relabelled"] += 1
                    else:
                        fresh = _select_source(frames[key], view)
                        if force or not _up_to_date(fresh["src"], dest):
                            _copy(fresh["src"], dest)
                            stats["copied"] += 1
                            stats["bytes_copied"] += os.path.getsize(dest)
                            if record and record.get("src") != fresh["src"]:
                                stats["rechosen"] += 1
                        else:  # measured again, same image: only the record changes
                            stats["relabelled"] += 1
                        manifest[step] = record = fresh
                        dirty = True
                    if record["metrics"] and record["reason"] != "p0":
                        where = ("non_p0"
                                 if record["metrics"][record["chosen"]]["p_index"] != 0
                                 else "flagged")
                        stats[where].append({
                            "desktop": desktop, "step": key.step, "view": view,
                            "chosen": record["chosen"], "reason": record["reason"],
                            "src": record["src"],
                        })
                except Exception as exc:  # a broken source must not stop the run
                    stats["failures"].append({
                        "desktop": desktop, "step": key.step, "view": view,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                if dirty and stats["copied"] % MANIFEST_FLUSH_EVERY == 0:
                    _write_manifest(manifest_path, manifest)  # resume point
                    dirty = False
                if progress is not None:
                    progress(desktop, key.step, view)

            if dirty:
                _write_manifest(manifest_path, manifest)
            stats["manifests"].setdefault(view, {})[desktop] = manifest

    stats["elapsed_s"] = round(time.perf_counter() - started, 2)
    return stats


# ---------------------------------------------------------------------------
# copies of steps that have no photo (task U38)
# ---------------------------------------------------------------------------
def missing_rows_from_db(db_path: str, views: Iterable[str],
                         desktops: Iterable[int]) -> set[FrameKey]:
    """The frame rows flagged ``missing`` in the database (opened read-only)."""
    import sqlite3

    if not db_path or not os.path.isfile(db_path):
        return set()
    uri = "file:" + _norm(os.path.abspath(db_path)) + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    found: set[FrameKey] = set()
    try:
        for desktop in desktops:
            for view in views:
                for (step,) in conn.execute(
                        "SELECT step FROM frame WHERE desktop=? AND view=? "
                        "AND COALESCE(missing, 0)=1", (int(desktop), str(view))):
                    found.add(FrameKey(int(desktop), int(step), str(view)))
    finally:
        conn.close()
    return found


def missing_copies(index: dict[int, DesktopIndex], cache_dir: str, views=("scan",),
                   desktops=None, db_missing: Iterable[FrameKey] = ()) -> list[dict]:
    """What the cache still holds for steps that have **no photo** (task U38).

    A step is one when the index lists it as ``missing`` -- or, with
    ``db_missing``, when its frame row is flagged ``missing`` -- and the index
    has no frame for it. Whatever the cache holds under its number was copied
    from another photo that had the number before the index changed: D61's
    scanner step 26 was ``RGB261`` until U25 declared it a missed shot, and
    ``s026.png``, its thumbnail and its manifest record stayed.
    :func:`build_cache` never copies such a step, and nothing ever deletes what
    an earlier run copied: this finds it.

    One ``{"desktop", "step", "view", "files", "manifest"}`` per step with
    anything left -- ``files`` the full copy and the thumbnail that exist,
    ``manifest`` whether ``manifest.json`` still lists the step -- in
    ``(desktop, view, step)`` order.
    """
    root = _norm(cache_dir).rstrip("/")
    wanted = None if desktops is None else {int(d) for d in desktops}
    from_db: dict[tuple[int, str], set[int]] = {}
    for key in db_missing:
        from_db.setdefault((int(key.desktop), str(key.view)), set()).add(int(key.step))
    numbers = set(index) | {d for d, _v in from_db}
    out: list[dict] = []
    for desktop in sorted(numbers):
        if wanted is not None and desktop not in wanted:
            continue
        di = index.get(desktop)
        frames = di.frames if di is not None else {}
        for view in views:
            steps = set(from_db.get((desktop, view), set()))
            if di is not None:
                steps |= {int(k.step) for k in di.missing if k.view == view}
            steps -= {int(k.step) for k in frames if k.view == view}
            if not steps:
                continue
            ext = VIEW_EXT.get(view, "png")
            manifest = _load_manifest(f"{root}/{view}/D{desktop:02d}/{MANIFEST_NAME}")
            for step in sorted(steps):
                key = FrameKey(desktop, step, view)
                files = [p for p in (cache_path(cache_dir, key, ext), thumb_path(cache_dir, key))
                         if os.path.isfile(p)]
                listed = str(step) in manifest
                if files or listed:
                    out.append({"desktop": desktop, "step": step, "view": view,
                                "files": files, "manifest": listed})
    return out


def prune_missing(found: list[dict], cache_dir: str) -> dict:
    """Delete what :func:`missing_copies` found: the files and the manifest records.

    The manifest record goes too: the manifest is the authority on which steps
    a desktop has, and :func:`build_thumbs` would otherwise make the thumbnail
    again from it. Returns ``{"files", "records", "failures"}`` -- what was
    deleted, and ``{"path", "error"}`` for what could not be (a file in use is
    left, reported, and the run goes on).
    """
    root = _norm(cache_dir).rstrip("/")
    stats: dict[str, Any] = {"files": 0, "records": 0, "failures": []}
    by_manifest: dict[str, list[int]] = {}
    for item in found:
        for path in item["files"]:
            try:
                os.remove(path)
                stats["files"] += 1
            except OSError as exc:
                stats["failures"].append({"path": path, "error": f"{type(exc).__name__}: {exc}"})
        if item["manifest"]:
            path = f"{root}/{item['view']}/D{int(item['desktop']):02d}/{MANIFEST_NAME}"
            by_manifest.setdefault(path, []).append(int(item["step"]))
    for path, steps in by_manifest.items():
        manifest = _load_manifest(path)
        dropped = [s for s in steps if manifest.pop(str(s), None) is not None]
        if dropped:
            try:
                _write_manifest(path, manifest)
                stats["records"] += len(dropped)
            except OSError as exc:
                stats["failures"].append({"path": path, "error": f"{type(exc).__name__}: {exc}"})
    return stats


def _missing_line(item: dict) -> str:
    """One step's leftovers, for the log."""
    what = [os.path.basename(p) if "/thumbs/" not in _norm(p) else "thumbs/" + os.path.basename(p)
            for p in item["files"]]
    if item["manifest"]:
        what.append("manifest record")
    return (f"D{int(item['desktop']):02d} s{int(item['step']):03d} {item['view']}: "
            f"{', '.join(what)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def gallery_frames_from_db(index: dict[int, DesktopIndex], db_path: str,
                           views: Iterable[str], desktops: Iterable[int]) -> int:
    """Take the components gallery's frames from the database, in place.

    The gallery (task U10) is the one place the database knows more than the
    index: a sheet row of the laid-out parts (D66 step 13) is moved into the
    gallery by ``load-index``/``import-logs`` because only the step table says
    what it is, and the index -- built before the step table exists -- still
    lists it as a sequence step. So for every gallery view asked for, the
    frames are the database's rows (read-only), which is exactly what the
    session will open. Returns how many gallery frames were taken.
    """
    import sqlite3

    from tda.core.model import is_component_view

    wanted = [v for v in views if is_component_view(v)]
    if not wanted or not db_path or not os.path.isfile(db_path):
        return 0
    uri = "file:" + _norm(os.path.abspath(db_path)) + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    taken = 0
    try:
        for desktop in desktops:
            di = index.get(int(desktop))
            if di is None:
                continue
            for view in wanted:
                for key in [k for k in di.frames if k.view == view]:
                    del di.frames[key]
                rows = conn.execute(
                    "SELECT step, path, aux_json, ts FROM frame WHERE desktop=? AND view=? "
                    "AND COALESCE(missing, 0)=0 AND path IS NOT NULL AND path<>'' "
                    "ORDER BY step", (int(desktop), view)).fetchall()
                for step, path, aux_json, ts in rows:
                    key = FrameKey(int(desktop), int(step), view)
                    aux = json.loads(aux_json) if aux_json else {}
                    di.frames[key] = FrameFile(key=key, path=path, aux=aux, ts=ts,
                                               src_step_dir="")
                    taken += 1
    finally:
        conn.close()
    return taken


def _default_cache_dir(paths_path: str = DEFAULT_PATHS_PATH) -> str:
    """``cache_dir`` from configs/paths.yaml."""
    import yaml

    path = paths_path if (os.path.isabs(paths_path) or os.path.exists(paths_path)) \
        else os.path.join(REPO_ROOT, paths_path)
    with open(path, "r", encoding="utf-8") as fh:
        found = (yaml.safe_load(fh) or {}).get("cache_dir")
    # No literal fallback: the configuration owns this path, and a default
    # pointing at one machine's real cache is how a test writes to it.
    if not found:
        raise KeyError(f"{path} defines no cache_dir")
    return str(found)


def _configure_raw(paths_path: str = DEFAULT_PATHS_PATH):
    """Find the raw drive for this run (:mod:`tda.core.rawroot`); ``None`` on no config."""
    import yaml

    from tda.core import rawroot

    path = paths_path if (os.path.isabs(paths_path) or os.path.exists(paths_path)) \
        else os.path.join(REPO_ROOT, paths_path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except (OSError, ValueError):
        return None
    return rawroot.configure(cfg if isinstance(cfg, dict) else {})


def _run_prune(args, cache_dir: str, index_path: str, views, desktops) -> int:
    """``--prune-missing``: delete the copies of steps with no photo; 1 on a failure."""
    index = load_index(index_path)
    gallery_frames_from_db(index, args.db or "", views, desktops)
    found = missing_copies(index, cache_dir, views, desktops,
                           missing_rows_from_db(args.db or "", views, desktops))
    log = open(args.log, "a", encoding="utf-8", buffering=1) if args.log else None

    def emit(line: str) -> None:
        print(line, flush=True)
        if log:
            log.write(line + "\n")

    try:
        emit(f"[{time.strftime('%H:%M:%S')}] prune-missing views={views} "
             f"desktops={args.first}-{args.last} cache={cache_dir}: {len(found)} step(s) "
             f"with no photo have cached copies")
        for item in found:
            emit(f"  prune {_missing_line(item)}")
        stats = prune_missing(found, cache_dir)
        emit(f"[{time.strftime('%H:%M:%S')}] deleted {stats['files']} file(s) and "
             f"{stats['records']} manifest record(s), failures={len(stats['failures'])}")
        for item in stats["failures"]:
            emit(f"  FAIL {item['path']}: {item['error']}")
        return 1 if stats["failures"] else 0
    finally:
        if log:
            log.close()


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: build the cache for a range of desktops, logging progress.

    Every run also lists what the cache still holds for steps with no photo
    (:func:`missing_copies`, task U38); ``--prune-missing`` deletes exactly
    that and does nothing else.

    Returns 0, or 1 when any step could not be cached (the failures are listed
    and counted in the log, so a shell or scheduler notices a partial run).
    """
    import argparse

    ap = argparse.ArgumentParser(description="Build the local TDA image cache.")
    ap.add_argument("--cache", default=None, help="cache dir (default: paths.yaml cache_dir)")
    ap.add_argument("--index", default=None, help="index JSON (default: <cache>/index.json)")
    ap.add_argument("--views", default="scan", help="comma separated views")
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=66)
    ap.add_argument("--log", default=None, help="progress log file (default: stdout only)")
    ap.add_argument("--recompute", action="store_true",
                    help="measure every burst again instead of reusing the stored metrics")
    ap.add_argument("--force", action="store_true",
                    help="rebuild: measure again and re-copy every file")
    ap.add_argument("--prune-missing", action="store_true",
                    help="delete the cached copies (image, thumbnail, manifest record) of "
                         "every step with no photo -- the index's missing steps and, with "
                         "--db, the database's missing rows -- and stop: nothing is copied "
                         "and the source drive is not read")
    # ``--db`` (add_thumb_args) is read for the components gallery's frames too
    add_thumb_args(ap)
    args = ap.parse_args(argv)

    cache_dir = args.cache or _default_cache_dir()
    index_path = args.index or f"{_norm(cache_dir).rstrip('/')}/index.json"
    views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    desktops = [d for d in range(args.first, args.last + 1)]
    if args.prune_missing:  # the index and the cache only: no source drive
        return _run_prune(args, cache_dir, index_path, views, desktops)
    if args.thumbs_only:  # the tier alone needs neither the index nor the source drive
        return run_thumb_cli(args, cache_dir, views, desktops)

    raw = _configure_raw()
    index = load_index(index_path)
    gallery_frames_from_db(index, args.db or "", views, desktops)
    total = sum(1 for d in desktops if d in index
                for k in index[d].frames if k.view in views)
    log = open(args.log, "a", encoding="utf-8", buffering=1) if args.log else None

    def emit(line: str) -> None:
        print(line, flush=True)
        if log:
            log.write(line + "\n")

    started = time.time()
    done = 0

    def on_step(desktop: int, step: int, view: str) -> None:
        nonlocal done
        done += 1
        if done % 25 == 0 or done == total:
            elapsed = time.time() - started
            eta = elapsed / done * (total - done) if done else 0.0
            emit(f"[{time.strftime('%H:%M:%S')}] {done}/{total} "
                 f"({100.0 * done / max(total, 1):.1f}%) D{desktop:02d} s{step:03d} {view} "
                 f"elapsed {elapsed / 60:.1f}min eta {eta / 60:.1f}min")

    emit(f"[{time.strftime('%H:%M:%S')}] start views={views} desktops={args.first}-{args.last} "
         f"steps={total} cache={cache_dir} index={index_path}")
    if raw is not None and raw.log_line:
        emit(f"  {raw.log_line}")
    code = 0
    try:
        stats = build_cache(index, cache_dir, views=views, desktops=desktops, progress=on_step,
                            force=args.force, recompute=args.recompute)
        emit(f"[{time.strftime('%H:%M:%S')}] done steps={stats['steps']} copied={stats['copied']} "
             f"skipped={stats['skipped']} rechosen={stats['rechosen']} "
             f"relabelled={stats['relabelled']} GB={stats['bytes_copied'] / 1e9:.2f} "
             f"non_p0={len(stats['non_p0'])} flagged={len(stats['flagged'])} "
             f"failures={len(stats['failures'])} elapsed={stats['elapsed_s'] / 60:.1f}min")
        counts: dict[str, int] = {}
        for item in stats["non_p0"] + stats["flagged"]:
            counts[item["reason"]] = counts.get(item["reason"], 0) + 1
        emit(f"  reasons {dict(sorted(counts.items()))}")
        for item in stats["non_p0"]:
            emit(f"  non_p0 D{item['desktop']:02d} s{item['step']:03d} "
                 f"chosen={item['chosen']} reason={item['reason']} src={item['src']}")
        for item in stats["flagged"]:
            emit(f"  flagged D{item['desktop']:02d} s{item['step']:03d} "
                 f"reason={item['reason']} (P_0 kept)")
        for item in stats["failures"]:
            emit(f"  FAIL D{item['desktop']:02d} s{item['step']:03d} {item['view']}: {item['error']}")
        if stats["failures"]:
            code = 1
            emit(f"[{time.strftime('%H:%M:%S')}] {len(stats['failures'])} step(s) could not be "
                 f"cached - see the FAIL lines above")
        # what an earlier run copied for a step that has no photo now (U38):
        # said on every run, deleted only when asked
        left = missing_copies(index, cache_dir, views, desktops,
                              missing_rows_from_db(args.db or "", views, desktops))
        if left:
            emit(f"  {len(left)} step(s) with no photo still have cached copies of another "
                 f"photo (the window does not show them): python -m tda.core.cache "
                 f"--prune-missing deletes them")
            for item in left:
                emit(f"  stale {_missing_line(item)}")
        code = run_thumb_cli(args, cache_dir, views, desktops, emit) or code  # no-op w/o --thumbs
    finally:
        if log:
            log.close()
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
