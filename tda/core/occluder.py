"""「被线挡住」: a part in its place, hidden by something nobody labels (task U15).

Cables are not annotated. A part that is where it belongs but sits under a
cable -- wholly or in part -- used to get a bare visibility label at most (the
``1``-``7`` keys), which said neither *why* nor that the visible pixels the
compiler worked out are not really seen. The annotator now says it on the
frame: ``I`` on a sequence frame cycles the selected part through

* **完全** -- ``occluded_full``, ``occluder="cable"``, and an **empty** visible
  mask: the compiled visible mask is empty and the label says why; the shape
  (amodal, from whichever frame it was drawn on) stays;
* **部分** -- ``occluded_partial``, ``occluder="cable"``, no mask of its own:
  the compiler's visible pixels are kept, and the label and the reason say part
  of them is under a cable;
* **撤回** -- the label and its reason go; a single-frame mask drawn by hand in
  the same override stays (as :func:`tda.ui.session_edit.clear_visibility`).
  It is never refused: on a frame that can no longer carry the label (the part
  lost its shape here, was taken out, or lies on the bench) the next press is
  撤回 whatever the label was.

This module is the rule both the window and the batch tools read -- what a
label is, what each step of the cycle writes, which stored rows a frame can no
longer carry (``cli check``) -- and it imports neither Qt nor the session.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pycocotools import mask as coco_mask

from tda.core import masks
from tda.core.compiler import select_keyframe
from tda.core.model import (
    OCCLUDER_CABLE,
    OCCLUDERS,
    FrameKey,
    FrameOverride,
    Visibility,
    is_component_view,
)
from tda.core.states import needs_geom
from tda.core.truth_inputs import InputCache, instances_of, pose_segment_of, state_of

__all__ = [
    "CYCLE",
    "FULL",
    "OCCLUDER_WITHOUT_SHAPE",
    "PARTIAL",
    "cable_label",
    "empty_rle",
    "keeps_reason",
    "kept_mask",
    "next_label",
    "occluded_override",
    "occluders_without_shape",
    "shape_applies",
]

FULL = Visibility.OCCLUDED_FULL.value
PARTIAL = Visibility.OCCLUDED_PARTIAL.value
#: What successive presses of ``I`` set on a sequence frame; ``None`` is 撤回.
CYCLE: tuple[Optional[str], ...] = (FULL, PARTIAL, None)
#: ``cli check``'s code for a stored reason with no shape under it.
OCCLUDER_WITHOUT_SHAPE = "occluder_without_shape"


def cable_label(override: Optional[FrameOverride]) -> Optional[str]:
    """The 被线挡住 label an override carries -- :data:`FULL` / :data:`PARTIAL` -- or ``None``.

    Only a label *with* its reason is one: a bare ``occluded_full`` from the
    ``3`` key says nothing about a cable.
    """
    if override is None or not override.occluder:
        return None
    vis = str(override.visibility or "")
    return vis if vis in (FULL, PARTIAL) else None


def next_label(override: Optional[FrameOverride]) -> Optional[str]:
    """What the next press of ``I`` sets: 完全 -> 部分 -> 撤回 -> 完全."""
    current = cable_label(override)
    if current == FULL:
        return PARTIAL
    if current == PARTIAL:
        return None
    return FULL


def keeps_reason(visibility: Optional[str]) -> bool:
    """Can a label still carry an occluder? Only the two occluded labels can."""
    return str(visibility or "") in (FULL, PARTIAL)


def empty_rle(hw: tuple[int, int]) -> dict:
    """An all-empty visible mask of a ``hw`` frame: what 完全 writes.

    Built from the run lengths -- one run of ``H*W`` zeros -- rather than by
    encoding a 12 MP array of them, and kept per size: ``Ctrl+I`` over forty
    OAK frames used to spend most of a second making the same string forty
    times. The same RLE :func:`tda.core.masks.encode_rle` makes of zeros.
    """
    height, width = int(hw[0]), int(hw[1])
    return {"size": [height, width], "counts": _empty_counts(height, width)}


@lru_cache(maxsize=16)
def _empty_counts(height: int, width: int) -> str:
    """The compressed counts of one run of ``height * width`` zeros."""
    rle = coco_mask.frPyObjects({"size": [height, width], "counts": [height * width]},
                                height, width)
    return masks.rle_counts(rle)


def kept_mask(previous: Optional[FrameOverride]) -> Optional[dict]:
    """The single-frame mask of ``previous`` that outlives its 被线挡住 label.

    A hand-drawn mask stays; an **empty** one is 完全's own (or the same
    statement drawn by hand) and goes with the label that made it.
    """
    if previous is None or previous.visible_rle is None:
        return None
    if previous.occluder and masks.rle_area(previous.visible_rle) == 0:
        return None
    return previous.visible_rle


def occluded_override(key: FrameKey, instance: str, previous: Optional[FrameOverride],
                      label: Optional[str], hw: tuple[int, int],
                      occluder: str = OCCLUDER_CABLE) -> Optional[FrameOverride]:
    """The row one step of the cycle leaves, ``None`` when it leaves none.

    ``label`` is :data:`FULL`, :data:`PARTIAL` or ``None`` (撤回); ``previous``
    the row there now. 完全 replaces any mask with an empty one (nothing is
    seen); 部分 keeps a hand-drawn mask and drops 完全's empty one; 撤回 keeps a
    hand-drawn mask under no label, and leaves no row when there is none.
    """
    if occluder not in OCCLUDERS:
        raise ValueError(f"occluder must be one of {OCCLUDERS}, got {occluder!r}")
    if label == FULL:
        return FrameOverride(key, instance, empty_rle(hw), FULL, occluder)
    mask = kept_mask(previous)
    if label == PARTIAL:
        return FrameOverride(key, instance, mask, PARTIAL, occluder)
    if label is not None:
        raise ValueError(f"a 被线挡住 label is {FULL!r}, {PARTIAL!r} or None, got {label!r}")
    if mask is None:
        return None
    return FrameOverride(key, instance, mask, None, None)


def shape_applies(db, tax, key: FrameKey, instance: str,
                  cache: Optional[InputCache] = None, keyframes=None) -> str:
    """Why ``instance`` cannot carry a 被线挡住 label at ``key``, or ``""`` when it can.

    It can where a mask shape of it applies. Otherwise the answer is
    ``"removed"`` (the frame does not need it at all: taken out, or a state
    with no geometry), ``"bench"`` (it lies on the staging area, a rectangle,
    not in its place) or ``"no_shape"`` (no keyframe of its chain reaches this
    frame) -- the compiler's own selection, without compiling anything.
    ``keyframes`` is the instance's keyframes of this view when the caller
    already read them (a run of frames reads them once).
    """
    cache = cache if cache is not None else InputCache()
    instances = instances_of(db, key.desktop, cache)
    state = state_of(db, tax, key.desktop, key.step, cache, view=key.view)
    seg = pose_segment_of(db, key, cache)
    bench = None if is_component_view(key.view) else db.bench_roi(key.desktop, key.view, seg)
    needs = needs_geom(instances, state, tax, bench_roi=bench)
    if instance not in needs:
        return "removed"
    if needs[instance] != "mask":
        return "bench"
    placement = state[instance].placement
    if keyframes is None:
        keyframes = db.keyframes(key.desktop, key.view, instance)
    chain = [kf for kf in keyframes
             if kf.pose_segment == seg and kf.placement == placement]
    return "" if select_keyframe(chain, key.step) is not None else "no_shape"


def occluders_without_shape(db, tax, desktop: int, view: str) -> list[tuple[int, str]]:
    """``(step, instance)`` of every stored reason the frame can no longer carry (``cli check``).

    A row naming an occluder says "it is here, in its place, under a cable":
    on a frame where the part has no shape (the shape was deleted, the part
    merged away), is not there at all (the step table now takes it out
    earlier) or lies on the bench (a rectangle -- whose box a 完全 label would
    empty) that is a statement about nothing, and the exports would carry it
    beside an annotation that has no amodal to keep or no box at all. Data
    repair, not a UI state: the window refuses to write one, and ``I`` there
    takes it back.
    """
    cache = InputCache()
    out: list[tuple[int, str]] = []
    for fo in db.occluded_overrides(desktop, view):
        if shape_applies(db, tax, fo.frame, fo.instance, cache):
            out.append((int(fo.frame.step), str(fo.instance)))
    return out
