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

**画框外** (task U39) is the fourth answer, only in ``Ctrl+```'s dialog (the
cycle of the key does not reach it): the part is in the machine, in its state,
but **outside this view's picture** on this frame -- or the camera cannot take
it at all (D63's air shroud, beyond the main view's edge). It is not 被挡住: a
part under a cable or behind another part is in the picture and keeps an
amodal shape to draw; one out of the frame has nothing to draw. Written as the
components gallery's 「看不见」 is (task U10): ``out_of_view``, no pixels, no
reason (:data:`OFF_FRAME`, :func:`off_frame`). It needs no shape -- one there
is kept, as an estimate of the whole part, but the frame shows none of it --
and the compiler reports no ``missing_shape`` for it. 撤回 takes it back like
the cable labels (the key's next press after it is 撤回).

This module is the rule both the window and the batch tools read -- what a
label is, what each step of the cycle writes, which stored rows a frame can no
longer carry (``cli check``) -- and it imports neither Qt nor the session.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pycocotools import mask as coco_mask

from tda.core import masks
from tda.core.compiler import marked_absent, select_keyframe
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
    "CHOICES",
    "CYCLE",
    "FULL",
    "OCCLUDER_WITHOUT_SHAPE",
    "OFF_FRAME",
    "PARTIAL",
    "cable_label",
    "empty_rle",
    "frame_label",
    "keeps_reason",
    "kept_mask",
    "next_label",
    "occluded_override",
    "occluders_without_shape",
    "off_frame",
    "off_frame_applies",
    "shape_applies",
]

FULL = Visibility.OCCLUDED_FULL.value
PARTIAL = Visibility.OCCLUDED_PARTIAL.value
#: 画框外 (task U39): in the machine, outside this view's picture on this frame.
#: The components gallery's 「看不见」 row: ``out_of_view``, no pixels, no reason.
OFF_FRAME = Visibility.OUT_OF_VIEW.value
#: What successive presses of ``I`` set on a sequence frame; ``None`` is 撤回.
CYCLE: tuple[Optional[str], ...] = (FULL, PARTIAL, None)
#: What ``Ctrl+```'s dialog offers, in its order: the cycle, then 画框外 (U39).
CHOICES: tuple[Optional[str], ...] = (*CYCLE, OFF_FRAME)
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


def off_frame(override: Optional[FrameOverride]) -> bool:
    """Does ``override`` say 画框外 -- ``out_of_view`` with no pixels (task U39)?

    The compiler's :func:`~tda.core.compiler.marked_absent`: on a gallery shot
    the same row is 「看不见」 (task U10). A bare ``out_of_view`` set by a digit
    key in Review on a frame with no hand-drawn mask is the same statement.
    """
    return marked_absent(override)


def frame_label(override: Optional[FrameOverride]) -> Optional[str]:
    """What `` ` `` / ``Ctrl+``` said on this frame: 完全 / 部分 / 画框外, or ``None``.

    :func:`cable_label` (a label with its reason -- 被线挡住, or 角度看不到) or
    :data:`OFF_FRAME` (task U39).
    """
    if off_frame(override):
        return OFF_FRAME
    return cable_label(override)


def next_label(override: Optional[FrameOverride]) -> Optional[str]:
    """What the next press of ``I`` sets: 完全 -> 部分 -> 撤回 -> 完全.

    After 画框外 (task U39, set in ``Ctrl+```'s dialog only) it is 撤回: the
    key takes it back, and never sets it.
    """
    if off_frame(override):
        return None
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

    ``label`` is :data:`FULL`, :data:`PARTIAL`, :data:`OFF_FRAME` or ``None``
    (撤回); ``previous`` the row there now. 完全 replaces any mask with an empty
    one (nothing is seen); 部分 keeps a hand-drawn mask and drops 完全's empty
    one; 撤回 keeps a hand-drawn mask under no label, and leaves no row when
    there is none. 画框外 (task U39) is ``out_of_view`` with no pixels and no
    reason: nothing of the part is in the picture, so no mask -- a hand-drawn
    one it replaces is in the op's inverse.
    """
    if occluder not in OCCLUDERS:
        raise ValueError(f"occluder must be one of {OCCLUDERS}, got {occluder!r}")
    if label == OFF_FRAME:
        return FrameOverride(key, instance, None, OFF_FRAME, None)
    if label == FULL:
        return FrameOverride(key, instance, empty_rle(hw), FULL, occluder)
    mask = kept_mask(previous)
    if label == PARTIAL:
        return FrameOverride(key, instance, mask, PARTIAL, occluder)
    if label is not None:
        raise ValueError(f"a 被线挡住 label is {FULL!r}, {PARTIAL!r}, {OFF_FRAME!r} or None, "
                         f"got {label!r}")
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


def off_frame_applies(db, tax, key: FrameKey, instance: str,
                      cache: Optional[InputCache] = None) -> str:
    """Why ``instance`` cannot be said 画框外 at ``key``, or ``""`` when it can (U39).

    :func:`shape_applies` without the shape: 画框外 says there is nothing of it
    in the picture to draw, so no keyframe is asked for. ``"removed"`` (the
    frame does not need it: taken out, or a state with no geometry) and
    ``"bench"`` (on the staging area, which is in the picture by definition)
    stay refusals. No keyframe is read.
    """
    why = shape_applies(db, tax, key, instance, cache, keyframes=())
    return "" if why == "no_shape" else why


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
