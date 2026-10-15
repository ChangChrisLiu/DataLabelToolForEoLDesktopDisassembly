"""The task card's ✂ rows, Qt-free (task U8 round 3).

A ✂ row says that the state of a part changes between the frame on screen and
the frame the reverse walk came from (``step`` and ``neighbour``), that both
sides need geometry, and -- while it is *open* -- that one version of the shape
is still in force on both sides. Three readers need exactly that answer and no
other:

* the task card (:mod:`tda.ui.session_tasks`), which shows the row;
* ``cli check``, which counts the open rows on confirmed frames
  (:func:`stale_split_rows`, ``shape_predates_state``) -- and must not import
  Qt to do it;
* the commit rule of round 3: **a shape commit never crosses an open state
  boundary** (:func:`open_split_steps`), so ``Enter`` or ``Ctrl+K`` on one
  frame of a state run cuts the version at the run's edge instead of
  rewriting the frames of the other state.

So the decisions live here once -- :func:`kind_for`, :func:`has_split`,
:func:`keyframe_at` -- and the card calls them.
"""
from __future__ import annotations

from typing import Iterable, Optional

from tda.core.compiler import select_keyframe
from tda.core.db import Db
from tda.core.model import FrameKey, Placement, Visibility, is_component_view
from tda.core.states import FrameState, diff_states, needs_geom
from tda.core.taxonomy import Taxonomy
from tda.core.truth_inputs import (
    InputCache,
    annotatable_steps,
    instances_of,
    pose_segment_of,
    state_of,
)

__all__ = [
    "KIND_ADD_SHAPE", "KIND_SPLIT_KEYFRAME", "KIND_STATE_ONLY", "NOTHING_VISIBLE",
    "SHAPE_PREDATES_STATE", "SPLIT_TRANSITIONS", "STATE_ONLY_TRANSITIONS", "has_split",
    "keyframe_at", "kind_for", "open_split_steps", "reverse_walk", "split_rows_at",
    "stale_split_rows",
]

#: The card's kinds this module decides between (the values of
#: ``tda.ui.session_api.KIND_*``, which a test keeps equal).
KIND_ADD_SHAPE = "add_shape"
KIND_SPLIT_KEYFRAME = "split_keyframe"
KIND_STATE_ONLY = "state_only"

REMOVED = "removed"
IN_CHASSIS = Placement.IN_CHASSIS.value
_VERIFIED = "verified"
#: Step types whose card is a single "confirm" row: no ✂ is ever shown there.
_CONFIRM_ONLY = ("dupli", "failed")

#: State transitions that keep the shape but need a new version of it here.
SPLIT_TRANSITIONS = frozenset(
    {("open", "closed"), ("unplugged", "plugged"), ("displaced", "installed")}
)
#: ``loosened -> fastened``: the shape carries over untouched.
STATE_ONLY_TRANSITIONS = frozenset({("loosened", "fastened")})
#: Visibility labels that say nothing of the instance is visible on a frame:
#: set there, a ✂ row has nothing to draw (U2b round 3, I-A).
NOTHING_VISIBLE = frozenset({Visibility.OUT_OF_VIEW.value, Visibility.OCCLUDED_FULL.value})
#: The problem ``cli check`` reports for an open ✂ row on a confirmed frame.
SHAPE_PREDATES_STATE = "shape_predates_state"


def kind_for(changes: dict, needs_neighbour: bool, needs_here: bool) -> str:
    """Map one instance's difference from the neighbour onto a kind (spec 4.2).

    ``changes`` reads ``neighbour value -> value on this frame``.  The three
    named transition sets come straight from the spec.  Anything else falls back
    on where geometry is required: a shape needed *here* but not next door has
    to be drawn, one needed on both sides needs a second version, and one needed
    on neither is a label change.
    """
    transition = changes.get("state")
    if transition is not None:
        old, new = transition
        if old == REMOVED and new != REMOVED:
            return KIND_ADD_SHAPE
        if transition in SPLIT_TRANSITIONS:
            return KIND_SPLIT_KEYFRAME
        if transition in STATE_ONLY_TRANSITIONS:
            return KIND_STATE_ONLY
    if changes.get("placement") is not None and needs_here:
        # the chassis and bench chains of an instance are independent (spec 3.3),
        # so a part that changed side carries no shape on this one yet, whichever
        # way the annotator is walking
        return KIND_ADD_SHAPE
    if needs_here and not needs_neighbour:
        return KIND_ADD_SHAPE
    if needs_here and needs_neighbour:
        return KIND_SPLIT_KEYFRAME
    return KIND_STATE_ONLY


def keyframe_at(db: Db, tax: Taxonomy, key: FrameKey, instance: str,
                cache: Optional[InputCache] = None):
    """The keyframe in force for ``instance`` at ``key``, or ``None``.

    The chain of the frame's pose segment and of the placement the instance has
    at that step -- what the compiler selects from.
    """
    seg = pose_segment_of(db, key, cache)
    held = state_of(db, tax, key.desktop, key.step, cache, view=key.view).get(instance)
    placement = IN_CHASSIS if held is None else held.placement
    chain = [kf for kf in db.keyframes(key.desktop, key.view, instance)
             if kf.pose_segment == seg and kf.placement == placement]
    return select_keyframe(chain, key.step)


def has_split(db: Db, tax: Taxonomy, desktop: int, view: str, instance: str,
              step: int, neighbour: int, cache: Optional[InputCache] = None) -> bool:
    """Is there a shape boundary between ``step`` and ``neighbour`` (U2b round 2)?

    A ✂ row asks for a *new version* of a shape from this frame on.  The old
    version reaching this frame is exactly what the row is about, so "a shape
    applies here" was true before anybody did anything.  Done means the version
    in force here is not the one in force next door (a keyframe anchored here,
    in the reverse walk), or a frame override for this frame that settles the
    question: a drawn per-frame shape, or a label saying nothing of it can be
    seen here (``out_of_view`` / ``occluded_full`` -- nothing to draw).  Any
    other label is only a label (U2b round 3, I-A).
    """
    here_key = FrameKey(desktop, step, view)
    try:
        override = (db.frame_overrides(here_key) or {}).get(instance)
    except Exception:  # noqa: BLE001 - a missing table is "no override"
        override = None
    if override is not None and (override.visible_rle is not None
                                 or str(override.visibility) in NOTHING_VISIBLE):
        return True
    here = keyframe_at(db, tax, here_key, instance, cache)
    if here is None:
        return False
    there = keyframe_at(db, tax, FrameKey(desktop, neighbour, view), instance, cache)
    return there is None or there.id != here.id


def _grouped(there: FrameState, here: FrameState) -> dict[str, dict]:
    changed: dict[str, dict] = {}
    for instance, attr, before, after in diff_states(there, here):
        changed.setdefault(instance, {})[attr] = (before, after)
    return changed


def split_rows_at(db: Db, tax: Taxonomy, desktop: int, view: str, step: int,
                  neighbour: Optional[int], cache: Optional[InputCache] = None,
                  only: Optional[str] = None) -> list[tuple[str, tuple, bool]]:
    """``(instance, transition, done)`` of the ✂ rows of ``step``'s card.

    The card diffed against ``neighbour`` (the frame the walk came from).
    ``only`` narrows it to one instance. No rows on a start frame (no
    neighbour) and on a ``dupli`` / ``failed`` step, whose card is a single
    confirmation -- the same gates :func:`tda.ui.session_tasks.task_card_for`
    has.
    """
    if is_component_view(view):
        return []   # a gallery shot has no state, so no boundary (task U10)
    cache = cache if cache is not None else InputCache()
    records = {rec.step: rec for rec in db.steps(desktop)}
    rec = records.get(step)
    if rec is None or neighbour is None or neighbour not in records:
        return []
    if rec.dupli or rec.step_type in _CONFIRM_ONLY:
        return []
    instances = instances_of(db, desktop, cache)
    here_key, there_key = FrameKey(desktop, step, view), FrameKey(desktop, neighbour, view)
    state_here = state_of(db, tax, desktop, step, cache)
    state_there = state_of(db, tax, desktop, neighbour, cache)
    roi_here = db.bench_roi(desktop, view, pose_segment_of(db, here_key, cache))
    roi_there = db.bench_roi(desktop, view, pose_segment_of(db, there_key, cache))
    needs_here = needs_geom(instances, state_here, tax, bench_roi=roi_here)
    needs_anywhere = needs_geom(instances, state_here, tax)
    needs_there = needs_geom(instances, state_there, tax, bench_roi=roi_there)
    rows = []
    for instance, changes in _grouped(state_there, state_here).items():
        if (only is not None and instance != only) or instance not in instances:
            continue
        if instance in needs_anywhere and instance not in needs_here:
            continue  # on the bench, and this view cannot see one: not its work
        if kind_for(changes, instance in needs_there,
                    instance in needs_here) != KIND_SPLIT_KEYFRAME:
            continue
        rows.append((instance, changes.get("state"),
                     has_split(db, tax, desktop, view, instance, step, neighbour, cache)))
    return rows


def reverse_walk(db: Db, desktop: int, view: str) -> list[int]:
    """The steps the reverse walk visits, ascending: the view's annotatable frames.

    Each one's card is diffed against the next one in this list -- which is
    what the session's ``task_neighbour`` answers in reverse order.
    """
    steps = [int(row["step"]) for row in db.frames_for(desktop, view)]
    return annotatable_steps(db, desktop, view, steps)


def open_split_steps(db: Db, tax: Taxonomy, desktop: int, view: str, instance: str,
                     low: Optional[int] = None, high: Optional[int] = None,
                     cache: Optional[InputCache] = None,
                     walk: Optional[Iterable[int]] = None) -> list[int]:
    """Frames ``low <= f < high`` whose card has an open ✂ row for ``instance``.

    Ascending. ``walk`` is :func:`reverse_walk` when the caller has it already.
    """
    cache = cache if cache is not None else InputCache()
    steps = list(walk) if walk is not None else reverse_walk(db, desktop, view)
    out = []
    for index, step in enumerate(steps[:-1]):
        if (low is not None and step < low) or (high is not None and step >= high):
            continue
        rows = split_rows_at(db, tax, desktop, view, step, steps[index + 1], cache,
                             only=instance)
        if any(not done for _inst, _transition, done in rows):
            out.append(step)
    return out


def stale_split_rows(db: Db, tax: Taxonomy, desktop: int,
                     view: str) -> list[tuple[int, str]]:
    """``(step, instance)`` of every open ✂ row on a confirmed frame.

    A frame confirmed before its state changed -- a clip set ``closed`` there
    with ``T`` after the frame was frozen with its open shape -- exports
    ``state: closed`` next to the open mask, and the only place that said so was
    its task card. This is the card's answer for every confirmed frame, so
    ``cli check`` counts exactly the rows an annotator standing on the frame
    would see -- and names the frame where ``Ctrl+K`` cuts the version.
    """
    cache = InputCache()
    steps = reverse_walk(db, desktop, view)
    out: list[tuple[int, str]] = []
    for index, step in enumerate(steps[:-1]):
        frame = db.get_frame(FrameKey(desktop, step, view)) or {}
        if frame.get("review_status") != _VERIFIED:
            continue
        for instance, _transition, done in split_rows_at(db, tax, desktop, view, step,
                                                         steps[index + 1], cache):
            if not done:
                out.append((step, str(instance)))
    return out
