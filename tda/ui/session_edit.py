"""What one annotation gesture writes, without Qt (spec 4.3 编辑语义).

:class:`tda.ui.session.AnnotationSession` owns "which frame is open" and the
signals the panels listen to; this module is everything it *writes*, as plain
functions over a :class:`~tda.core.db.Db` and a
:class:`~tda.core.truth.TruthService`.  That split keeps the session thin and
makes the rules below testable without a ``QApplication``.

The four scopes of spec 4.3 -- re-trace the keyframe, split it, override this
one frame, change the layering -- are :func:`commit_edit` and
:func:`commit_pair_override`; :func:`suggest_scope` is which of them a given
edit defaults to.  The operations they log, and the reads they share with
:mod:`tda.ui.session_tasks`, live in :mod:`tda.ui.session_ops`.

What every mutating function does
---------------------------------
1. Write the annotation inputs inside one
   :meth:`~tda.core.db.Db.transaction`, together with the ``op_log`` entry that
   records the change and its inverse (spec 4.6).
2. Work out which logical steps the change reaches: for a shape that is
   :meth:`~tda.core.truth.TruthService.affected_steps`, for a single-frame
   override it is this frame alone.
3. Recompile exactly those frames and report how many frozen rows disagreed.

They all return ``{"affected", "conflicts", "problems", "op"}``: the steps
actually recompiled, the number of queued conflicts, the compiler's problems per
step, and the :class:`~tda.ui.commands.Op` the session pushes onto its undo
stack.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

import numpy as np

from tda.core import masks
from tda.core.compiler import marked_absent, select_keyframe
from tda.core.cut_follow import CUT, is_cut, same_parts
from tda.core.db import Db
from tda.core.occluder import FULL as CABLE_FULL
from tda.core.occluder import PARTIAL as CABLE_PARTIAL
from tda.core.occluder import cable_label, kept_mask
from tda.core.model import (
    FrameKey,
    FrameOverride,
    OccluderMask,
    PairOverride,
    ShapeKeyframe,
    ShapePart,
    Visibility,
    ZOrderRec,
    is_component_view,
    machine_only,
)
from tda.core.truth import TruthService
from tda.core.split_rows import open_split_steps, reverse_walk, split_rows_at
from tda.core.truth_inputs import InputCache, frame_hw, pose_segment_of, state_of
from tda.core.zorder_floor import floor_index
from tda.ui import session_api as api
from tda.ui.class_names import state_zh
from tda.ui.session_api import SessionRefusal
from tda.ui.commands import Op
from tda.ui.session_ops import (
    DIRECTIONS,
    FORWARD,
    GEOM_BOX,
    GEOM_MASK,
    MAIN,
    REVERSE,
    annotatable_steps,
    apply_frame_override,
    apply_keyframes as _apply_keyframe_rows,
    apply_occluder,
    apply_pair_override,
    apply_zorder,
    as_mask,
    chain_for,
    chain_steps,
    default_anchor,
    frozen_among,
    keyframe_state,
    mask_steps,
    new_keyframe,
    occluder_union,
    override_state,
    placement_of,
    refresh_steps,
    segment_steps,
    settle,
    write_zorder,
)
# re-exported: the session and the tests reach the scope vocabulary through
# this module, which is the one import site for everything an edit needs
from tda.ui.session_scope import (  # noqa: F401
    SCOPE_SPLIT_PREFIX,
    SCOPE_ZORDER_ABOVE,
    SCOPE_ZORDER_BELOW,
    ZORDER_HINT_FRAC,
    hosts_of,
    split_zorder_scope,
    splits_the_shape,
    suggest_scope,
)
from tda.ui.session_tasks import item_text, task_card_for

__all__ = [
    "DIRECTIONS",
    "FORWARD",
    "REVERSE",
    "ZORDER_HINT_FRAC",
    "annotatable_steps",
    "apply_frame_override",
    "apply_keyframes",
    "apply_occluder",
    "apply_pair_override",
    "apply_zorder",
    "carried_here",
    "commit_box",
    "commit_into_previous",
    "commit_edit",
    "commit_occluder",
    "commit_pair_override",
    "default_anchor",
    "hosts_of",
    "plan_cuts",
    "preview",
    "preview_pair",
    "refresh_steps",
    "require_instance",
    "split_zorder_scope",
    "splits_the_shape",
    "clear_visibility",
    "set_visibility",
    "set_zorder_move",
    "suggest_scope",
    "version_behind",
    "item_text",
    "task_card_for",
]

#: The label 「看不见」 writes on a components-gallery part (task U10).
ABSENT_LABEL = Visibility.OUT_OF_VIEW.value


def _result(db: Db, truth: TruthService, key: FrameKey, steps: Sequence[int],
            op: Optional[Op], extra: Optional[dict] = None) -> dict:
    """Bring the truth table up to date with one edit, and report what it cost.

    Only the frame the annotator is looking at is compiled here.  The compiled
    rows of the *other* unverified frames in the interval are a cache of a pure
    function (spec 3.4): leaving them stale costs nothing, because every reader
    -- a visit, a batch refresh, an export -- recompiles what it needs, whereas
    compiling them now costs the annotator 300 ms per frame at scanner
    resolution and thirty times that on a chassis shape.

    The **frozen** frames are different -- confirmed, or demoted with signed
    rows still in them (:func:`~tda.ui.session_ops.frozen_among`): a frozen
    row is the one thing the compiler may not overwrite, so a disagreement has
    to be found and queued.
    That work is real, so it is handed to the background sweeper through the
    persisted queue rather than skipped.

    ``affected`` is the full reach of the edit (what "影响 N 帧" shows),
    ``compiled`` what was actually done now, ``rechecks`` what was deferred.
    """
    affected = annotatable_steps(db, key.desktop, key.view, steps)
    stats = settle(db, truth, key.desktop, key.view, affected, key.step)
    out = {"affected": affected, "compiled": stats["compiled"],
           "rechecks": stats["rechecks"], "conflicts": stats["conflicts"],
           "problems": stats["problems"], "frame": stats["frame"], "op": op}
    out.update(extra or {})
    return out


# --------------------------------------------------------------------------- #
# pixels
# --------------------------------------------------------------------------- #
def commit_edit(db: Db, truth: TruthService, key: FrameKey, instance: str,
                mask: np.ndarray, scope: str, direction: str = REVERSE,
                annotator: str = "system", pair: Optional[str] = None,
                extra: Optional[dict] = None) -> dict:
    """Write one pixel edit back with the scope the annotator chose (spec 4.3).

    ``scope`` is one of :data:`tda.ui.session_api.COMMIT_SCOPES`:

    ``keyframe``
        Re-trace the keyframe in force at this step, bumping its version, so the
        change reaches every frame of its interval.  When no keyframe applies
        yet the shape is created with the default anchor of
        :func:`~tda.ui.session_ops.default_anchor` and the instance is appended
        to the layer order, i.e. on top of everything but the floor layers
        (spec 4.2, :func:`_append_to_zorder`).  The window writes it for
        ``Ctrl+Enter`` (改全部), and for ``Enter`` where the version starts
        (task U16).
    ``split``
        A new version of the shape from this step on, in the browsing direction
        (spec 3.3): annotating in reverse the new keyframe is anchored at the
        current step, so it covers this frame and every earlier one while the
        old keyframe keeps the frames after it.  Going forward the old keyframe
        is pulled back to ``k-1`` and the new one inherits its anchor.  What
        ``Enter`` writes where the version in force is carried from frames the
        walk has passed (:func:`carried_here`, task U16).
    ``frame_override``
        Only this frame: the edited pixels, minus the frame's occluders, become
        the instance's visible mask here and nothing else changes.

    ``pair`` names the instance this one is put *above* in the same gesture: a
    layering answer whose added pixels are not in this instance's shape yet has
    to write both, or the annotator's paint is silently dropped (spec 4.3).

    Raises ``ValueError`` for an unknown scope or direction, or for a mask that
    is not in the frame's coordinates.
    """
    if direction not in DIRECTIONS:
        raise SessionRefusal(
            f"direction must be one of {DIRECTIONS}, got {direction!r}")
    cache = InputCache()
    hw = frame_hw(db, key, truth.cache_dir)
    edited = as_mask(mask, hw)
    if scope == api.SCOPE_FRAME_OVERRIDE:
        return _commit_frame_override(db, truth, key, instance, edited, hw, annotator,
                                      extra)
    if scope not in (api.SCOPE_KEYFRAME, api.SCOPE_SPLIT):
        raise SessionRefusal(f"unknown commit scope {scope!r}")
    # The mask comes off the canvas overlay, so it is C-ordered and encoding it
    # means transposing a 12 MP canvas -- 40 ms of every commit on an OAK frame.
    # Its bounding box is measured here rather than trusted (0.5 ms) and the
    # encode then only touches what the box holds.
    parts = [ShapePart(MAIN, masks.encode_rle_boxed(edited))]
    return _commit_shape(db, truth, key, instance, parts, GEOM_MASK, scope, direction,
                         cache, annotator, pair=pair, extra=extra)


def commit_box(db: Db, truth: TruthService, key: FrameKey, instance: str,
               box: Sequence[float], scope: str = api.SCOPE_KEYFRAME,
               direction: str = REVERSE, annotator: str = "system") -> dict:
    """Draw the staging-area rectangle of a part on the bench (spec 4.2 S4).

    Bench geometry is a box rather than a mask: the part is tracked only well
    enough to say that it left the chassis and where it went, so the keyframe
    carries one ``box`` part and takes no place in the layer order -- a bench box
    neither occludes anything nor is occluded (spec 3.3 step 5).
    """
    parts = [ShapePart(MAIN, None, tuple(float(v) for v in box))]
    return _commit_shape(db, truth, key, instance, parts, GEOM_BOX, scope, direction,
                         InputCache(), annotator)


def _commit_shape(db: Db, truth: TruthService, key: FrameKey, instance: str,
                  parts: list[ShapePart], geom_type: str, scope: str, direction: str,
                  cache: InputCache, annotator: str,
                  pair: Optional[str] = None,
                  extra: Optional[dict] = None,
                  dropped: Optional[tuple[dict, dict]] = None,
                  cut_from: Optional[ShapeKeyframe] = None) -> dict:
    """The shared body of :func:`commit_edit` and :func:`commit_box`.

    ``pair`` is the instance this one is to be put *above* in the same gesture
    (spec 4.3 改层级 with new pixels): the ``PairOverride`` is written in the
    same transaction and travels in the same op, so one ``Ctrl+Z`` takes back
    both halves of what was one decision.

    ``dropped`` -- ``(before, after)`` keyframe states -- is a keyframe the
    caller deleted in its own enclosing transaction just before this commit
    (:func:`commit_into_previous`): it goes into the same op, so the undo
    puts it back with everything else.  ``cut_from`` is that keyframe: the
    frames a cut below keeps were showing *its* shape, not the version the
    re-trace writes into (U16 review, R1), so the copy is made of it.
    """
    seg = pose_segment_of(db, key, cache)
    placement = placement_of(db, truth.tax, key, instance, cache)
    chain = chain_for(db, key, instance, seg, placement)
    chosen = select_keyframe(chain, key.step)
    # a shape commit never crosses an open state boundary (U8 round 3)
    cuts = plan_cuts(db, truth.tax, key, instance, chain, chosen, scope, direction,
                     geom_type, cache)
    if cuts["refuse"]:
        raise SessionRefusal(cuts["refuse"])
    source, draft_id = _adoption(extra)
    ref: dict = {"keyframe_id": None}
    old_ref: dict = {"keyframe_id": None if chosen is None else chosen.id}
    before: list[dict] = [] if dropped is None else [dropped[0]]
    after: list[dict] = [] if dropped is None else [dropped[1]]

    no_split = scope == api.SCOPE_SPLIT and _cannot_split(chosen, key.step, direction,
                                                          chain)
    if no_split:
        # there is nothing left to cut off: the keyframe in force already ends
        # exactly here, so "split" means "re-trace this one" (see _cannot_split).
        # The result says so (``no_split``): the window used to stay silent and
        # the D13 annotator believed a split had been made (task U6).
        scope = api.SCOPE_KEYFRAME

    with db.transaction():
        cut_kf = None
        if cuts["below"] is not None:
            # the old shape keeps the frames of the other state: the version in
            # force is cut at the boundary, pixels unchanged, in this same op
            cut_ref: dict = {"keyframe_id": None}
            cut_kf = _copy_keyframe(chosen if cut_from is None else cut_from, key,
                                    cuts["below"])
            before.append(keyframe_state(None, cut_ref, template=cut_kf))
            cut_ref["keyframe_id"] = db.add_keyframe(cut_kf)
            after.append(keyframe_state(cut_kf, cut_ref))
        # the forward walk's two (U29): ``ahead`` is where the edit stops, the
        # frames of the next state keeping the shape in force through a cut copy
        # at its anchor -- the reverse walk's cut below, mirrored; ``behind``
        # (a re-trace) is where the version in force is pulled back to, so the
        # frames of the state behind keep it -- the reverse walk's ``top``
        stop = cuts["ahead"]
        if stop is not None:
            ahead_ref: dict = {"keyframe_id": None}
            cut_kf = _copy_keyframe(chosen, key, int(chosen.anchor_step))
            before.append(keyframe_state(None, ahead_ref, template=cut_kf))
            ahead_ref["keyframe_id"] = db.add_keyframe(cut_kf)
            after.append(keyframe_state(cut_kf, ahead_ref))
        if cuts["behind"] is not None:
            before.append(keyframe_state(chosen, old_ref))
            anchor = int(chosen.anchor_step) if stop is None else stop
            chosen.anchor_step = cuts["behind"]
            db.update_keyframe(chosen)
            after.append(keyframe_state(chosen, old_ref))
            kf = new_keyframe(key, instance, seg, anchor, placement, parts, geom_type)
            _stamp_source(kf, source, draft_id)
            kf.version = int(chosen.version) + 1
            before.append(keyframe_state(None, ref, template=kf))
            ref["keyframe_id"] = db.add_keyframe(kf)
            after.append(keyframe_state(kf, ref))
        elif cuts["top"] is not None:
            # Enter inside a state run whose top is an open boundary: the edit
            # is the run's own version, the frames above keep the one in force
            kf = new_keyframe(key, instance, seg, cuts["top"], placement, parts, geom_type)
            _stamp_source(kf, source, draft_id)
            kf.version = int(chosen.version) + 1
            before.append(keyframe_state(None, ref, template=kf))
            ref["keyframe_id"] = db.add_keyframe(kf)
            after.append(keyframe_state(kf, ref))
        elif scope == api.SCOPE_SPLIT:
            anchor = key.step
            if direction == FORWARD and chosen is not None:
                before.append(keyframe_state(chosen, old_ref))
                anchor = chosen.anchor_step if stop is None else stop
                chosen.anchor_step = key.step - 1
                db.update_keyframe(chosen)
                after.append(keyframe_state(chosen, old_ref))
            kf = new_keyframe(key, instance, seg, anchor, placement, parts, geom_type)
            _stamp_source(kf, source, draft_id)
            # a new version of the shape has to outrank the one it was cut from:
            # select_keyframe breaks an anchor tie on the higher version
            kf.version = 1 if chosen is None else int(chosen.version) + 1
            before.append(keyframe_state(None, ref, template=kf))
            ref["keyframe_id"] = db.add_keyframe(kf)
            after.append(keyframe_state(kf, ref))
        elif chosen is not None and chosen.geom_type == geom_type:
            ref["keyframe_id"] = chosen.id
            before.append(keyframe_state(chosen, ref))
            # a cut copy stays one only while its pixels are the copy (U11)
            still_copy = is_cut(chosen) and same_parts(chosen.parts, parts)
            chosen.parts = parts
            if stop is not None:
                chosen.anchor_step = stop        # the copy holds the frames ahead
            if not still_copy:
                _stamp_source(chosen, source, draft_id)
            db.update_keyframe(chosen)
            after.append(keyframe_state(chosen, ref))
            kf = chosen
        else:
            anchor = default_anchor(db, truth.tax, key, instance, seg, placement, cache)
            kf = new_keyframe(key, instance, seg, anchor, placement, parts, geom_type)
            _stamp_source(kf, source, draft_id)
            before.append(keyframe_state(None, ref, template=kf))
            ref["keyframe_id"] = db.add_keyframe(kf)
            after.append(keyframe_state(kf, ref))

        zorder = None if geom_type == GEOM_BOX else _append_to_zorder(db, key, seg, instance)
        edit_reach = set(truth.affected_steps(key.desktop, key.view, instance, kf))
        reach = edit_reach | {key.step}
        if cut_kf is not None:
            # same pixels, but another keyframe is selected there: re-checked
            reach |= set(truth.affected_steps(key.desktop, key.view, instance, cut_kf))
        pair_state = None
        if pair is not None:
            po = PairOverride(key.desktop, key.view, seg, instance, pair)
            existed = any(p == po for p in db.pair_overrides(key.desktop, key.view, seg))
            db.set_pair_override(po)
            # exactly the frames preview_pair promised, on top of the shape's own
            reach |= set(pair_steps(db, truth.tax, key, instance, pair, seg, cache))
            pair_state = {"pose_segment": seg, "above": instance, "below": pair,
                          "existed": existed}
        steps = annotatable_steps(db, key.desktop, key.view, reach)
        # a shape drawn on a gallery part marked 「看不见」 says it IS in this
        # shot (task U10): the label goes, in this same op
        absent = (is_component_view(key.view)
                  and marked_absent(db.frame_overrides(key).get(instance)))
        if absent:
            db.delete_frame_override(key, instance)
        common = {"desktop": key.desktop, "view": key.view, "step": key.step,
                  "instance": instance, "scope": scope, "direction": direction,
                  "steps": steps}
        payload = _with_extra(common, extra) | {"keyframes": after,
                            "zorder": None if zorder is None else zorder["after"],
                            "pair": _pair_payload(pair_state, True)}
        inverse = common | {"keyframes": list(reversed(before)),
                            "zorder": None if zorder is None else zorder["before"],
                            "pair": _pair_payload(pair_state, False)}
        if absent:
            payload["label"] = {"step": key.step, "visibility": None}
            inverse["label"] = {"step": key.step, "visibility": ABSENT_LABEL}
        db.log_op(key.desktop, key.view, "commit_keyframe",
                  _loggable(payload), _loggable(inverse), annotator)

    op = Op(kind="commit_keyframe", payload=payload, inverse=inverse)
    return _result(db, truth, key, steps, op,
                   {"keyframe_id": kf.id, "anchor_step": kf.anchor_step, "scope": scope,
                    "changed": True, "no_split": bool(no_split),
                    "cut_below": cuts["below"], "cut_top": cuts["top"],
                    "cut_ahead": cuts["ahead"], "cut_behind": cuts["behind"],
                    "cleared_absent": bool(absent),
                    "edit_reach": annotatable_steps(db, key.desktop, key.view,
                                                    edit_reach | {key.step})})


# --------------------------------------------------------------------------- #
# a shape commit never crosses an open state boundary (U8 round 3)
# --------------------------------------------------------------------------- #
def _has_pixels(kf: Optional[ShapeKeyframe]) -> bool:
    """Is there a shape at all? A cut must never mint an empty keyframe."""
    if kf is None:
        return False
    for part in kf.parts or ():
        if part.box is not None:
            return True
        if part.rle and masks.rle_area(part.rle) > 0:
            return True
    return False


def _copy_keyframe(chosen: ShapeKeyframe, key: FrameKey, anchor: int) -> ShapeKeyframe:
    """``chosen``'s shape as a new version anchored at ``anchor`` (pixels unchanged)."""
    parts = [ShapePart(p.name, None if p.rle is None else dict(p.rle),
                       None if p.box is None else tuple(p.box)) for p in chosen.parts]
    kf = new_keyframe(key, chosen.instance, chosen.pose_segment, anchor, chosen.placement,
                      parts, chosen.geom_type)
    kf.version = int(chosen.version) + 1
    kf.amodal_complete = chosen.amodal_complete
    # marked as the copy it is, so it can follow its boundary (U11); the draft
    # reference of the shape it was cut from goes with the pixels
    _stamp_source(kf, CUT, chosen.draft_id)
    return kf


def plan_cuts(db: Db, tax, key: FrameKey, instance: str, chain: list[ShapeKeyframe],
              chosen: Optional[ShapeKeyframe], scope: str, direction: str,
              geom_type: str, cache: Optional[InputCache] = None) -> dict:
    """Where a shape commit at ``key`` has to stop (U8 round 3).

    **A shape commit never crosses an open state boundary.** The frames the
    written keyframe would select are looked through for open ✂ rows of this
    instance (:func:`tda.core.split_rows.open_split_steps`: the state changes
    between a frame and the next one of the walk, and one version is in force
    on both sides):

    * ``below`` -- the highest such frame under the current one: the version in
      force is cut there first, pixels unchanged (either scope);
    * ``top`` -- the re-trace only (``keyframe``: ``Ctrl+Enter``, or ``Enter``
      where the version starts), whose reach runs up to the anchor: the lowest
      such frame above the current one. The edit is written as a version
      anchored there instead of re-tracing the one in force, so the frames of
      the next state keep their shape;
    * ``refuse`` -- the re-trace on a frame that is itself an open ✂ row: it
      would rewrite the other state's frames, and ``Enter`` there is the split
      (task U16).

    **The forward walk mirrors it** (task U29; ``below`` and ``top`` stay
    ``None`` there). Going forward a commit writes from this frame towards the
    *later* steps, so:

    * ``ahead`` -- the lowest open ✂ at or above the current frame inside the
      reach (the boundary between it and the next frame of the walk): the edit
      stops there, and the frames of the next state keep the shape in force
      through a cut copy at its anchor (either scope) -- what ``below`` is in
      the reverse walk, a placeholder on the side the walk has not reached;
    * ``behind`` -- the re-trace only: the highest such boundary under the
      current frame. The version in force is pulled back to it, so the frames
      of the state behind keep their shape, and the edit is a new version from
      the boundary on -- what ``top`` is in the reverse walk;
    * ``refuse`` -- the re-trace on a frame whose own (forward) card has the
      open ✂ row, the boundary right behind it: the same sentence, naming the
      frames behind it would rewrite.

    Nothing is cut when no non-empty mask is in force (the part is drawn here
    for the first time, or by a box). Returns ``{"below", "top", "ahead",
    "behind", "refuse"}``.
    """
    out: dict = {"below": None, "top": None, "ahead": None, "behind": None, "refuse": ""}
    if scope not in (api.SCOPE_KEYFRAME, api.SCOPE_SPLIT) or direction not in DIRECTIONS:
        return out
    if geom_type != GEOM_MASK or chosen is None or chosen.geom_type != GEOM_MASK \
            or not _has_pixels(chosen):
        return out
    lower = [kf.anchor_step for kf in chain if kf.anchor_step < key.step]
    low = max(lower) + 1 if lower else None
    re_trace = scope == api.SCOPE_KEYFRAME or _cannot_split(chosen, key.step, direction,
                                                            chain)
    walk = reverse_walk(db, key.desktop, key.view)
    if direction == FORWARD:
        return _plan_forward(db, tax, key, instance, chosen, low, re_trace, scope, walk,
                             cache, out)
    high = int(chosen.anchor_step) if re_trace else int(key.step)
    # only boundaries of this chain's pose segment: a cut anchored in another
    # segment is a version no frame of this one ever selects
    opened = [f for f in open_split_steps(db, tax, key.desktop, key.view, instance, low,
                                          high, cache, walk)
              if pose_segment_of(db, FrameKey(key.desktop, f, key.view), cache)
              == chosen.pose_segment]
    below = [f for f in opened if f < key.step]
    above = [f for f in opened if f >= key.step]
    if below:
        out["below"] = below[-1]
    if re_trace and above:
        if above[0] == key.step and scope == api.SCOPE_KEYFRAME:
            later = [s for s in walk if s > key.step]
            there = later[0] if later else None
            # the run the named state holds over: up to the next boundary (N5)
            out["refuse"] = _across(db, tax, key, instance, there, there,
                                    above[1] if len(above) > 1 else chosen.anchor_step, cache)
        else:
            out["top"] = above[0]
    return out


#: 改全部 on a frame that is itself a state boundary of the version in force.
ACROSS_BOUNDARY = ("这一帧是状态边界（✂）：改全部会把第 {first}–{last} 步{what}"
                   "的形状一起改掉。用 Q（从这帧起新版本）")


def _across(db: Db, tax, key: FrameKey, instance: str, there: Optional[int], first,
            last, cache: Optional[InputCache]) -> str:
    """:data:`ACROSS_BOUNDARY` for the frames from ``first`` to ``last``, in ``there``'s state."""
    held = (None if there is None else
            state_of(db, tax, key.desktop, there, cache).get(instance))
    what = "" if held is None else f"（{state_zh(held.state)}）"
    return ACROSS_BOUNDARY.format(first=first, last=last, what=what)


def _plan_forward(db: Db, tax, key: FrameKey, instance: str, chosen: ShapeKeyframe,
                  low: Optional[int], re_trace: bool, scope: str, walk: list[int],
                  cache: Optional[InputCache], out: dict) -> dict:
    """:func:`plan_cuts` for the forward walk (task U29): ``ahead``, ``behind``, ``refuse``.

    The reach of the written shape is ``key.step``..anchor for a split (the
    version in force is pulled back to the frame before), the version's whole
    span for a re-trace; the boundaries inside it are the frames ``f`` whose
    reverse card has an open ✂ row -- the state changes between ``f`` and the
    next frame of the walk with one version on both sides -- which is the
    question the reverse walk asks too, so the two walks cut at the same
    places.
    """
    seg = chosen.pose_segment
    opened = [f for f in open_split_steps(db, tax, key.desktop, key.view, instance,
                                          low if re_trace else key.step,
                                          int(chosen.anchor_step), cache, walk)
              if pose_segment_of(db, FrameKey(key.desktop, f, key.view), cache) == seg]
    ahead = [f for f in opened if f >= key.step]
    behind = [f for f in opened if f < key.step]
    if ahead:
        out["ahead"] = ahead[0]
    if not (re_trace and behind):
        return out
    earlier = [s for s in walk if s < key.step]
    if scope == api.SCOPE_KEYFRAME and earlier and behind[-1] == earlier[-1]:
        # the boundary is right behind this frame: its forward card's ✂ row
        # the run the named state holds over: from the boundary before it (N5)
        span = [s for s in earlier if (low is None or s >= low)
                and (len(behind) < 2 or s > behind[-2])
                and pose_segment_of(db, FrameKey(key.desktop, s, key.view), cache) == seg]
        first = span[0] if span else earlier[-1]
        out["refuse"] = _across(db, tax, key, instance, earlier[-1], first, earlier[-1],
                                cache)
        out["ahead"] = None
        return out
    out["behind"] = behind[-1]
    return out


def _stamp_source(kf: ShapeKeyframe, source: str, draft_id: Optional[int]) -> None:
    """Say where the pixels this commit writes came from -- every time.

    Including "from nobody but the annotator", which is what makes the pair
    honest: the row describes the commit that last wrote it, not the first one
    that ever did.
    """
    kf.source = source
    kf.draft_id = draft_id


def _pair_payload(state: Optional[dict], forward: bool) -> Optional[dict]:
    """The ``pair`` half of a ``commit_keyframe`` payload, either way round."""
    if state is None:
        return None
    return {"pose_segment": state["pose_segment"], "above": state["above"],
            "below": state["below"],
            "exists": True if forward else bool(state["existed"])}


def apply_keyframes(db: Db, truth: TruthService, payload: dict,
                    current: Optional[int] = None) -> dict:
    """:func:`tda.ui.session_ops.apply_keyframes`, plus the pair a commit carried.

    A layering commit that also re-traced the shape is one op with two writes
    (see :func:`_commit_shape`), so undo and redo have to move both; the shared
    ``steps`` cover the union, so one settle brings the frames up to date.

    **One transaction covers both halves.**  With the pair in a block of its own
    a failure in between left the shape carrying pixels that nothing painted
    over any more -- a state no point of the history describes.  The block below
    is the outermost one (:meth:`tda.core.dbconn.ConnectionMixin.transaction` is
    re-entrant), so the inner writes commit with it or not at all.
    """
    pair = payload.get("pair")
    label = payload.get("label")
    if pair is None and label is None:
        return _apply_keyframe_rows(db, truth, payload, current)
    with db.transaction():
        if pair is not None:
            po = PairOverride(payload["desktop"], payload["view"], pair["pose_segment"],
                              pair["above"], pair["below"])
            if pair["exists"]:
                db.set_pair_override(po)
            else:
                db.delete_pair_override(po)
        if label is not None:
            # the 「看不见」 label a gallery commit took back (task U10)
            key = FrameKey(payload["desktop"], int(label["step"]), payload["view"])
            if label["visibility"] is None:
                db.delete_frame_override(key, payload["instance"])
            else:
                db.set_frame_override(FrameOverride(key, payload["instance"], None,
                                                    label["visibility"]))
        return _apply_keyframe_rows(db, truth, payload, current)


def preview(db: Db, truth: TruthService, key: FrameKey, instance: str, scope: str,
            direction: str = REVERSE, geom_type: str = GEOM_MASK) -> dict:
    """Which frames an edit *would* reach, without writing or compiling anything.

    This is the "影响 N 帧 / 将产生 N 个冲突" strip of spec 4.3: the annotator has
    to see how far a change carries -- and how many frozen frames it will
    disturb -- **before** deciding on the scope.  The chain is simulated rather
    than written, and no mask is touched, so the answer costs a few state
    lookups.

    Returns ``{"steps", "verified_steps"}``; the second is the subset a human
    has frozen -- the frame confirmed, or any row of it signed
    (:func:`~tda.ui.session_ops.frozen_among`) -- i.e. the frames that would go
    to the conflict queue.
    """
    cache = InputCache()
    seg = pose_segment_of(db, key, cache)
    placement = placement_of(db, truth.tax, key, instance, cache)
    chain = chain_for(db, key, instance, seg, placement)
    chosen = select_keyframe(chain, key.step)

    if scope == api.SCOPE_FRAME_OVERRIDE:
        steps = annotatable_steps(db, key.desktop, key.view, [key.step])
    else:
        # the reach *after* the cuts the commit would make (U8 round 3): a
        # re-trace inside a state run reaches that run, not the version's span
        cuts = plan_cuts(db, truth.tax, key, instance, chain, chosen, scope, direction,
                         geom_type, cache)
        if cuts["below"] is not None:
            chain = chain + [_copy_keyframe(chosen, key, cuts["below"])]
        stop = cuts["ahead"]
        if stop is not None:
            chain = chain + [_copy_keyframe(chosen, key, int(chosen.anchor_step))]
        others = [kf for kf in chain if kf is not chosen]
        if cuts["behind"] is not None:
            # forward re-trace (U29): the version in force pulled back, a new one on
            target = new_keyframe(key, instance, seg,
                                  int(chosen.anchor_step) if stop is None else stop,
                                  placement, [], geom_type)
            target.version = int(chosen.version) + 1
            simulated = others + [replace(chosen, anchor_step=cuts["behind"]), target]
        elif cuts["top"] is not None:
            target = new_keyframe(key, instance, seg, cuts["top"], placement, [], geom_type)
            target.version = int(chosen.version) + 1
            simulated = chain + [target]
        elif scope == api.SCOPE_SPLIT and not _cannot_split(chosen, key.step, direction,
                                                            chain):
            if direction == FORWARD and chosen is not None:
                # what the commit writes: the old version pulled back to k-1,
                # the new one from here to its anchor (or to the boundary ahead)
                target = new_keyframe(key, instance, seg,
                                      int(chosen.anchor_step) if stop is None else stop,
                                      placement, [], geom_type)
                target.version = int(chosen.version) + 1
                simulated = others + [replace(chosen, anchor_step=key.step - 1), target]
            else:
                target = new_keyframe(key, instance, seg, key.step, placement, [], geom_type)
                target.version = 1 if chosen is None else int(chosen.version) + 1
                simulated = chain + [target]
        elif chosen is not None and chosen.geom_type == geom_type:
            if stop is not None:
                target = replace(chosen, anchor_step=stop)
                simulated = others + [target]
            else:
                target, simulated = chosen, chain
        else:
            anchor = default_anchor(db, truth.tax, key, instance, seg, placement, cache)
            target = new_keyframe(key, instance, seg, anchor, placement, [], geom_type)
            simulated = chain + [target]
        steps = chain_steps(db, truth.tax, key, instance, seg, placement, simulated,
                            target, cache)
    # frozen, not only flagged: a demoted frame's signed rows conflict too (U7)
    return {"steps": steps,
            "verified_steps": frozen_among(db, key.desktop, key.view, steps)}


def carried_here(db: Db, truth: TruthService, key: FrameKey, instance: str,
                 direction: str = REVERSE, geom_type: str = GEOM_MASK) -> bool:
    """Is the version in force here also in force on a frame the walk has passed?

    Task U16. The annotator walks the teardown backwards and, changing a shape
    on frame k, means "from this frame on": the frames behind them -- already
    looked at, often confirmed -- are to stay as they are. ``Enter`` re-traced
    the version in force wherever it applied, and on the trial that reached
    the confirmed frames behind and turned them red. So ``Enter`` cuts a new
    version here (:data:`~tda.ui.session_api.SCOPE_SPLIT`) exactly when this
    answers ``True``: the version is *carried* -- some annotatable frame behind
    this one in ``direction`` (later steps in the reverse walk, earlier ones
    going forward), in this pose segment, needs this part's geometry and
    selects the same keyframe. Where the version starts -- nothing behind
    carries it -- a split would cut off nothing, and ``Enter`` re-traces it.

    ``False`` when no keyframe of this geometry applies (a part drawn here for
    the first time takes its default anchor, which is not "from here on").
    Writes nothing; reverse-walk answers with the version anchored here cost
    one query.
    """
    cache = InputCache()
    seg = pose_segment_of(db, key, cache)
    placement = placement_of(db, truth.tax, key, instance, cache)
    chain = chain_for(db, key, instance, seg, placement)
    chosen = select_keyframe(chain, key.step)
    if chosen is None or chosen.geom_type != geom_type:
        return False
    if direction == REVERSE and int(chosen.anchor_step) <= int(key.step):
        return False        # it ends exactly here: nothing behind carries it
    steps = chain_steps(db, truth.tax, key, instance, seg, placement, chain, chosen, cache)
    if direction == REVERSE:
        return any(s > key.step for s in steps)
    return any(s < key.step for s in steps)


#: 「改全部」 where this frame's version is separated from the previous one by a
#: state boundary (U16 review, M1): the frames behind are the other state's.
INTO_PREVIOUS_ACROSS_BOUNDARY = (
    "这一帧和后面第 {there} 步之间是状态边界（✂）：后面的帧是{what}的样子，"
    "不能把这一版写过去")
#: ... its tail when pixels wait in the layer: the key that keeps them (R2).
REDRAW_WITH_ENTER = "；只重画这一版用 Q"
#: ... and where there is no previous version behind to write into.
NOTHING_BEHIND = "后面的帧用的就是这一版（或没有要它形状的帧）：用 Q"


def version_behind(db: Db, truth: TruthService, key: FrameKey, instance: str,
                   direction: str = REVERSE, cache: Optional[InputCache] = None
                   ) -> Optional[dict]:
    """The previous version the frames behind still use, when this one starts here.

    U16 review, M1. After ``Enter``'s split on a carried frame k the version
    in force *starts* at k, so 改全部 there re-traced only that version and
    the frames behind -- the ones the window just named, "要一起改用
    Ctrl+Enter" -- never changed. This is the other answer ``Ctrl+Enter`` gives
    on such a frame: the version in force starts here (nothing behind carries
    it, :func:`carried_here`) and the frames behind in the walk are on the
    *previous* version -- the next keyframe up the chain, however this one came
    about (a split, or a first draw with the part carried behind).

    ``None`` when that is not so: the version is carried (改全部 re-traces
    it), no previous version is behind, a box, the forward walk (deleting this
    frame's version would not hand it to the frame behind there; no window
    path browses forward). Otherwise ``{"here", "behind", "steps", "frozen",
    "refuse"}``: the two keyframes; the frames behind that would take the
    shape -- the previous version's frames up to the first open ✂ boundary
    above (where the write stops, as a re-trace does, U8 round 3) -- and the
    frozen ones among them; and the session's refusal when a state boundary
    lies between this frame and the next one behind (``""`` when none).
    """
    if direction != REVERSE or is_component_view(key.view):
        return None
    cache = cache if cache is not None else InputCache()
    seg = pose_segment_of(db, key, cache)
    placement = placement_of(db, truth.tax, key, instance, cache)
    chain = chain_for(db, key, instance, seg, placement)
    here = select_keyframe(chain, key.step)
    if here is None or here.geom_type != GEOM_MASK:
        return None
    if carried_here(db, truth, key, instance, direction):
        return None
    behind = select_keyframe(chain, int(here.anchor_step) + 1)
    if behind is None or behind.geom_type != GEOM_MASK:
        return None
    rest = [kf for kf in chain if kf is not here]
    reach = chain_steps(db, truth.tax, key, instance, seg, placement, rest, behind, cache)
    steps = [s for s in reach if s > key.step]
    if not steps:
        return None
    walk = reverse_walk(db, key.desktop, key.view)
    opened = [f for f in open_split_steps(db, truth.tax, key.desktop, key.view, instance,
                                          key.step + 1, int(behind.anchor_step), cache,
                                          walk)
              if pose_segment_of(db, FrameKey(key.desktop, f, key.view), cache) == seg]
    if opened:
        steps = [s for s in steps if s <= opened[0]]
    later = [s for s in walk if s > key.step]
    refuse = ""
    if later and split_rows_at(db, truth.tax, key.desktop, key.view, key.step, later[0],
                               cache, only=instance):
        held = state_of(db, truth.tax, key.desktop, later[0], cache).get(instance)
        refuse = INTO_PREVIOUS_ACROSS_BOUNDARY.format(
            there=later[0], what="" if held is None else f"「{state_zh(held.state)}」")
    return {"here": here, "behind": behind, "steps": steps,
            "frozen": frozen_among(db, key.desktop, key.view, steps), "refuse": refuse}


def commit_into_previous(db: Db, truth: TruthService, key: FrameKey, instance: str,
                         mask: np.ndarray, direction: str = REVERSE,
                         annotator: str = "system", extra: Optional[dict] = None,
                         painted: bool = False) -> dict:
    """「把这一版也写到后面的帧」: this frame's version folds into the previous one.

    U16 review, M1 -- what ``Ctrl+Enter`` does where :func:`version_behind`
    answers: the shape (the editing layer: the caller passes the stored one
    when nothing was painted) is written into the previous version's keyframe,
    every part, and this frame's own version is deleted so the frame falls back
    to it. That is a re-trace of the previous version from here (the same
    :func:`_commit_shape` path, cuts at open state boundaries included), with
    the deleted keyframe in the same op: one ``Ctrl+Z`` restores both. A cut
    below is a copy of the deleted version -- what those frames were showing
    (R1). The previous version's frozen frames are re-checked as any
    re-trace's are. Refused across a state boundary -- with the other key
    named when ``painted`` pixels wait in the layer (R2) -- and where there is
    nothing behind.
    """
    found = version_behind(db, truth, key, instance, direction)
    if found is None:
        raise SessionRefusal(NOTHING_BEHIND)
    if found["refuse"]:
        raise SessionRefusal(found["refuse"] + (REDRAW_WITH_ENTER if painted else ""))
    here = found["here"]
    edited = as_mask(mask, frame_hw(db, key, truth.cache_dir))
    parts = [ShapePart(MAIN, masks.encode_rle_boxed(edited))]
    ref: dict = {"keyframe_id": here.id}
    dropped = (keyframe_state(here, ref), keyframe_state(None, ref, template=here))
    note = dict(extra or {}) | {"into_previous": int(here.anchor_step)}
    with db.transaction():
        # inside one transaction: a refusal below rolls the delete back too
        db.delete_keyframe(here.id)
        result = _commit_shape(db, truth, key, instance, parts, GEOM_MASK,
                               api.SCOPE_KEYFRAME, direction, InputCache(), annotator,
                               extra=note, dropped=dropped, cut_from=here)
    result["into_previous"] = list(found["steps"])
    return result


def pair_steps(db: Db, tax, key: FrameKey, one: str, two: str, seg: int,
               cache: Optional[InputCache] = None) -> list[int]:
    """Steps where both instances are mask layers, so a pair override can bite.

    The one definition, so what :func:`preview_pair` promises and what
    :func:`commit_pair_override` recompiles cannot drift apart.
    """
    mine = set(mask_steps(db, tax, key, one, seg, cache))
    theirs = set(mask_steps(db, tax, key, two, seg, cache))
    return sorted(mine & theirs)


def preview_pair(db: Db, truth: TruthService, key: FrameKey, instance: str,
                 other: str) -> dict:
    """Which frames a layering exception would reach (spec 4.3 改层级).

    A ``PairOverride`` holds for a whole pose segment, but it can only change a
    pixel where the two instances are layers of the same group: both need a
    *mask*, in the same placement.  A part lying on the bench is tracked by a
    rectangle and is no layer at all (spec 3.3 step 5), so those steps are not
    reached however long either keyframe chain runs -- which is what a
    scope-blind preview used to report.
    """
    cache = InputCache()
    seg = pose_segment_of(db, key, cache)
    steps = pair_steps(db, truth.tax, key, instance, other, seg, cache)
    return {"steps": steps,
            "verified_steps": frozen_among(db, key.desktop, key.view, steps)}


def _cannot_split(chosen: Optional[ShapeKeyframe], step: int, direction: str,
                  chain: Sequence[ShapeKeyframe] = ()) -> bool:
    """Would a split produce a keyframe covering no frame the old one kept?

    Annotating in reverse a split anchors the new shape at the current step, so
    it covers ``(previous anchor, step]`` and the old one keeps everything after
    ``step``.  When the old keyframe's own anchor *is* ``step`` there is nothing
    after it to keep: the two would share an anchor and the newer one would win
    everywhere, which is a re-trace wearing a second row.  Going forward the same
    thing happens at the other end -- the old keyframe would be pulled back to
    ``step - 1`` and inherit nothing.

    That forward case is read off ``chain`` (U16 review, M2): when the
    keyframe below the one in force is anchored at ``step - 1`` the pulled-back
    one would land on that anchor and, a version newer, replace it on every
    frame it covered -- with nothing re-checked.  Re-traced instead.
    """
    if chosen is None:
        return False
    if direction == REVERSE:
        return int(chosen.anchor_step) == int(step)
    lower = [int(kf.anchor_step) for kf in chain
             if kf is not chosen and int(kf.anchor_step) < int(step)]
    return bool(lower) and max(lower) == int(step) - 1


#: ``ShapeKeyframe.source`` of a shape the annotator took from a Label Studio
#: draft (spec 3.1 来源).  Deliberately **not** ``labelstudio``: that value marks
#: the importer's own untouched draft rows, and everything that acts on them --
#: the ``import-ls`` purge (:func:`tda.core.ls_import._purge`), the draft-key
#: delete rule in :mod:`tda.ui.steps_delete` -- matches it exactly.  An adopted
#: shape is the annotator's work on a real instance key and must survive all of
#: them.
SOURCE_ADOPTED = "ls_adopted"
#: ``ShapeKeyframe.source`` of a shape the annotator drew themselves.
SOURCE_MANUAL = "manual"


def _adoption(extra: Optional[dict]) -> tuple[str, Optional[int]]:
    """``(source, draft_id)`` of the commit that is about to write the shape.

    Spec 3.1 gives a keyframe a 来源 and a 初稿引用, and a shape that started as
    somebody's old polygon should say so in its own row rather than only in the
    op log.  The draft with the **largest overlap** is the one the row points
    at: a mask can be built from two drafts, but ``draft_id`` is one column.

    Both fields describe the **last** commit that wrote the row, so a later
    hand-drawn commit takes them back to ``manual``/``NULL``.  Leaving them
    alone instead made ``ls_adopted`` permanent: a shape re-traced from
    scratch months later still pointed at a draft it no longer had a pixel in
    common with.  Neither value takes part in ``input_hash``
    (:func:`tda.core.compiler_visibility.input_hash` hashes a keyframe as
    id + version + content digest), so this cannot move a compiled row on its
    own, and both travel in the undo payload so a Ctrl+Z restores them exactly.
    """
    entries = [e for e in ((extra or {}).get("adopted") or []) if isinstance(e, dict)]
    if not entries:
        return (SOURCE_MANUAL, None)
    best = max(entries, key=lambda e: int(e.get("overlap_px") or 0))
    draft_id = best.get("keyframe_id")
    return (SOURCE_ADOPTED, None if draft_id is None else int(draft_id))


def _with_extra(common: dict, extra: Optional[dict]) -> dict:
    """The caller's op-log note under the payload's own keys, never over them.

    ``extra`` is what the *window* knows about why a commit was made -- "the
    annotator was warned the shape is implausibly large and went ahead" -- which
    belongs in the audit trail of that commit. It is checked for
    JSON-serialisability by the session before anything is written
    (:func:`tda.ui.session_commits._loggable_extra`); here it only has to be
    unable to rewrite the payload's own account of what happened.
    """
    return common if not extra else dict(extra) | common


def _loggable(payload: dict) -> dict:
    """The op-log copy of a payload: the shared ``ref`` holders flattened away."""
    return payload | {
        "keyframes": [
            {k: v for k, v in state.items() if k != "ref"}
            | {"keyframe_id": state["ref"].get("keyframe_id")}
            for state in payload.get("keyframes", ())
        ]
    }


def _append_to_zorder(db: Db, key: FrameKey, seg: int, instance: str,
                      part: str = MAIN) -> Optional[dict]:
    """Put a newly drawn instance into the layer order (spec 4.2).

    On top -- except the two floor layers (task U17,
    :func:`tda.core.zorder_floor.floor_index`): a ``chassis`` goes to the very
    bottom and a ``motherboard`` directly above the lowest chassis, whatever
    order the annotator drew them in on the start frame.  The class is the
    instance record's.  Only here: a move the annotator makes stays theirs.

    Returns the before/after states for the undo payload, or ``None`` when the
    instance was in the order already and nothing had to change.
    """
    rec = db.zorder(key.desktop, key.view, seg)
    order = [tuple(entry) for entry in rec.order]
    if any(entry[0] == instance for entry in order):
        return None
    # a 框外零件 (task U18) is no floor layer and goes on top, in draw order;
    # nor does one of class chassis count as the machine's chassis
    cls_of = {k: r.cls for k, r in machine_only(db.instances(key.desktop)).items()}
    at = floor_index(order, cls_of.get(instance), cls_of)
    after = order[:at] + [(instance, part)] + order[at:]
    db.set_zorder(ZOrderRec(key.desktop, key.view, seg, after, version=rec.version + 1))
    return {
        "before": {"pose_segment": seg, "order": [list(e) for e in order],
                   "version": rec.version},
        "after": {"pose_segment": seg, "order": [list(e) for e in after],
                  "version": rec.version + 1},
    }


def _commit_frame_override(db: Db, truth: TruthService, key: FrameKey, instance: str,
                           edited: np.ndarray, hw: tuple[int, int], annotator: str,
                           extra: Optional[dict] = None) -> dict:
    """Spec 4.3 单帧覆盖: the edit applies to this frame and to no other.

    The stored mask is the *visible* one, so the frame's occluders come off it
    here -- the compiler subtracts them before an override is applied, never
    after (spec 3.3 steps 6-7), and a pinned mask has to mean the same thing as
    the value it replaces.
    """
    visible = edited & ~occluder_union(db, key, hw)
    previous = db.frame_overrides(key).get(instance)
    common = {"desktop": key.desktop, "view": key.view, "step": key.step,
              "instance": instance, "steps": [key.step]}
    # the pixels change, the label -- and the reason it gives (U15) -- do not;
    # except that pixels drawn on a part 被线挡住（完全） say part of it is seen
    label = None if previous is None else previous.visibility
    seen = cable_label(previous) == CABLE_FULL and bool(visible.any())
    payload = _with_extra(common, extra) | {
        "exists": True, "visible_rle": masks.encode_rle_boxed(visible),
        "visibility": CABLE_PARTIAL if seen else label,
        "occluder": None if previous is None else previous.occluder}
    inverse = common | override_state(previous)
    with db.transaction():
        db.set_frame_override(
            FrameOverride(key, instance, payload["visible_rle"], payload["visibility"],
                          payload["occluder"])
        )
        db.log_op(key.desktop, key.view, "set_frame_override", payload, inverse, annotator)
    return _result(db, truth, key, [key.step], Op("set_frame_override", payload, inverse),
                   {"cable_partial": seen})


# --------------------------------------------------------------------------- #
# layering, visibility, occluders
# --------------------------------------------------------------------------- #
def require_instance(known, instance: str) -> None:
    """Refuse a layering gesture that names something this frame does not have.

    Silently treating an unknown neighbour as "the top of the stack" is how a
    typo, or a stale panel row, turns into a z-order nobody asked for.
    """
    if known is not None and instance not in known:
        raise SessionRefusal(f"no such instance in this frame: {instance!r}")


def set_zorder_move(db: Db, truth: TruthService, key: FrameKey, instance: str,
                    above_of: str, annotator: str = "system",
                    known: Optional[set[str]] = None) -> dict:
    """Move ``instance`` directly above ``above_of`` in the stored layer order.

    The order is a total order over ``(instance, part)`` pairs, bottom first, so
    a multi-part instance moves as one block and keeps its internal order.  A
    neighbour the order does not mention yet is first pinned at the top and the
    instance placed above it, which fixes their relative order and leaves every
    other pair where it was; a neighbour that is not in ``known`` -- the
    instances this frame actually has -- is refused.
    """
    require_instance(known, above_of)
    cache = InputCache()
    seg = pose_segment_of(db, key, cache)
    rec = db.zorder(key.desktop, key.view, seg)
    order = [tuple(entry) for entry in rec.order]
    moved = [entry for entry in order if entry[0] == instance] or [(instance, MAIN)]
    rest = [entry for entry in order if entry[0] != instance]
    if not any(entry[0] == above_of for entry in rest):
        rest = rest + [(above_of, MAIN)]  # unordered so far: pin it, then go above it
    cut = max(i for i, entry in enumerate(rest) if entry[0] == above_of)
    after = rest[: cut + 1] + moved + rest[cut + 1:]
    steps = segment_steps(db, truth.tax, key, instance, seg, cache)

    common = {"desktop": key.desktop, "view": key.view, "pose_segment": seg, "steps": steps}
    payload = common | {"order": [list(e) for e in after], "version": rec.version + 1}
    inverse = common | {"order": [list(e) for e in order], "version": rec.version}
    with db.transaction():
        write_zorder(db, key.desktop, key.view, payload)
        db.log_op(key.desktop, key.view, "set_zorder", payload, inverse, annotator)
    return _result(db, truth, key, steps, Op("set_zorder", payload, inverse))


def commit_pair_override(db: Db, truth: TruthService, key: FrameKey, above: str,
                         below: str, annotator: str = "system",
                         known: Optional[set[str]] = None,
                         extra: Optional[dict] = None) -> dict:
    """Record "``above`` beats ``below``" for this pose segment (spec 4.3 改层级).

    A pairwise exception rather than a new global order: it says one thing about
    one pair and leaves every other relation where the annotator put it.
    """
    require_instance(known, above)
    require_instance(known, below)
    cache = InputCache()
    seg = pose_segment_of(db, key, cache)
    po = PairOverride(key.desktop, key.view, seg, above, below)
    existed = any(p == po for p in db.pair_overrides(key.desktop, key.view, seg))
    # exactly the frames preview_pair promised: where both are mask layers
    steps = pair_steps(db, truth.tax, key, above, below, seg, cache)

    common = {"desktop": key.desktop, "view": key.view, "pose_segment": seg,
              "above": above, "below": below, "steps": steps}
    payload = _with_extra(common, extra) | {"exists": True}
    inverse = common | {"exists": existed}
    with db.transaction():
        db.set_pair_override(po)
        db.log_op(key.desktop, key.view, "set_pair_override", payload, inverse, annotator)
    return _result(db, truth, key, steps, Op("set_pair_override", payload, inverse))


def set_visibility(db: Db, truth: TruthService, key: FrameKey, instance: str, vis: str,
                   annotator: str = "system") -> dict:
    """Override one instance's ``visibility`` label on this frame (spec 6.2).

    A label is a frame-level statement, so it is stored as a
    :class:`~tda.core.model.FrameOverride` -- merged into the one already there,
    which keeps a hand-drawn single-frame mask intact.  A 「被线挡住」 reason
    (task U15) is not: a digit is a new statement without one, so the reason
    goes, and so does the empty mask 完全 wrote (:func:`tda.core.occluder.kept_mask`).
    """
    if vis not in api.VISIBILITY_VALUES:
        raise SessionRefusal(
            f"visibility must be one of {api.VISIBILITY_VALUES}, got {vis!r}")
    previous = db.frame_overrides(key).get(instance)
    common = {"desktop": key.desktop, "view": key.view, "step": key.step,
              "instance": instance, "steps": [key.step]}
    payload = common | {"exists": True, "visibility": vis, "occluder": None,
                        "visible_rle": _mask_without_reason(previous)}
    inverse = common | override_state(previous)
    with db.transaction():
        db.set_frame_override(FrameOverride(key, instance, payload["visible_rle"], vis))
        db.log_op(key.desktop, key.view, "set_frame_override", payload, inverse, annotator)
    return _result(db, truth, key, [key.step], Op("set_frame_override", payload, inverse))


def clear_visibility(db: Db, truth: TruthService, key: FrameKey, instance: str,
                     annotator: str = "system") -> dict:
    """Take a hand-set ``visibility`` label of this frame back (undoable).

    A single-frame mask in the same override stays; the row goes only when it
    held nothing else. What a second 「看不见」 on a components-gallery shot
    does (task U10): the part is to be drawn again, with no label of its own.
    A 「被线挡住」 reason goes with its label, and so does 完全's empty mask.
    """
    previous = db.frame_overrides(key).get(instance)
    if previous is None or previous.visibility is None:
        return _result(db, truth, key, [key.step], None)
    common = {"desktop": key.desktop, "view": key.view, "step": key.step,
              "instance": instance, "steps": [key.step]}
    mask = _mask_without_reason(previous)
    keep = mask is not None
    payload = common | {"exists": keep, "visibility": None, "occluder": None,
                        "visible_rle": mask}
    inverse = common | override_state(previous)
    with db.transaction():
        if keep:
            db.set_frame_override(FrameOverride(key, instance, mask, None))
        else:
            db.delete_frame_override(key, instance)
        db.log_op(key.desktop, key.view, "set_frame_override", payload, inverse, annotator)
    return _result(db, truth, key, [key.step], Op("set_frame_override", payload, inverse))


def _mask_without_reason(previous: Optional[FrameOverride]) -> Optional[dict]:
    """The single-frame mask left when a label is replaced: 完全's empty one goes (U15)."""
    if previous is None:
        return None
    return kept_mask(previous) if previous.occluder else previous.visible_rle


def commit_occluder(db: Db, truth: TruthService, key: FrameKey, mask: np.ndarray,
                    occluder_type: str = "hand", annotator: str = "system") -> dict:
    """Store one occluder layer of this frame (spec 3.1, 4.2 step 4).

    One layer per type, because the compiler subtracts each type separately and
    the database keys ``occluder_mask`` on it.
    """
    layer = as_mask(mask, frame_hw(db, key, truth.cache_dir))
    previous = next((o for o in db.occluders(key) if o.occluder_type == occluder_type), None)
    common = {"desktop": key.desktop, "view": key.view, "step": key.step,
              "occluder_type": occluder_type, "steps": [key.step]}
    payload = common | {"exists": True, "rle": masks.encode_rle_boxed(layer)}
    inverse = common | {"exists": previous is not None,
                        "rle": None if previous is None else previous.rle}
    with db.transaction():
        db.set_occluder(OccluderMask(key, occluder_type, payload["rle"]))
        db.log_op(key.desktop, key.view, "set_occluder", payload, inverse, annotator)
    return _result(db, truth, key, [key.step], Op("set_occluder", payload, inverse))
