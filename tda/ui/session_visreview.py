"""「可见性审查」 in the session (task U24): the fifth Review queue and its verdicts.

The queue is :func:`tda.core.vis_review.runs_of` over the view's confirmed rows
whose label is not plain ``visible`` (or that carry a hand label), minus the
ones stamped reviewed; :class:`~tda.ui.session_queues.ReviewState` keeps it
(:meth:`~tda.ui.session_queues.ReviewState.visibility_queue`).

A verdict is **one op**, ``review_visibility``, carrying per frame the flags
object before and after (the stamp) and -- when the verdict changes the label --
the frame override before and after.  Undo puts both back.  A verdict that
changes the label is the annotator's own statement about a frame they already
confirmed, so it goes the way task U15's cable label goes (its review M8):
:func:`~tda.ui.session_cable.disputed_before` first (a difference that was
already there opens its conflict for Review and is not signed),
then the write, then :func:`~tda.ui.session_cable.accept_own_label` -- the
frame stays confirmed under the new label, both ways (the op carries
:data:`~tda.ui.session_cable.CABLE_BY`, so :func:`~tda.ui.session_cable.apply_cable`
re-signs its undo and redo the same way).

A queue line is a run of frames with one signature, so a verdict covers the
whole run as that one op (the coordinator's ruling on U24): the safeguards
above hold frame by frame -- the ones already in dispute are not signed, and
pixels a label *reveals* are signed only on the frame on screen (U15 review
round 4); the others keep their conflicts for Review.

A stamp alone (``Space``) changes no input of any frame: no compilation, no
re-check, and the frame in hand keeps its compiled frame.
"""
from __future__ import annotations

from typing import Iterable, Optional

from tda.core.model import (
    OCCLUDER_CABLE,
    OCCLUDER_VIEWPOINT,
    FrameKey,
    FrameOverride,
)
from tda.core.occluder import FULL, PARTIAL, occluded_override, shape_applies
from tda.core.truth import VERIFIED
from tda.core.truth_inputs import InputCache, frame_hw
from tda.core.vis_review import (
    VERDICT_CABLE,
    VERDICT_COMPUTED,
    VERDICT_LABEL,
    VERDICT_VIEWPOINT,
    VERDICTS,
    make_stamp,
    with_stamp,
)
from tda.ui import session_api as api
from tda.ui import visreview_text as T
from tda.ui.commands import Op
from tda.ui.session_api import SessionRefusal
from tda.ui.session_cable import (
    CABLE_BY,
    CABLE_NEEDS_SHAPE,
    CABLE_ON_BENCH,
    CABLE_REMOVED,
    _same,
    _settled,
    _signed,
    _steps_text,
    disputed_before,
)
from tda.ui.session_edit import _mask_without_reason
from tda.ui.session_ops import _write_override, override_state, settle

__all__ = ["OP_KIND", "VisReviewMixin", "apply_vis_review"]

#: The op one verdict logs.
OP_KIND = "review_visibility"

#: Why 完全/部分 cannot go on a part here, per reason (the U15 rule, both reasons).
_REFUSALS = {
    OCCLUDER_CABLE: {"removed": CABLE_REMOVED, "bench": CABLE_ON_BENCH,
                     "no_shape": CABLE_NEEDS_SHAPE},
    OCCLUDER_VIEWPOINT: {"removed": T.VIEWPOINT_REMOVED, "bench": T.VIEWPOINT_BENCH,
                         "no_shape": T.VIEWPOINT_NEEDS_SHAPE},
}


def apply_vis_review(db, truth, payload: dict, current: Optional[int] = None) -> dict:
    """Both directions of a ``review_visibility`` op: flags, overrides, then settle.

    ``payload["frames"]`` holds ``{"step", "flags"}`` per frame and, where the
    verdict changed the label, ``"override"`` (an :func:`override_state`).
    Only the frames whose override moved are settled; a stamp is no input of
    the compiler.
    """
    desktop, view, instance = payload["desktop"], payload["view"], payload["instance"]
    moved: list[int] = []
    with db.transaction():
        for state in payload["frames"]:
            key = FrameKey(desktop, int(state["step"]), view)
            if "override" in state:
                _write_override(db, key, instance, state["override"])
                moved.append(int(state["step"]))
            db.set_instance_flags(key, instance, state.get("flags"))
    if not moved:
        return {"problems": {}, "frame": None, "rechecks": [], "compiled": [],
                "conflicts": 0, "moved": []}
    stats = settle(db, truth, desktop, view, moved, current)
    stats["moved"] = moved
    return stats


class VisReviewMixin:
    """The visibility review for :class:`tda.ui.session.AnnotationSession`."""

    # -- the queue -----------------------------------------------------------
    def visibility_queue(self) -> list[dict]:
        """The 「可见性审查」 entries of the open view (runs; see the module doc)."""
        if not self.is_open:
            return []
        return self.review.visibility_queue()

    def visibility_progress(self) -> tuple[int, int]:
        """``(reviewed, total)`` rows of the open view."""
        if not self.is_open:
            return 0, 0
        return self.review.visibility_progress()

    def set_visibility_live(self, live: bool) -> None:
        """Say whether the queue is on screen: :meth:`queues` lists it only then.

        It reads every confirmed row of the view, so annotating -- where the
        Review panel is hidden and refreshed on every frame change for its
        conflict count -- does not pay for it.
        """
        self.review.visibility_live = bool(live)

    # -- verdicts ------------------------------------------------------------
    def review_visibility(self, instance: str, verdict: str, label: Optional[str] = None,
                          steps: Optional[Iterable[int]] = None) -> dict:
        """One verdict on ``instance`` over ``steps``: stamps, and the label it names.

        ``verdict`` is one of :data:`~tda.core.vis_review.VERDICTS`:
        ``computed`` stamps the rows as they are; ``label`` sets ``label`` (a
        visibility value) by hand, without a reason; ``cable`` / ``viewpoint``
        set 完全 / 部分 (``label`` :data:`~tda.core.occluder.FULL` /
        ``PARTIAL``) with that reason. ``steps`` is the frames -- the queue's
        run, one frame in its 按帧 listing -- and defaults to the open frame;
        all of them are **one op**, one ``Ctrl+Z``. The U15 safeguards hold per
        frame: a frame that already disagreed is not signed, and pixels a label
        reveals are signed on the open frame only (:func:`accept_own_label`).

        Refused (:class:`~tda.ui.session_api.SessionRefusal`), with nothing
        written, when a frame is not confirmed, on a components-gallery shot,
        and -- for 完全 / 部分 -- where the part has no shape in its place on
        some frame; a refusal over several frames names them.
        """
        if not self.is_open:
            raise SessionRefusal("no view is open")
        if self.gallery:
            raise SessionRefusal(T.NOT_IN_GALLERY)
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
        if verdict == VERDICT_LABEL and label not in api.VISIBILITY_VALUES:
            raise SessionRefusal(
                f"visibility must be one of {api.VISIBILITY_VALUES}, got {label!r}")
        if verdict not in (VERDICT_COMPUTED, VERDICT_LABEL) and label not in (FULL, PARTIAL):
            raise SessionRefusal(f"完全 / 部分 is {FULL!r} or {PARTIAL!r}, got {label!r}")
        key = self.current()
        instance = str(instance)
        wanted = [key.step] if steps is None else sorted({int(s) for s in steps})
        frames, before = [], []
        refused: dict[str, list[int]] = {}
        cache = InputCache()
        shapes = (self.db.keyframes(key.desktop, key.view, instance)
                  if verdict in (VERDICT_CABLE, VERDICT_VIEWPOINT) else None)
        for step in wanted:
            here = FrameKey(key.desktop, step, key.view)
            try:
                state, back = self._verdict_frame(here, instance, verdict, label,
                                                  cache, shapes)
            except SessionRefusal as why:
                refused.setdefault(str(why), []).append(step)
                continue
            frames.append(state)
            before.append(back)
        if refused:
            if len(wanted) == 1:
                raise SessionRefusal(next(iter(refused)))
            raise SessionRefusal("；".join(f"第 {_steps_text(found)} 帧：{why}"
                                          for why, found in refused.items()))
        common = {"desktop": key.desktop, "view": key.view, "step": key.step,
                  "instance": instance, "steps": wanted, "verdict": verdict}
        moved = [f["step"] for f in frames if "override" in f]
        if moved:
            common[CABLE_BY] = self.annotator    # re-signed both ways (apply_cable)
        payload = common | {"frames": frames}
        inverse = common | {"frames": before}
        op = Op(OP_KIND, payload, inverse)
        disputed = (disputed_before(self.db, self.truth, key.desktop, key.view, instance,
                                    moved) if moved else set())
        with self.db.transaction():
            for state in frames:
                here = FrameKey(key.desktop, int(state["step"]), key.view)
                if "override" in state:
                    _write_override(self.db, here, instance, state["override"])
                self.db.set_instance_flags(here, instance, state["flags"])
            self.db.log_op(key.desktop, key.view, OP_KIND, payload, inverse, self.annotator)
        if not moved:
            # a stamp: nothing the compiler reads moved, so nothing is compiled
            self.undo_stack.push(op, apply=False)
            self.review.invalidate()
            self._refresh_dirty()
            self._announce()
            return {"op": op, "verdict": verdict, "steps": wanted, "moved": [],
                    "accepted": [], "withheld": [], "conflicts": 0}
        out = _settled(self.db, self.truth, key, moved, op, label)
        out = _signed(self.db, self.truth, key.desktop, key.view, instance, moved,
                      self.annotator, disputed, out, key.step)
        out.update(verdict=verdict, steps=wanted, moved=moved)
        return self._after_edit(out)

    def _verdict_frame(self, key: FrameKey, instance: str, verdict: str,
                       label: Optional[str], cache: Optional[InputCache] = None,
                       shapes=None) -> tuple[dict, dict]:
        """``(after, before)`` of one frame: its flags, and its override when it moves.

        ``cache`` / ``shapes`` (the part's keyframes of this view) are read
        once for a run.
        """
        frame = self.db.get_frame(key) or {}
        if frame.get("review_status") != VERIFIED:
            raise SessionRefusal(T.NOT_CONFIRMED.format(step=key.step))
        row = self.db.compiled_label(key, instance)
        if row is None:
            raise SessionRefusal(f"第 {key.step} 帧没有 {instance} 这一行")
        previous = self.db.frame_overrides(key).get(instance)
        after: Optional[FrameOverride] = previous
        if verdict == VERDICT_COMPUTED:
            visibility, reason = row.get("visibility"), (previous.occluder if previous
                                                          else None)
        elif verdict == VERDICT_LABEL:
            # what a digit always wrote (set_visibility): the label, no reason,
            # a hand-drawn single-frame mask kept and 完全's empty one dropped
            after = FrameOverride(key, instance, _mask_without_reason(previous), label, None)
            visibility, reason = label, None
        else:
            reason = OCCLUDER_CABLE if verdict == VERDICT_CABLE else OCCLUDER_VIEWPOINT
            why = shape_applies(self.db, self.tax, key, instance, cache, shapes)
            if why:
                raise SessionRefusal(_REFUSALS[reason].get(why, why))
            hw = frame_hw(self.db, key, self.truth.cache_dir) if label == FULL else (0, 0)
            after = occluded_override(key, instance, previous, label, hw, reason)
            visibility = label
        flags = self.db.instance_flags(key, instance)
        stamp = make_stamp(self.annotator, verdict, visibility, reason)
        state = {"step": key.step, "flags": with_stamp(flags, stamp)}
        back = {"step": key.step, "flags": flags}
        new, old = override_state(after), override_state(previous)
        if not _same(new, old):
            state["override"], back["override"] = new, old
        return state, back
