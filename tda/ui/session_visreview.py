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

W3i (2026-10-31), the user's check round: the toggle 「也列 visible 的段（接头
优先）」 (:meth:`VisReviewMixin.set_visibility_all`) lists the plain ``visible``
runs too, and the button 「这一段起直到拔掉都被线挡住」
(:meth:`VisReviewMixin.review_cable_until`) answers a line from its first frame
to the part's next state change (:func:`~tda.ui.session_cable.until_span`,
across pose segments) as one ``review_visibility`` op: 完全 on every frame, the
stamp (verdict ``cable``) on the confirmed ones.

W3m (2026-11-01), the mirror from the frame on screen **back**: the buttons
「第 24 帧起往前都被线挡住」 and 「第 24 帧起往前都可见」
(:meth:`VisReviewMixin.review_cable_since`) act from the canvas frame over
:func:`~tda.ui.session_cable.since_span` -- the first 完全 + the ``cable`` stamp
as above; the second withdraws the hand labels over the span and stamps the
confirmed frames ``computed`` where the compiler then says plain ``visible``
(the user's statement and the compiler agree), so the line leaves the queue
there; where the compiler still hides the part nothing is stamped and the row
re-enters the queue as an ordinary computed line. The progress follows. One op
each, one ``Ctrl+Z``.
"""
from __future__ import annotations

from typing import Iterable, Optional

from tda.core.model import (
    OCCLUDER_CABLE,
    OCCLUDER_VIEWPOINT,
    FrameKey,
    FrameOverride,
    Visibility,
)
from tda.core.occluder import FULL, PARTIAL, frame_label, occluded_override, shape_applies
from tda.core.truth import VERIFIED
from tda.core.truth_inputs import InputCache, annotatable_steps, frame_hw
from tda.core.vis_review import (
    VERDICT_CABLE,
    VERDICT_COMPUTED,
    VERDICT_LABEL,
    VERDICT_VIEWPOINT,
    VERDICTS,
    make_stamp,
    stamp_covers,
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
    CABLE_SINCE_EMPTY,
    CABLE_UNTIL_EMPTY,
    RULE_SINCE,
    RULE_UNTIL,
    _same,
    _settled,
    _signed,
    _steps_text,
    disputed_before,
    since_span,
    until_span,
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

    def set_visibility_all(self, on: bool) -> None:
        """「也列 visible 的段（接头优先）」 (W3i): list the plain ``visible`` runs too.

        Remembered for the session, across views; the progress counts them
        only while it is on.
        """
        self.review.visibility_all = bool(on)

    def visibility_all(self) -> bool:
        """Is the check round's toggle on? (:meth:`set_visibility_all`)"""
        return bool(self.review.visibility_all)

    # -- 「这一段起直到拔掉都被线挡住」 (W3i) ---------------------------------------------
    def review_cable_until(self, instance: str, from_step: Optional[int] = None) -> dict:
        """完全 被线挡住 from ``from_step`` to the part's next state change, stamped: one op.

        The span is :func:`~tda.ui.session_cable.until_span`'s (across pose
        segments, to the frame before the part is unplugged / taken out /
        moved / opened; to the view's last frame when nothing changes it
        again). Every frame of it gets the label (the U15 override, ``hw``
        read per frame); the **confirmed** ones get the stamp too (verdict
        ``cable``, as `` ` `` leaves on a run), the others only the label --
        a stamp answers a confirmed row. A confirmed frame that already carries
        the label **and** a cable stamp for it is left alone, so a second click
        writes nothing. One ``review_visibility`` op, one
        ``Ctrl+Z``; the U15 safeguards hold per frame
        (:func:`~tda.ui.session_cable.accept_own_label`).

        Refused, nothing written, on a gallery shot, from a frame this view
        does not annotate, and where the part has no shape in its place on
        some frame of the span (named). Returns what a verdict returns plus
        ``frames`` (the span), ``stamped`` (the confirmed frames), ``until``,
        ``change``, ``segments`` and ``rule``; ``changed`` is ``False`` (no
        op) when every frame already said exactly that.
        """
        instance, key, start, cache = self._span_start(instance, from_step)
        span = until_span(self.db, self.tax, start, instance, cache)
        if not span.steps:
            raise SessionRefusal(CABLE_UNTIL_EMPTY)
        extra = {"rule": RULE_UNTIL, "until": span.until, "segments": span.segments,
                 "change": list(span.change) if span.change else None}
        return self._review_label_span(key, instance, list(span.steps), cache, extra)

    # -- 「第 N 帧起往前都被线挡住」 / 「第 N 帧起往前都可见」 (W3m) ------------------------------
    def review_cable_since(self, instance: str, from_step: Optional[int] = None,
                           label: Optional[str] = FULL) -> dict:
        """From ``from_step`` back to the part's previous state change: 完全, or the labels withdrawn.

        The span is :func:`~tda.ui.session_cable.since_span`'s (the mirror of
        :meth:`review_cable_until`: across pose segments, from the frame after
        the part's previous own state change -- or the view's first frame --
        to ``from_step``, the frame on screen by default).

        ``label`` :data:`~tda.core.occluder.FULL` is 「往前都被线挡住」: exactly
        what :meth:`review_cable_until` does over that span (the label on every
        frame, the ``cable`` stamp on the confirmed ones, the same refusals, a
        second click a no-op). ``None`` is 「往前都可见」: every 被线挡住 / 画框外
        label the part carries on the span is withdrawn (frames without one are
        left alone, nothing is refused), and each **confirmed** frame of the
        span whose compiled label is then plain ``visible`` is stamped
        ``computed`` for it -- the frame counts as reviewed, the line leaves
        the queue there; a confirmed frame the compiler still hides (another
        part over it, too small, out of view) gets no stamp and comes back to
        the queue as an ordinary computed line, for the reviewer. One
        ``review_visibility`` op, one ``Ctrl+Z`` (the inverse holds the
        withdrawn rows and the old stamps as they were); the U15 safeguards
        hold per frame.

        Returns what :meth:`review_cable_until` returns, with ``since`` for
        ``until``, ``rule`` :data:`~tda.ui.session_cable.RULE_SINCE`, and, for
        撤回, ``withdrawn`` (the labels taken back) and ``frames`` the frames
        written (``span`` the whole span); ``changed`` is ``False`` and no op
        is made when nothing would move and nothing is left to stamp.
        """
        instance, key, start, cache = self._span_start(instance, from_step)
        span = since_span(self.db, self.tax, start, instance, cache)
        if not span.steps:
            raise SessionRefusal(CABLE_SINCE_EMPTY)
        extra = {"rule": RULE_SINCE, "since": span.since, "segments": span.segments,
                 "change": list(span.change) if span.change else None}
        if label is None:
            return self._review_clear_span(key, instance, list(span.steps), extra)
        return self._review_label_span(key, instance, list(span.steps), cache, extra)

    def _span_start(self, instance: str, from_step: Optional[int]):
        """``(instance, the open frame, the frame the span starts from, a cache)``; the gates."""
        if not self.is_open:
            raise SessionRefusal("no view is open")
        if self.gallery:
            raise SessionRefusal(T.NOT_IN_GALLERY)
        key = self.current()
        start = FrameKey(key.desktop, key.step if from_step is None else int(from_step), key.view)
        return str(instance), key, start, InputCache()

    def _review_label_span(self, key: FrameKey, instance: str, steps: list[int],
                           cache: InputCache, extra: dict) -> dict:
        """完全 on every frame of ``steps``, the ``cable`` stamp on the confirmed ones: one op.

        The body :meth:`review_cable_until` and :meth:`review_cable_since`
        share; ``extra`` (the rule and its bounds) goes into the op's payload
        (both directions) and the result.
        """
        shapes = self.db.keyframes(key.desktop, key.view, instance)
        refused: dict[str, list[int]] = {}
        frames, before, stamped = [], [], []
        for step in steps:
            here = FrameKey(key.desktop, step, key.view)
            why = shape_applies(self.db, self.tax, here, instance, cache, shapes)
            if why:
                refused.setdefault(_REFUSALS[OCCLUDER_CABLE].get(why, why), []).append(step)
                continue
            previous = self.db.frame_overrides(here).get(instance)
            hw = frame_hw(self.db, here, self.truth.cache_dir)
            after = occluded_override(here, instance, previous, FULL, hw, OCCLUDER_CABLE)
            flags = self.db.instance_flags(here, instance)
            confirmed = (self.db.get_frame(here) or {}).get("review_status") == VERIFIED
            state = {"step": step, "flags": flags}
            back = {"step": step, "flags": flags}
            new, old = override_state(after), override_state(previous)
            same = _same(new, old)
            # a confirmed frame already labelled so and stamped so (verdict cable
            # for 完全 / cable) needs nothing: a second click is a no-op (review S1)
            answered = (same and stamp_covers(flags, FULL, OCCLUDER_CABLE)
                        and (flags or {}).get("verdict") == VERDICT_CABLE)
            if confirmed and not answered:
                state["flags"] = with_stamp(flags, make_stamp(self.annotator, VERDICT_CABLE,
                                                              FULL, OCCLUDER_CABLE))
                stamped.append(step)
            if not same:
                state["override"], back["override"] = new, old
            if (confirmed and not answered) or "override" in state:
                frames.append(state)
                before.append(back)
        if refused:
            raise SessionRefusal("；".join(f"第 {_steps_text(found)} 帧：{why}"
                                          for why, found in refused.items()))
        extra = extra | {"frames": list(steps), "stamped": stamped}
        if not frames:
            return {"op": None, "changed": False, "verdict": VERDICT_CABLE, "label": FULL,
                    "steps": list(steps), "moved": [], "accepted": [], "withheld": [],
                    "conflicts": 0} | extra
        wanted = [f["step"] for f in frames]
        common = {"desktop": key.desktop, "view": key.view, "step": key.step,
                  "instance": instance, "steps": wanted, "verdict": VERDICT_CABLE,
                  } | {k: v for k, v in extra.items() if k not in ("frames", "stamped")}
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
            self.undo_stack.push(op, apply=False)
            self.review.invalidate()
            self._refresh_dirty()
            self._announce()
            return {"op": op, "changed": True, "verdict": VERDICT_CABLE, "label": FULL,
                    "steps": wanted, "moved": [], "accepted": [], "withheld": [],
                    "conflicts": 0} | extra
        out = _settled(self.db, self.truth, key, moved, op, FULL)
        out = _signed(self.db, self.truth, key.desktop, key.view, instance, moved,
                      self.annotator, disputed, out, key.step)
        out.update(verdict=VERDICT_CABLE, steps=wanted, moved=moved, **extra)
        return self._after_edit(out)

    def _review_clear_span(self, key: FrameKey, instance: str, steps: list[int],
                           extra: dict) -> dict:
        """「往前都可见」: the hand labels over ``steps`` withdrawn, the frames seen again stamped ``computed``.

        The override is written first and the moved frames compiled (and, on
        confirmed frames, re-signed through
        :func:`~tda.ui.session_cable.accept_own_label`) **before** the stamp is
        made: a ``computed`` stamp records the label it answers for
        (:func:`~tda.core.vis_review.stamp_covers`), and that label is the
        compiler's, known only once the override is gone. The op is logged
        after, with the flags before and after; its inverse puts the withdrawn
        rows and the old stamps back as they were.
        """
        desktop, view = key.desktop, key.view
        overrides: dict[int, tuple[dict, dict]] = {}      # step -> (new, old) override state
        withdrawn: set[str] = set()
        confirmed_steps: list[int] = []
        for step in steps:
            here = FrameKey(desktop, step, view)
            previous = self.db.frame_overrides(here).get(instance)
            if ((self.db.get_frame(here) or {}).get("review_status") == VERIFIED
                    and self.db.compiled_label(here, instance) is not None):
                confirmed_steps.append(step)
            label = frame_label(previous)
            if label is None:
                continue                         # no label of ours here: left alone
            after = occluded_override(here, instance, previous, None, (0, 0), OCCLUDER_CABLE)
            new, old = override_state(after), override_state(previous)
            if _same(new, old):
                continue
            withdrawn.add(str(label))
            overrides[step] = (new, old)
        moved = sorted(overrides)
        out = {"affected": [], "compiled": [], "rechecks": [], "conflicts": 0,
               "problems": {}, "frame": None, "accepted": [], "withheld": []}
        if moved:
            disputed = disputed_before(self.db, self.truth, desktop, view, instance, moved)
            with self.db.transaction():
                for step in moved:
                    _write_override(self.db, FrameKey(desktop, step, view), instance,
                                    overrides[step][0])
            affected = annotatable_steps(self.db, desktop, view, moved)
            stats = settle(self.db, self.truth, desktop, view, affected, key.step)
            out.update(affected=affected, compiled=stats["compiled"], rechecks=stats["rechecks"],
                       conflicts=stats["conflicts"], problems=stats["problems"],
                       frame=stats["frame"])
            out = _signed(self.db, self.truth, desktop, view, instance, moved, self.annotator,
                          disputed, out, key.step)
        # the rows now say what the compiler says: where that is plain visible --
        # the user's statement and the compiler agree -- stamp the confirmed frame
        # computed; where the compiler still hides the part (another part over it,
        # too small, out of view) nothing is stamped and the row comes back to the
        # queue as an ordinary computed line, for the reviewer to answer
        stamps: dict[int, tuple[Optional[dict], Optional[dict]]] = {}    # step -> (new, old)
        for step in confirmed_steps:
            here = FrameKey(desktop, step, view)
            visibility = (self.db.compiled_label(here, instance) or {}).get("visibility")
            held = self.db.frame_overrides(here).get(instance)
            if visibility != Visibility.VISIBLE.value or (held is not None and held.occluder):
                continue
            flags = self.db.instance_flags(here, instance)
            if (stamp_covers(flags, visibility, None)
                    and (flags or {}).get("verdict") == VERDICT_COMPUTED):
                continue                         # answered already: a second click is a no-op
            stamps[step] = (with_stamp(flags, make_stamp(self.annotator, VERDICT_COMPUTED,
                                                         visibility, None)), flags)
        stamped = sorted(stamps)
        wanted = sorted(set(moved) | set(stamped))
        extra = extra | {"frames": moved, "span": list(steps), "withdrawn": sorted(withdrawn),
                         "stamped": stamped}
        out.update(verdict=VERDICT_COMPUTED, label=None, steps=wanted or list(steps),
                   moved=moved, **extra)
        if not wanted:
            out.update(op=None, changed=False)
            return out
        frames, before = [], []
        for step in wanted:
            flags = self.db.instance_flags(FrameKey(desktop, step, view), instance)
            new_flags, old_flags = stamps.get(step, (flags, flags))
            state = {"step": step, "flags": new_flags}
            back = {"step": step, "flags": old_flags}
            if step in overrides:
                state["override"], back["override"] = overrides[step]
            frames.append(state)
            before.append(back)
        common = {"desktop": desktop, "view": view, "step": key.step, "instance": instance,
                  "steps": wanted, "verdict": VERDICT_COMPUTED,
                  } | {k: v for k, v in extra.items()
                       if k not in ("frames", "stamped", "span", "withdrawn")}
        if moved:
            common[CABLE_BY] = self.annotator    # re-signed both ways (apply_cable)
        op = Op(OP_KIND, common | {"frames": frames}, common | {"frames": before})
        with self.db.transaction():
            for step in stamped:
                self.db.set_instance_flags(FrameKey(desktop, step, view), instance,
                                           stamps[step][0])
            self.db.log_op(desktop, view, OP_KIND, op.payload, op.inverse, self.annotator)
        out.update(op=op, changed=True)
        if not moved:
            self.undo_stack.push(op, apply=False)
            self.review.invalidate()
            self._refresh_dirty()
            self._announce()
            return out
        return self._after_edit(out)

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
