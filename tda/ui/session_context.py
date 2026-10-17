"""The session's half of the 框外零件 (task U18, spec 3.2c).

:mod:`tda.core.context` says what a context object is and how it is stored;
this module is what the window calls on the open frame:

* :meth:`ContextMixin.add_context_object` -- make one (one ``op_log`` row with
  its inverse) and push an ``add_context`` op on the undo stack, so ``Ctrl+Z``
  takes it back -- with its shapes, if any were drawn and not undone before it
  (:func:`apply_context`: both directions of the op, one handler);
* ``I`` on one -- 「不在这张里」: an ``out_of_view`` label with no pixels on this
  frame, the gallery's 看不见, toggled (:meth:`ContextMixin.toggle_context_absent`),
  and ``Ctrl+I`` the same over a span of the pose segment in one op
  (:meth:`ContextMixin.set_context_absent_range`);
* the instance list's rows for the context objects the frame does not have
  (:meth:`ContextMixin.context_rows`), so one can be picked and drawn there.

Its shapes go through the ordinary commit path: :func:`tda.ui.session_ops.placement_of`
puts them in their own chain, and :func:`tda.core.truth_inputs.gather` compiles
them in their own group.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from tda.core.compiler import marked_absent, select_keyframe
from tda.core.context import (
    ContextError,
    chain_of,
    context_label,
    plan_context,
    remove_context,
    restore_context,
    write_context,
)
from tda.core.db import Db
from tda.core.model import (
    CONTEXT_OTHER,
    FrameKey,
    FrameOverride,
    Visibility,
    is_context,
    is_context_key,
)
from tda.core.truth import TruthService
from tda.core.truth_inputs import InputCache, annotatable_steps, pose_segment_of
from tda.ui import context_text as T
from tda.ui.class_names import class_zh
from tda.ui.commands import Op
from tda.ui import session_edit as edit
from tda.ui.session_api import SessionRefusal
from tda.ui.session_cable import range_bounds
from tda.ui.session_ops import _write_override, frozen_among, override_state, settle

__all__ = ["OP_KIND", "ContextMixin", "apply_context", "context_name"]

#: The undo stack's kind for a creation (registered in :data:`tda.ui.commands.KINDS`).
OP_KIND = "add_context"
#: The ``Ctrl+I`` op: the U15 range kind, written without a cable reason.
OP_RANGE = "set_frame_overrides"
ABSENT = Visibility.OUT_OF_VIEW.value


def context_name(rec) -> str:
    """``框外·散热器支架 ctx.cooler_bracket.01`` -- its name and its key."""
    if rec is None:
        return ""
    label = context_label(rec)
    zh = label or (T.OTHER_ZH if rec.cls == CONTEXT_OTHER else class_zh(rec.cls, rec.attrs))
    return f"框外·{zh} {rec.key}"


def apply_context(db: Db, truth: TruthService, payload: dict,
                  current: Optional[int] = None) -> dict:
    """Both directions of an ``add_context`` op: make it again, or take it back.

    Taking it back removes it with everything that names it
    (:func:`tda.core.context.remove_context`) and keeps that in the op's
    shared ``stash``, which the redo restores from -- shapes and all.
    """
    desktop, view = int(payload["desktop"]), str(payload["view"])
    stash = payload.get("stash")
    stash = stash if isinstance(stash, dict) else {}
    if payload.get("exists"):
        record = stash.get("record") or {"instance": payload["instance"], "frames": {}}
        restore_context(db, record)
    else:
        record = remove_context(db, desktop, str(payload["key"]))
        stash["record"] = record
    frames = record.get("frames") or {}
    for other, steps in frames.items():
        if other != view:
            truth.queue_rechecks(desktop, other, frozen_among(db, desktop, other, steps))
    here = list(frames.get(view) or [])
    if current is not None:
        here.append(int(current))
    return settle(db, truth, desktop, view, here, current)


class ContextMixin:
    """框外零件 for :class:`tda.ui.session.AnnotationSession` (task U18)."""

    # -- reads ---------------------------------------------------------------
    def context_records(self) -> dict:
        """``key -> InstanceRec`` of the open desktop's context objects."""
        if not self.is_open:
            return {}
        return {key: rec for key, rec in self.db.instances(self.desktop).items()
                if is_context(rec)}

    def context_absent_here(self, overrides: Optional[dict] = None) -> set[str]:
        """The context objects this frame says are not in it (``I``).

        ``overrides`` is the frame's, when the caller has read them already.
        """
        if not self.is_open:
            return set()
        if overrides is None:
            overrides = self.db.frame_overrides(self.current())
        return {name for name, fo in overrides.items()
                if is_context_key(name) and marked_absent(fo)}

    def context_rows(self, listed: set[str], instances: Optional[dict] = None) -> list[dict]:
        """Instance-list rows for the context objects ``listed`` does not have.

        A frame where none of its shapes applies does not have it; the row is
        still there -- 「—」, not drawn -- so the annotator can pick it and draw
        it on this frame too. ``instances`` is the desktop's table, when read.
        """
        records = (self.context_records() if instances is None
                   else {k: r for k, r in instances.items() if is_context(r)})
        out = []
        for key, rec in sorted(records.items()):
            if key in listed:
                continue
            out.append({"key": key, "cls": rec.cls, "state": "", "placement": "context",
                        "visibility": "", "z": -1, "hidden": key in self._hidden,
                        "vis_override": False, "has_shape": False, "cable": None,
                        "context": True, "label": context_label(rec)})
        return out

    def _context_applies(self, instance: str, key: Optional[FrameKey] = None) -> bool:
        """Does a shape of the context object apply at ``key`` (the open frame)?"""
        key = key or self.current()
        seg = pose_segment_of(self.db, key)
        chain = chain_of(self.db.keyframes(key.desktop, key.view, instance), key, seg)
        return select_keyframe(chain, key.step) is not None

    # -- making one ----------------------------------------------------------
    def add_context_object(self, cls: str, label: str = "", note: str = "") -> str:
        """Make a 框外零件 of ``cls`` on the open desktop; returns its key.

        One ``op_log`` row with its inverse (:data:`tda.core.context.OP_ADD`)
        and one ``add_context`` op on the undo stack. Refused without an open
        frame and with pixels of another part waiting in the editing layer
        (the window asks first).
        """
        if not self.is_open:
            raise SessionRefusal("没有打开的帧")
        if self.layer.active and self.layer.changed():
            raise SessionRefusal(f"{self.layer.instance} 正在画、还没提交：先 Enter 或 Esc")
        try:
            rec = plan_context(self.db.instances(self.desktop), self.tax, self.desktop,
                               cls, label, note, self.annotator)
            write_context(self.db, rec, self.annotator)
        except ContextError as refused:
            raise SessionRefusal(str(refused)) from None
        stash: dict = {}
        common = {"desktop": self.desktop, "view": self.view, "key": rec.key,
                  "stash": stash}
        payload = common | {"exists": True, "instance": asdict(rec)}
        inverse = common | {"exists": False}
        result = settle(self.db, self.truth, self.desktop, self.view,
                        [self.current().step], self.current().step)
        result["op"] = Op(OP_KIND, payload, inverse)
        self._after_edit(result)
        return rec.key

    def context_changed(self) -> None:
        """A context object went elsewhere (Steps mode): recompile what is shown."""
        self._drop_gone_context_edit()
        self._invalidate()
        self._announce()

    def _drop_gone_context_edit(self) -> None:
        """An undo took the context object being drawn away: its layer goes too."""
        editing = self.layer.instance
        if (editing and is_context_key(editing)
                and editing not in self.db.instances(self.desktop)):
            self.layer.clear()
            self.sigEditingChanged.emit(None)

    # -- 「不在这张里」 --------------------------------------------------------
    def context_absent_gate(self, instance: str) -> tuple[bool, str]:
        """``(marked here, why I cannot)`` for ``I`` on a context object here."""
        if not is_context_key(instance):
            return False, ""
        if str(instance) in self.context_absent_here():
            return True, ""
        rec = self.context_records().get(str(instance))
        if rec is None:
            return False, T.NOT_IN_TABLE.format(instance=instance)
        if not self._context_applies(str(instance)):
            return False, T.NO_SHAPE_HERE.format(label=context_name(rec))
        return False, ""

    def toggle_context_absent(self, instance: str) -> dict:
        """``I`` on a context object: 「不在这张里」, or -- again -- take it back.

        The label is ``out_of_view`` with no pixels on this frame (undoable);
        taking it back keeps a single-frame mask drawn there. Refused where no
        shape of it applies (it is not here anyway) and while its pixels are
        being drawn.
        """
        marked, why = self.context_absent_gate(instance)
        if why:
            raise SessionRefusal(why)
        if marked:
            self.clear_visibility(str(instance))
            return {"absent": False}
        if self.layer.instance == str(instance) and self.layer.changed():
            raise SessionRefusal(f"{instance} 正在画：先 Esc 放弃，再按 I")
        if self.layer.instance == str(instance):
            self.clear_edit()
        self._mark_absent(str(instance))
        return {"absent": True}

    def _mark_absent(self, instance: str) -> None:
        """The label with no pixels: a single-frame mask there goes with it."""
        key = self.current()
        previous = self.db.frame_overrides(key).get(instance)
        common = {"desktop": key.desktop, "view": key.view, "step": key.step,
                  "instance": instance, "steps": [key.step]}
        payload = common | {"exists": True, "visibility": ABSENT, "occluder": None,
                            "visible_rle": None}
        inverse = common | override_state(previous)
        with self.db.transaction():
            self.db.set_frame_override(FrameOverride(key, instance, None, ABSENT))
            self.db.log_op(key.desktop, key.view, "set_frame_override", payload, inverse,
                           self.annotator)
        self._after_edit(edit._result(self.db, self.truth, key, [key.step],
                                      Op("set_frame_override", payload, inverse)))

    def set_context_absent_range(self, instance: str, absent: bool, to_step: int) -> dict:
        """``Ctrl+I`` on a context object: 「不在」 (or taken back) from here to ``to_step``.

        One op over the annotatable frames of this pose segment between the
        two: 「不在」 on the frames a shape of it applies to (the others do not
        have it anyway), 撤回 on the frames that carry the label. Returns
        ``{"frames": [steps written]}``.
        """
        if not is_context_key(instance):
            raise SessionRefusal(T.NOT_IN_TABLE.format(instance=instance))
        if absent and self.layer.instance == str(instance) and self.layer.changed():
            raise SessionRefusal(f"{instance} 正在画：先 Esc 放弃，再按 Ctrl+I")
        key = self.current()
        cache = InputCache()
        low, high = range_bounds(self.db, key, cache)
        to_step = int(to_step)
        if not low <= to_step <= high:
            raise SessionRefusal(f"第 {to_step} 步不在这一位姿段里（第 {low}–{high} 步）")
        first, last = sorted((key.step, to_step))
        seg = pose_segment_of(self.db, key, cache)
        steps = [s for s in annotatable_steps(self.db, key.desktop, key.view,
                                              range(first, last + 1))
                 if pose_segment_of(self.db, FrameKey(key.desktop, s, key.view), cache) == seg]
        frames, before = [], []
        for step in steps:
            here = FrameKey(key.desktop, step, key.view)
            previous = self.db.frame_overrides(here).get(str(instance))
            if absent:
                if marked_absent(previous) or not self._context_applies(str(instance), here):
                    continue
                after = {"exists": True, "visible_rle": None, "visibility": ABSENT,
                         "occluder": None}
            else:
                if not marked_absent(previous):
                    continue
                after = {"exists": False, "visible_rle": None, "visibility": None,
                         "occluder": None}
            frames.append({"step": step} | after)
            before.append({"step": step} | override_state(previous))
        if not frames:
            return {"frames": []}
        if self.layer.instance == str(instance) and absent:
            self.clear_edit()
        written = [f["step"] for f in frames]
        common = {"desktop": key.desktop, "view": key.view, "step": key.step,
                  "instance": str(instance), "steps": written}
        payload = common | {"frames": frames}
        inverse = common | {"frames": before}
        with self.db.transaction():
            for state in frames:
                _write_override(self.db, FrameKey(key.desktop, state["step"], key.view),
                                str(instance), state)
            self.db.log_op(key.desktop, key.view, OP_RANGE, payload, inverse,
                           self.annotator)
        result = settle(self.db, self.truth, key.desktop, key.view, written, key.step)
        result["op"] = Op(OP_RANGE, payload, inverse)
        self._after_edit(result)
        return {"frames": written}
