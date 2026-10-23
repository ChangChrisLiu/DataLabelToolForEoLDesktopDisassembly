"""「其实是硬盘架」 / 「其实是卡扣」 as an undoable op of the session (task U21).

:mod:`tda.core.reclass` does the work -- a merge into a fresh instance of the
other class, the cage rule for it in the same transaction, one op-log row --
and this is the session's half: which instance-list rows offer it
(:meth:`ReclassMixin.reclass_offers`), the confirmation's plan (a rehearsal
that writes nothing), the write, and the undo/redo handler. The keys change,
so the frames of **this view** the reclass reaches are settled -- the one in
hand compiled, the frozen ones queued for a re-check -- and the other views
follow on their next opening, as for ``Shift+T``.

Both directions of the session op name the reclass by its keys, not by an
op-log id: Ctrl+Z undoes the newest reclass row that made the new key
(:func:`tda.core.reclass.latest_reclass`), Ctrl+Y makes it again under the
same keys (the new part's and its cage latch's).
"""
from __future__ import annotations

from typing import Optional

from tda.core.db import Db
from tda.core.implied import is_implied
from tda.core.merge_instances import MergeError
from tda.core.model import FrameKey, is_context, is_context_key, is_provisional
from tda.core.reclass import (
    CAPTION,
    ReclassPlan,
    latch_of,
    latest_reclass,
    plan_reclass,
    reclass_gate,
    reclass_instance,
    undo_reclass,
)
from tda.core.reclass_replay import reclass_target
from tda.core.truth import TruthService
from tda.core.truth_inputs import annotatable_steps
from tda.ui.commands import Op
from tda.ui.context_text import NO_VERB as CONTEXT_NO_VERB
from tda.ui.gallery import NO_VERB
from tda.ui.session_api import VERB_NEEDS_CLEAN_LAYER, SessionRefusal
from tda.ui.session_ops import settle

__all__ = ["OP_KIND", "ReclassMixin", "apply_reclass", "commit_reclass", "reclass_offer"]

#: ``op`` kind of the session's undo stack (:data:`tda.ui.commands.KINDS`),
#: the same word as the op-log row's.
OP_KIND = "reclass_instance"


def reclass_offer(db: Db, desktop: int, key: str) -> Optional[str]:
    """The class ``key`` may be said to be instead, or ``None`` (the cheap gate)."""
    if is_provisional(key) or is_context_key(key):
        return None
    rec = db.instances(int(desktop)).get(str(key))
    if rec is None or is_context(rec) or is_implied(rec):
        return None
    return reclass_target(rec.cls)


def _reach(db: Db, desktop: int, view: str, plan: ReclassPlan, current: int) -> list[int]:
    """The frames of ``view`` a reclass settles: the one in hand, the ones it re-checks."""
    steps = set(plan.rechecks.get(view) or ()) | {int(current)}
    steps |= {int(step) for step, v in plan.merge.compiled_renamed if v == view}
    steps |= {int(step) for v, step in plan.merge.frames if v == view}
    return annotatable_steps(db, desktop, view, sorted(steps))


def apply_reclass(db: Db, truth: TruthService, payload: dict,
                  current: Optional[int] = None) -> dict:
    """Both directions of the session's reclass op: redo it, or undo the newest one."""
    desktop = int(payload["desktop"])
    try:
        if payload.get("undo"):
            op = latest_reclass(db, desktop, str(payload["undo"]))
            if op is None:
                raise SessionRefusal(f"撤销不了：没有把 {payload.get('key')} 改成 "
                                     f"{payload['undo']} 的记录了")
            undo_reclass(db, truth.tax, op, str(payload.get("annotator") or ""))
        else:
            reclass_instance(db, truth.tax, desktop, str(payload["key"]), payload.get("cls"),
                             str(payload.get("annotator") or ""),
                             new_key=str(payload["new"]), latch_key=payload.get("latch"))
    except MergeError as refused:          # ReclassError is one
        raise SessionRefusal(str(refused)) from refused
    return settle(db, truth, desktop, payload["view"], payload["reach"], current)


def commit_reclass(db: Db, truth: TruthService, key: FrameKey, instance: str,
                   annotator: str) -> dict:
    """Reclass ``instance``, logged with its inverse; what every edit returns plus ``report``."""
    try:
        report = reclass_instance(db, truth.tax, key.desktop, str(instance), None, annotator)
    except MergeError as refused:
        raise SessionRefusal(str(refused)) from refused
    plan = report.plan
    reach = _reach(db, key.desktop, key.view, plan, key.step)
    common = {"desktop": int(key.desktop), "view": key.view, "reach": reach,
              "annotator": annotator, "key": plan.old}
    payload = common | {"cls": plan.new_cls, "new": plan.new,
                        "latch": latch_of({"cage": plan.cage})}
    inverse = common | {"undo": plan.new}
    stats = settle(db, truth, key.desktop, key.view, reach, key.step)
    return {"changed": True, "report": report, "plan": plan, "affected": reach,
            "compiled": stats["compiled"], "rechecks": stats["rechecks"],
            "conflicts": stats["conflicts"], "problems": stats["problems"],
            "frame": stats["frame"], "op": Op(OP_KIND, payload, inverse)}


class ReclassMixin:
    """「其实是硬盘架」 / 「其实是卡扣」 of :class:`~tda.ui.session.AnnotationSession`."""

    def reclass_offers(self, instances=None) -> dict[str, str]:
        """``instance -> the class it may be said to be`` for every row that offers it.

        A logged ``drive_latch`` or ``drive_cage`` (not implied, not a draft,
        not a 框外零件); nothing on a components-gallery shot (task U10).
        """
        if not self.is_open or getattr(self, "gallery", False):
            return {}
        if instances is None:
            instances = self.db.instances(self.current().desktop)
        out = {}
        for key, rec in instances.items():
            if is_provisional(key) or is_context(rec) or is_implied(rec):
                continue
            target = reclass_target(rec.cls)
            if target:
                out[key] = target
        return out

    def reclass_offer(self, instance: str) -> Optional[str]:
        """The class ``instance`` may be said to be here, or ``None``."""
        if not self.is_open or getattr(self, "gallery", False):
            return None
        return reclass_offer(self.db, self.current().desktop, str(instance))

    def reclass_caption(self, instance: str) -> str:
        """「其实是硬盘架」 / 「其实是卡扣」 for ``instance``, ``""`` when not offered."""
        target = self.reclass_offer(instance)
        return CAPTION.get(target or "", "")

    def reclass_gate(self, instance: str) -> str:
        """Why the reclass would refuse ``instance`` here, ``""`` if not (writes nothing)."""
        if not self.is_open:
            return "没有打开的帧"
        if getattr(self, "gallery", False):
            return NO_VERB
        if is_context_key(instance):
            return CONTEXT_NO_VERB.format(instance=instance)
        return reclass_gate(self.db, self.tax, self.current().desktop, str(instance))

    def reclass_plan(self, instance: str) -> ReclassPlan:
        """What the reclass would write (the confirmation); nothing is written."""
        why = self.reclass_gate(instance)
        if why:
            raise SessionRefusal(why)
        try:
            return plan_reclass(self.db, self.tax, self.current().desktop, str(instance))
        except MergeError as refused:
            raise SessionRefusal(str(refused)) from refused

    def reclass(self, instance: str) -> dict:
        """「其实是硬盘架」 / 「其实是卡扣」 on ``instance``; undoable with Ctrl+Z.

        A part being drawn with pixels not yet committed is refused; one whose
        edit is untouched has it dropped -- its key is about to change.
        """
        if not self.is_open:
            raise SessionRefusal("没有打开的帧")
        if getattr(self, "gallery", False):
            raise SessionRefusal(NO_VERB)
        if is_context_key(instance):
            raise SessionRefusal(CONTEXT_NO_VERB.format(instance=instance))
        editing = self.layer.instance
        if editing == instance and self.layer.changed():
            raise SessionRefusal(VERB_NEEDS_CLEAN_LAYER.format(instance=instance))
        result = commit_reclass(self.db, self.truth, self.current(), str(instance),
                                self.annotator)
        if editing == instance:
            self.clear_edit()
        self._hidden.discard(str(instance))
        return self._after_edit(result)
