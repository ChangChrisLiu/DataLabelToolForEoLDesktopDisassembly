"""「其实拆下来了」 / 「其实只是打开了」 as an undoable op (task U9).

:mod:`tda.core.verb_fix` decides which action's verb flips; this module is the
session's half: the gate the instance list and the task card show
(:meth:`VerbMixin.verb_offers`), the write -- through the step table's own path,
``StepTableData.load`` → ``apply_edit(step, "verb", ...)`` → ``save``, what an
edit in Steps mode does, so the action is validated by the same rules and the
automatic event log is recompiled the same way (only the constraint graph is
left to Apply, see :func:`write_verb`) -- the op-log row with its inverse, the
undo/redo handler, and :func:`~tda.ui.session_ops.settle` over every frame of
**this view** from the action's step on: the part's rows leave the card (or
come back) from there, and a frozen frame among them is queued for a re-check.
The other views follow on their next opening, which re-checks every frozen
frame against its inputs -- as for a state assertion (task U8).

Both directions of the op are the same record with a different verb, so one
handler (:func:`apply_correct_verb`) is the redo and the undo.
"""
from __future__ import annotations

from typing import Optional

from tda.core.db import Db
from tda.core.model import FrameKey
from tda.core.states import needs_geom
from tda.core.truth import TruthService
from tda.core.truth_inputs import InputCache, annotatable_steps, instances_of, state_of
from tda.core.verb_fix import (
    OP_CORRECT_VERB,
    OPEN,
    REMOVE,
    VerbFixError,
    offers,
    plan_verb_fix,
)
from tda.ui.class_names import state_zh
from tda.ui.commands import Op
from tda.ui.gallery import NO_VERB
from tda.ui.session_api import SessionRefusal
from tda.ui.session_ops import settle
from tda.ui.session_tasks import _has_shape, bench_roi_of
from tda.ui.steps_model import ANNOTATOR, EditError, StepTableData

__all__ = ["OP_CORRECT_VERB", "VerbMixin", "apply_correct_verb", "commit_verb_fix",
           "write_verb"]


def write_verb(db: Db, tax, rows: dict, annotator: Optional[str] = None) -> list[str]:
    """Give one action the verb ``rows["verb"]`` through the S1 path; joins a transaction.

    Returns what :meth:`StepTableData.save` returns: the event log's problems
    after the write. The action is found by step, ``idx`` and target, so a
    redo after anything else moved in the row still lands on it. The op's
    own ``annotator`` is who the write is credited to, on undo and redo too.

    The constraint graph is left alone (``save(graph=False)``): it is derived
    from the instances and the edges, never from a verb, so a flip can neither
    move an edge nor make a deadlock -- the deadlock refusal lives in that same
    derivation and has nothing to refuse here. Catching a stale graph up is
    an S1 Apply's job, not a side effect of ``Shift+T`` that Ctrl+Z would not
    take back.
    """
    who = str(annotator or rows.get("annotator") or ANNOTATOR)
    data = StepTableData.load(db, int(rows["desktop"]), tax, who)
    step, target, idx = int(rows["step"]), str(rows["target"]), int(rows["idx"])
    found = [i for i, action in enumerate(data.row(step).actions)
             if action.target == target and int(action.idx) == idx]
    if not found:
        raise SessionRefusal(f"第 {step} 步已经没有 {target} 的动作了（步骤表在别处改过）")
    data.apply_edit(step, "verb", str(rows["verb"]), found[0])
    return data.save(db, graph=False)


def apply_correct_verb(db: Db, truth: TruthService, payload: dict,
                       current: Optional[int] = None) -> dict:
    """Both directions of a ``correct_verb`` op: set the verb, then settle."""
    with db.transaction():
        write_verb(db, truth.tax, payload)
    return settle(db, truth, payload["desktop"], payload["view"], payload["reach"], current)


def _frames(db: Db, tax, desktop: int, view: str, instance: str, steps,
            drawn: bool) -> list[int]:
    """The ``steps`` where ``instance`` is asked for and has (``drawn``) or lacks a shape."""
    cache = InputCache()
    instances = instances_of(db, desktop, cache)
    out = []
    for step in steps:
        key = FrameKey(desktop, int(step), view)
        needs = needs_geom(instances, state_of(db, tax, desktop, int(step), cache), tax,
                           bench_roi=bench_roi_of(db, key, cache))
        if instance in needs and _has_shape(db, tax, desktop, view, instance, int(step),
                                            cache) == drawn:
            out.append(int(step))
    return out


def commit_verb_fix(db: Db, truth: TruthService, key: FrameKey, instance: str,
                    to: Optional[str], annotator: str) -> dict:
    """Flip the verb that opened (or took out) ``instance``, logged with its inverse.

    Returns what every edit returns (``affected``, ``rechecks``, ``problems``,
    ``frame``, ``op``) plus the :class:`~tda.core.verb_fix.VerbFix` as ``fix``,
    ``lost`` -- the frames of this view whose drawn shape of the part is no
    longer used (it is out of the machine there now) -- and ``todo``, the
    frames it now has to be drawn on (it never left). A refusal of the core or
    of the step table is a :class:`SessionRefusal`.
    """
    try:
        fix = plan_verb_fix(db, truth.tax, key.desktop, instance, key.step, to, name=state_zh)
    except VerbFixError as refused:
        raise SessionRefusal(str(refused)) from refused
    table = sorted({int(r.step) for r in db.steps(key.desktop)})
    reach = annotatable_steps(db, key.desktop, key.view, [s for s in table if s >= fix.step])
    lost = (_frames(db, truth.tax, key.desktop, key.view, instance, reach, drawn=True)
            if fix.new == REMOVE else [])
    # who made it travels with both directions: undo and redo are credited to them
    payload = fix.rows(fix.new) | {"view": key.view, "reach": reach, "annotator": annotator}
    inverse = fix.rows(fix.old) | {"view": key.view, "reach": reach, "annotator": annotator}
    try:
        with db.transaction():
            messages = write_verb(db, truth.tax, payload, annotator)
            db.log_op(key.desktop, key.view, OP_CORRECT_VERB, payload, inverse, annotator)
    except EditError as refused:
        raise SessionRefusal(str(refused)) from refused
    stats = settle(db, truth, key.desktop, key.view, reach, key.step)
    todo = (_frames(db, truth.tax, key.desktop, key.view, instance, reach, drawn=False)
            if fix.new == OPEN else [])
    return {"changed": True, "fix": fix, "affected": reach, "lost": lost, "todo": todo,
            "messages": list(messages), "compiled": stats["compiled"],
            "rechecks": stats["rechecks"], "conflicts": stats["conflicts"],
            "problems": stats["problems"], "frame": stats["frame"],
            "op": Op(OP_CORRECT_VERB, payload, inverse)}


class VerbMixin:
    """The verb correction of :class:`~tda.ui.session.AnnotationSession`."""

    def verb_offers(self, state=None, instances=None) -> dict[str, str]:
        """``instance -> "remove" | "open"`` for the parts offered it on the open frame.

        :func:`tda.core.verb_fix.offers`: a cover or removable latch whose
        state here is what its own ``open`` / ``remove`` left. ``state`` and
        ``instances`` are the frame's, when the caller has them already.
        """
        if not self.is_open or getattr(self, "gallery", False):
            return {}     # a components-gallery shot is not a step of the log (U10)
        key = self.current()
        if state is None or instances is None:
            cache = InputCache()
            instances = instances_of(self.db, key.desktop, cache)
            state = state_of(self.db, self.tax, key.desktop, key.step, cache)
        return offers(self.tax, instances, state, self.db.actions(key.desktop), key.step)

    def _with_verb_offers(self, rows: list[dict], state, instances) -> list[dict]:
        """Add ``verb_fix`` to instance-list rows (``None`` where not offered)."""
        found = self.verb_offers(state, instances)
        for row in rows:
            row["verb_fix"] = found.get(str(row.get("key", "")))
        return rows

    def correct_verb(self, instance: str, to: Optional[str] = None) -> dict:
        """「其实拆下来了」 (``to="remove"``) / 「其实只是打开了」 (``"open"``); undoable.

        Without ``to`` the direction is the part's state here: open becomes
        taken out, removed becomes opened. A part being drawn with pixels not
        yet committed is refused; one whose edit is untouched has it dropped,
        since the frame may no longer have the part at all.
        """
        if not self.is_open:
            raise SessionRefusal("没有打开的帧")
        if getattr(self, "gallery", False):
            # its number is a shot, not a step: a verb would be changed at the
            # wrong step of the teardown (U10)
            raise SessionRefusal(NO_VERB)
        editing = self.layer.instance
        if editing == instance and self.layer.changed():
            raise SessionRefusal(f"{instance} 正在画、还没提交：先 Enter 提交或 Esc 放弃，再改")
        result = commit_verb_fix(self.db, self.truth, self.current(), str(instance), to,
                                 self.annotator)
        if editing == instance:
            self.clear_edit()
        return self._after_edit(result)
