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

**A drive cage's adopted latch** (task U20 review, M2): D5/D6/D7's "HDD shield
locker" is the cage's latch by the import's rule while the sheet says it was
opened; ``Shift+T`` saying it was taken out leaves the cage without a latch.
So the flip runs the cage rule for that cage (:func:`tda.core.implied.cage_fix`)
in the same transaction and the same op: the implied latch the cage gets is
added (or, flipped back to 「其实只是打开了」, the cage's implied latches it no
longer needs are dropped -- one with shapes or hand-set states stays). The op
carries the records both ways (``cage``: ``add`` / ``drop``), so Ctrl+Z takes
the latch away with the verb and redo brings both back; the frames before the
cage comes out are settled with the rest (frozen ones queued for a re-check, as
after an import).

**「其实只是打开了」 hangs the latch on its cage** (task U21): the blue locker
is never taken off the cage -- it pivots open on the cage's big screw and leaves
with it -- so a logged ``drive_latch`` flipped back to opened that the rule
adopts for a logged ``drive_cage`` is adopted in the same op (``parent`` =
cage, ``attached``; the patch's ``repoint``), whether or not the import had
adopted it before, and the cage's untouched implied latch goes as above.
Ctrl+Z hangs it back off and brings the implied latch back.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from typing import Optional

from tda.core import cage_patch as P
from tda.core.db import Db
from tda.core.implied import LATCH_CLASS, OP_DELETE_IMPLIED, cage_fix, is_cage_latch
from tda.core.model import FrameKey, is_context_key
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
from tda.ui.context_text import NO_VERB as CONTEXT_NO_VERB
from tda.ui.gallery import NO_VERB
from tda.ui.session_api import VERB_NEEDS_CLEAN_LAYER, SessionRefusal
from tda.ui.session_ops import settle
from tda.ui.session_tasks import _has_shape, bench_roi_of
from tda.ui.steps_model import ANNOTATOR, EditError, StepTableData

__all__ = ["CAGE_PATCH", "OP_CORRECT_VERB", "VerbMixin", "apply_cage_patch",
           "apply_correct_verb", "commit_verb_fix", "has_work", "write_verb"]


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
    """Both directions of a ``correct_verb`` op: the cage's latches, the verb, then settle."""
    with db.transaction():
        apply_cage_patch(db, payload.get(CAGE_PATCH))
        write_verb(db, truth.tax, payload)
    return settle(db, truth, payload["desktop"], payload["view"], payload["reach"], current)


#: ``payload`` / ``inverse`` key of the cage latches a correction adds, drops
#: and hangs on the cage (task U20 review, M2; task U21): ``{"desktop", "cage",
#: "add": [records], "drop": [records], "repoint": [{"key", "before", "after"}],
#: "opens", "out_at"}`` -- the inverse is :func:`tda.core.cage_patch.mirror`.
CAGE_PATCH = "cage"
#: Kept for the callers that imported it from here (task U20).
has_work = P.has_work


def apply_cage_patch(db: Db, patch: Optional[dict]) -> None:
    """:func:`tda.core.cage_patch.apply_cage_patch`, refusing as the session does.

    **Joins the caller's transaction.**
    """
    try:
        P.apply_cage_patch(db, patch)
    except P.CagePatchError as refused:
        raise SessionRefusal(str(refused)) from refused


def _cage_patch(db: Db, tax, desktop: int, fix) -> Optional[dict]:
    """What the cage rule asks for after this flip (:func:`tda.core.implied.cage_fix`).

    Read on the stored table with the flipped verb, before anything is
    written. A surplus implied latch that has work of its own is left where it
    is (and said: ``kept``); a latch the rule adopts is hung on the cage
    (``repoint``, task U21); ``None`` when the rule asks for nothing. ``also``
    names the cage's other latches that are no implied ones -- D5's, added by
    hand: the same part, merged into the locker with 「合并到…」.
    """
    instances = db.instances(desktop)
    actions = [replace(a, verb=fix.new)
               if (a.step, a.idx, a.target) == (fix.step, fix.idx, fix.target) else a
               for a in db.actions(desktop)]
    found = cage_fix(instances, actions, tax, fix.target,
                     db.ops_of_kind(desktop, OP_DELETE_IMPLIED))
    if found is None:
        return None
    drop = [key for key in found.drop if not P.has_work(db, desktop, key)]
    repoint = [{"key": key, "before": [instances[key].parent, bool(instances[key].attached)],
                "after": [found.cage, True]} for key in found.adopt]
    patch = {"desktop": int(desktop), "cage": found.cage,
             "add": [asdict(rec) for rec in found.add],
             "drop": [asdict(instances[key]) for key in drop],
             "kept": [key for key in found.drop if key not in drop],
             "opens": found.opens, "out_at": found.out_at}
    if repoint:
        also = sorted(key for key, rec in instances.items()
                      if rec.cls == LATCH_CLASS and rec.parent == found.cage
                      and key not in found.adopt and not is_cage_latch(rec))
        patch.update(repoint=repoint, adopted=list(found.adopt), also=also)
    return patch


def _mirror(patch: Optional[dict]) -> Optional[dict]:
    """The other direction of a cage patch: what was added is dropped, and back."""
    return P.mirror(patch)


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
    # the cage of an adopted latch gets (or no longer needs) its implied latch (U20, M2)
    cage = _cage_patch(db, truth.tax, key.desktop, fix)
    moves = cage is not None and bool(cage["add"] or cage["drop"] or cage.get("repoint"))
    out_at = cage.get("out_at") if moves else None
    reach = annotatable_steps(db, key.desktop, key.view, [
        s for s in table
        if s >= fix.step or (moves and (out_at is None or s < int(out_at)))])
    lost = (_frames(db, truth.tax, key.desktop, key.view, instance,
                    [s for s in reach if s >= fix.step], drawn=True)
            if fix.new == REMOVE else [])
    # who made it travels with both directions: undo and redo are credited to them
    payload = fix.rows(fix.new) | {"view": key.view, "reach": reach, "annotator": annotator}
    inverse = fix.rows(fix.old) | {"view": key.view, "reach": reach, "annotator": annotator}
    if moves:
        payload[CAGE_PATCH], inverse[CAGE_PATCH] = cage, _mirror(cage)
    try:
        with db.transaction():
            apply_cage_patch(db, payload.get(CAGE_PATCH))
            messages = write_verb(db, truth.tax, payload, annotator)
            db.log_op(key.desktop, key.view, OP_CORRECT_VERB, payload, inverse, annotator)
    except EditError as refused:
        raise SessionRefusal(str(refused)) from refused
    stats = settle(db, truth, key.desktop, key.view, reach, key.step)
    todo = (_frames(db, truth.tax, key.desktop, key.view, instance,
                    [s for s in reach if s >= fix.step], drawn=False)
            if fix.new == OPEN else [])
    return {"changed": True, "fix": fix, "affected": reach, "lost": lost, "todo": todo,
            "messages": list(messages), "compiled": stats["compiled"],
            "rechecks": stats["rechecks"], "conflicts": stats["conflicts"],
            "problems": stats["problems"], "frame": stats["frame"],
            "cage": _cage_result(cage),
            "op": Op(OP_CORRECT_VERB, payload, inverse)}


def _cage_result(patch: Optional[dict]) -> Optional[dict]:
    """What the status line says about the cage (``None``: nothing happened to one)."""
    if not patch:
        return None
    out = {"cage": patch["cage"], "added": [r["key"] for r in patch["add"]],
           "dropped": [r["key"] for r in patch["drop"]], "kept": list(patch.get("kept") or ()),
           "opens": patch.get("opens")}
    if patch.get("adopted"):        # hung on the cage (U21); its other latches by hand
        out.update(adopted=list(patch["adopted"]), also=list(patch.get("also") or ()))
    return out


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
        """Add ``verb_fix`` to instance-list rows (``None`` where not offered).

        A screw's row carries the captive flip instead (task U14,
        :meth:`tda.ui.session_captive.CaptiveMixin.captive_offers`):
        ``"unscrewed"`` / ``"stays_in"``, the same button and key.
        """
        found = self.verb_offers(state, instances)
        captive = getattr(self, "captive_offers", None)
        screws = captive(state, instances) if callable(captive) else {}
        # 「其实是硬盘架」 / 「其实是卡扣」 on the same button (task U21)
        bay = getattr(self, "reclass_offers", None)
        reclass = bay(instances) if callable(bay) else {}
        for row in rows:
            key = str(row.get("key", ""))
            row["verb_fix"] = found.get(key) or screws.get(key)
            row["reclass"] = reclass.get(key)
        return rows

    def verb_gate(self, instance: str) -> str:
        """Why :meth:`correct_verb` would refuse ``instance`` here, ``""`` if not.

        The core's plan as a dry run, writing nothing: what the palette's
        greyed 「其实拆下」 button says (task U13). The editing layer's gate is
        the window's, which knows it without a pass over the pixels; a refusal
        of the step table itself is still only found by the write.
        """
        if not self.is_open:
            return "没有打开的帧"
        if getattr(self, "gallery", False):
            return NO_VERB
        if is_context_key(instance):
            return CONTEXT_NO_VERB.format(instance=instance)     # task U18
        key = self.current()
        try:
            plan_verb_fix(self.db, self.tax, key.desktop, str(instance), key.step,
                          name=state_zh)
        except VerbFixError as refused:
            return str(refused)
        return ""

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
        if is_context_key(instance):
            raise SessionRefusal(CONTEXT_NO_VERB.format(instance=instance))   # task U18
        editing = self.layer.instance
        if editing == instance and self.layer.changed():
            raise SessionRefusal(VERB_NEEDS_CLEAN_LAYER.format(instance=instance))
        result = commit_verb_fix(self.db, self.truth, self.current(), str(instance), to,
                                 self.annotator)
        if editing == instance:
            self.clear_edit()
        return self._after_edit(result)
