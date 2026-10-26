"""「这台没有这个零件」 as an undoable op of the session (task U34).

:mod:`tda.core.absent` writes it; this is the session's half: where the state
combo offers it (:meth:`AbsentMixin.absent_offered` -- the part's first
frame, a part with no shape anywhere), the write with its op, and the
undo/redo handler (:func:`apply_absent`), which both write the part's
step-0 events as the op's payload or inverse says and settle the view on
screen. The frozen frames of every view the change reaches are queued by the
core write; the one in hand is recompiled here, as for every other edit.

The instance list's state combo (the picker ``T``'s gesture shares) lists
:data:`~tda.core.absent.CHOICE` as 「一开始就不在（这台没有它）」 on the first
frame of a part nobody has drawn anywhere. It is no state assertion --
``removed`` stays the step table's to say
(:func:`tda.core.state_assert.assertable_states`) -- and ``T``'s own cycle
never stops on it (U34 round 2): the window routes the combo's choice here,
after asking.
"""
from __future__ import annotations

from typing import Optional

from tda.core import absent as AB
from tda.core.db import Db
from tda.core.log_identity import CHASSIS_KEY
from tda.core.model import FrameKey, is_context, is_extra, is_provisional
from tda.core.states import REMOVED, needs_geom
from tda.core.truth import TruthService
from tda.core.truth_inputs import InputCache, annotatable_steps, instances_of, state_of
from tda.ui.commands import Op
from tda.ui.session_api import VERB_NEEDS_CLEAN_LAYER, SessionRefusal
from tda.ui.session_ops import settle

__all__ = ["OP_KIND", "AbsentMixin", "apply_absent", "commit_absent"]

#: ``op`` kind of the session's undo stack, the same word as the op-log row's.
OP_KIND = AB.OP_KIND
#: Why the choice is not offered here: not the part's first frame.
NOT_FIRST = ("「{caption}」只在第一帧（第 {first} 步）上给：在那一帧选中它、在实例表的状态框里选，"
             "或 Steps → Instances 右键「这台没有这个零件」")


def _reach(db: Db, desktop: int, view: str, queued: dict, current: int) -> list[int]:
    """The frames of ``view`` to settle: the one in hand, and the frozen ones queued."""
    steps = set(queued.get(view) or ()) | {int(current)}
    return annotatable_steps(db, desktop, view, sorted(steps))


def apply_absent(db: Db, truth: TruthService, payload: dict,
                 current: Optional[int] = None) -> dict:
    """Both directions of the session's op: write the step-0 rows, settle the view."""
    desktop = int(payload["desktop"])
    queued = AB.write_rows(db, truth.tax, desktop, str(payload["instance"]),
                           payload.get("events") or [])
    view = str(payload["view"])
    steps = set(payload.get("reach") or ()) | set(queued.get(view) or ())
    if current is not None:
        steps.add(int(current))
    return settle(db, truth, desktop, view, annotatable_steps(db, desktop, view, sorted(steps)),
                  current)


def commit_absent(db: Db, truth: TruthService, key: FrameKey, instance: str,
                  annotator: str) -> dict:
    """Say ``instance`` was never in this machine; logged with its inverse."""
    try:
        _op_id, patch = AB.mark_absent(db, truth.tax, key.desktop, str(instance),
                                       annotator, view=key.view)
    except AB.AbsentError as refused:
        raise SessionRefusal(str(refused)) from refused
    reach = _reach(db, key.desktop, key.view, patch.rechecks, key.step)
    common = {"view": key.view, "reach": reach}
    payload = patch.payload() | common
    inverse = patch.inverse() | common
    stats = settle(db, truth, key.desktop, key.view, reach, key.step)
    return {"changed": True, "patch": patch, "affected": reach,
            "compiled": stats["compiled"], "rechecks": stats["rechecks"],
            "conflicts": stats["conflicts"], "problems": stats["problems"],
            "frame": stats["frame"], "op": Op(OP_KIND, payload, inverse)}


class AbsentMixin:
    """「这台没有这个零件」 of :class:`~tda.ui.session.AnnotationSession`."""

    def first_frame(self) -> Optional[int]:
        """The earliest frame this view can be annotated on, or ``None``."""
        available = list(getattr(self, "_available", ()) or ())
        return min(available) if available else None

    def absent_offers(self, rows_state=None, instances=None,
                      needs: Optional[dict] = None) -> set[str]:
        """The parts the state combo offers 「一开始就不在」 for on the open frame.

        Only on the first frame of this view, only for a real part of the
        machine (not a draft, a 框外零件 or a part added in S1) whose class
        has a ``removed`` state, that is in the picture here, and that has no
        shape in any view -- a part somebody drew is in the machine. Two
        queries for the shapes, once per call.
        """
        if not self.is_open or getattr(self, "gallery", False):
            return set()
        key = self.current()
        if key.step != self.first_frame():
            return set()
        cache = InputCache()
        if instances is None:
            instances = instances_of(self.db, key.desktop, cache)
        if rows_state is None:
            rows_state = state_of(self.db, self.tax, key.desktop, key.step, cache)
        if needs is None:
            needs = needs_geom(instances, rows_state, self.tax)
        candidates = [k for k in needs
                      if k in instances and not is_provisional(k)
                      and not is_context(instances[k]) and not is_extra(instances[k])
                      and REMOVED in self.tax.states_of(instances[k].cls)
                      and k != CHASSIS_KEY and instances[k].cls != CHASSIS_KEY]
        if not candidates:
            return set()
        shaped = AB.shaped_instances(self.db, key.desktop)
        return {k for k in candidates if k not in shaped}

    def absent_offered(self, instance: str) -> bool:
        """Does the state combo offer 「一开始就不在」 for ``instance`` here?"""
        return str(instance) in self.absent_offers()

    def absent_gate(self, instance: str) -> str:
        """Why 「一开始就不在」 cannot be said of ``instance`` here; ``""`` if it can."""
        if not self.is_open:
            return "没有打开的帧"
        if getattr(self, "gallery", False):
            return "零件合照里没有「一开始就不在」：到拆解序列的第一帧去说"
        first = self.first_frame()
        if self.current().step != first:
            return NOT_FIRST.format(caption=AB.CAPTION, first=first)
        why = AB.refusal(self.db, self.tax, self.current().desktop, str(instance))
        if why:
            return why
        if not self.absent_offered(instance):
            return f"{instance} 在第 {first} 步不在画面里，没有「{AB.SHORT}」可说"
        return ""

    def mark_absent_from_start(self, instance: str) -> dict:
        """「这台没有这个零件」 on ``instance``; undoable with ``Ctrl+Z``.

        A part being drawn with pixels not yet committed is refused; one whose
        edit is untouched has it dropped -- it is about to leave every frame.
        """
        why = self.absent_gate(instance)
        if why:
            raise SessionRefusal(why)
        layer = getattr(self, "layer", None)
        editing = getattr(layer, "instance", None)
        if editing == instance and layer.changed():
            raise SessionRefusal(VERB_NEEDS_CLEAN_LAYER.format(instance=instance))
        result = commit_absent(self.db, self.truth, self.current(), str(instance),
                               self.annotator)
        if editing == instance:
            self.clear_edit()
        self._hidden.discard(str(instance))
        return self._after_edit(result)
