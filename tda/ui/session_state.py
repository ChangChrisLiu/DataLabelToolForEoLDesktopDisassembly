"""Setting a part's state at the frame in hand, as an undoable op (task U8).

:mod:`tda.core.state_assert` decides what "at step k this part is in state s"
writes; this module is the session's half of it: which parts the gesture is
offered for on the open frame (:meth:`StateMixin.state_choices`), the op-log
row with its inverse, the undo/redo handler, and :func:`settle` -- the same
rule every other write goes through, so the frame in hand is recompiled and a
frozen frame the change reaches is queued for a re-check.

Nothing the session keeps goes stale behind it: the merged event log is read
fresh by every reader (:func:`tda.core.truth_inputs.events_of` builds an
:class:`~tda.core.truth_inputs.InputCache` per call or per sweep), the task
card is computed on demand, and the compiled-frame cache moves to a new epoch
through :meth:`~tda.ui.session.AnnotationSession._after_edit`. The stored
automatic events need no recompile either: they depend on the actions and the
initial (step-0) states only, and an assertion is never at step 0.
"""
from __future__ import annotations

from typing import Optional

from tda.core.db import Db
from tda.core.model import FrameKey, Placement, is_provisional
from tda.core.state_assert import (
    OP_ASSERT_STATE,
    StateAssertError,
    assert_state,
    assertable_states,
    write_events,
)
from tda.core.split_rows import open_split_steps, reverse_walk
from tda.core.states import FrameState, needs_geom
from tda.core.truth import TruthService
from tda.core.truth_inputs import InputCache, annotatable_steps, instances_of, state_of
from tda.ui.class_names import state_zh
from tda.ui.commands import Op
from tda.ui.session_api import SessionRefusal
from tda.ui.session_ops import settle

__all__ = ["OP_ASSERT_STATE", "StateMixin", "apply_assert_state", "commit_assertion",
           "gate", "run_around"]

IN_CHASSIS = Placement.IN_CHASSIS.value


def gate(instances: dict, state: FrameState, tax, instance: str,
         step: int, needs: Optional[dict] = None) -> tuple[list[str], str]:
    """``(choices, why not)`` of the state gesture for one part at one step.

    ``choices`` are :func:`~tda.core.state_assert.assertable_states` of its
    class when the part is in the picture at ``step`` -- in the chassis, in a
    state that needs a mask -- and ``[]`` otherwise, with the sentence saying
    why. Asked by the instance list for every row (``needs``, the frame's
    :func:`~tda.core.states.needs_geom`, computed once for all of them) and by
    ``T`` for one.
    """
    rec = instances.get(instance)
    if rec is None or is_provisional(instance):
        return [], f"{instance} 不是这台机器的零件，没有状态可改"
    choices = assertable_states(tax, rec.cls)
    if not choices:
        return [], f"{instance}（{rec.cls}）只有一种要画的状态，没有可换的"
    held = state.get(instance)
    needs = needs_geom(instances, state, tax) if needs is None else needs
    if (held is None or held.state not in choices or held.placement != IN_CHASSIS
            or instance not in needs):
        return [], f"{instance} 在第 {step} 步不在画面里（已拆下或跟着别的零件离开），没有状态可改"
    return list(choices), ""


def run_around(db: Db, tax, desktop: int, instance: str,
               step: int) -> tuple[int, int, Optional[tuple[int, str]]]:
    """``(first, last, after)``: the run of steps sharing ``step``'s state.

    ``after`` is ``(step, state)`` of the first step past the run, ``None``
    when the run reaches the end of the table. What the status line says after
    a change: closed at 20 next to a 21 that already was is "第 20–21 步都是
    关上；第 22 步起恢复 打开", not "第 21 步起本来就是 关上".
    """
    cache = InputCache()
    table = sorted({int(r.step) for r in db.steps(desktop)})
    held = {s: state_of(db, tax, desktop, s, cache).get(instance) for s in table}
    value = held[step].state if held.get(step) is not None else None

    def same(index: int) -> bool:
        found = held[table[index]]
        return found is not None and found.state == value

    first = last = table.index(step)
    while first > 0 and same(first - 1):
        first -= 1
    while last + 1 < len(table) and same(last + 1):
        last += 1
    after = None
    if last + 1 < len(table) and held[table[last + 1]] is not None:
        after = (table[last + 1], held[table[last + 1]].state)
    return table[first], table[last], after


def redraw_needed(db: Db, tax, key: FrameKey, instance: str, first: int,
                  last: int) -> dict:
    """Where the state run ``first..last`` still shows the other state's shape.

    ``{"top", "below"}``: an open ✂ row at the run's last frame (its state
    differs from the next one's, one version on both sides) and one at the walk
    frame just under it. Either means the run's shape has to be drawn; the
    commit rule of round 3 then cuts the version at those edges by itself, so
    the status line only has to say *which key* (:func:`tda.ui.app_state.state_note`).
    """
    walk = reverse_walk(db, key.desktop, key.view)
    under = [s for s in walk if s < first]
    low = under[-1] if under else first
    opened = set(open_split_steps(db, tax, key.desktop, key.view, instance, low, last + 1,
                                  walk=walk))
    return {"top": last in opened, "below": bool(under) and low in opened}


def apply_assert_state(db: Db, truth: TruthService, payload: dict,
                       current: Optional[int] = None) -> dict:
    """Both directions of an ``assert_state`` op: write its rows, then settle."""
    with db.transaction():
        write_events(db, payload["desktop"], payload["target"], payload["steps"],
                     payload["events"])
    return settle(db, truth, payload["desktop"], payload["view"], payload["reach"], current)


def commit_assertion(db: Db, truth: TruthService, key: FrameKey, instance: str,
                     state: str, annotator: str) -> dict:
    """Write "at this frame ``instance`` is ``state``" and log it with its inverse.

    Returns what every edit returns (``affected``, ``rechecks``, ``problems``,
    ``frame``, ``op``) plus ``changed``, the :class:`StatePatch` as ``patch``
    and :func:`run_around`'s answer as ``run`` / ``after``. Asserting the
    state that already holds writes nothing and has no op. A refusal of the
    core is a :class:`SessionRefusal` here, so the window shows its sentence
    instead of a traceback.
    """
    try:
        with db.transaction():
            patch = assert_state(db, truth.tax, key.desktop, instance, key.step, state,
                                 evidence_view=key.view, name=state_zh)
            if not patch.changed:
                return {"changed": False, "patch": patch, "affected": [], "problems": {},
                        "rechecks": [], "conflicts": 0, "frame": None}
            payload = patch.payload() | {"view": key.view}
            inverse = patch.inverse() | {"view": key.view}
            db.log_op(key.desktop, key.view, OP_ASSERT_STATE, payload, inverse, annotator)
    except StateAssertError as refused:
        raise SessionRefusal(str(refused)) from refused
    affected = annotatable_steps(db, key.desktop, key.view, patch.reach)
    stats = settle(db, truth, key.desktop, key.view, affected, key.step)
    first, last, after = run_around(db, truth.tax, key.desktop, instance, key.step)
    return {"changed": True, "patch": patch, "affected": affected,
            "run": (first, last), "after": after,
            "redraw": redraw_needed(db, truth.tax, key, instance, first, last),
            "compiled": stats["compiled"], "rechecks": stats["rechecks"],
            "conflicts": stats["conflicts"], "problems": stats["problems"],
            "frame": stats["frame"], "op": Op(OP_ASSERT_STATE, payload, inverse)}


class StateMixin:
    """The state gesture of :class:`~tda.ui.session.AnnotationSession`."""

    def state_choices(self, instance: str) -> list[str]:
        """States ``instance`` can be set to on the open frame; ``[]`` = not offered."""
        return self.state_gate(instance)[0]

    def state_gate(self, instance: str) -> tuple[list[str], str]:
        """:func:`gate` for the open frame: ``(choices, why not)``."""
        if not self.is_open:
            return [], "没有打开的帧"
        key = self.current()
        cache = InputCache()
        return gate(instances_of(self.db, key.desktop, cache),
                    state_of(self.db, self.tax, key.desktop, key.step, cache),
                    self.tax, str(instance), key.step)

    def _with_state_choices(self, rows: list[dict], state: FrameState,
                            instances: dict) -> list[dict]:
        """Add ``state_choices`` to instance-list rows (``[]`` where not offered)."""
        step = self.current().step
        needs = needs_geom(instances, state, self.tax)
        for row in rows:
            row["state_choices"] = gate(instances, state, self.tax,
                                        str(row.get("key", "")), step, needs)[0]
        return rows

    def next_state(self, instance: str) -> Optional[str]:
        """The state ``T`` moves ``instance`` to: the one after its current one."""
        choices, _why = self.state_gate(instance)
        if not choices:
            return None
        held = state_of(self.db, self.tax, self.desktop, self._step).get(instance)
        index = choices.index(held.state) if held is not None and held.state in choices else -1
        return choices[(index + 1) % len(choices)]

    def assert_state(self, instance: str, state: str) -> dict:
        """"At this frame ``instance`` is ``state``" (task U8); undoable.

        Refused (:class:`SessionRefusal`) where :meth:`state_gate` does not
        offer it, on a frame this view has no usable image of (nobody can see
        the state there: the sentence names the nearest frame that has one, or
        another view -- the state is shared by all four), and wherever the core
        refuses. The state from the next step on is left as it was, which is
        what makes one keystroke per frame enough to mark a run of frames.

        ``restore_unseen`` in the result says the restore step has no image in
        this view: the log's value is kept there, and the window says so.
        """
        key = self.current()
        if not self._has_picture(key.step):
            seen = [s for s in self._available if self._has_picture(s)]
            near = min(seen, key=lambda s: (abs(s - key.step), s)) if seen else None
            where = (f"到有图像的第 {near} 步，或换个视角去改" if near is not None
                     else "换个视角去改")
            raise SessionRefusal(
                f"第 {key.step} 步在 {self.view} 没有可用的图像，看不到它的状态：{where}"
                f"（状态四个视角共用）")
        choices, why = self.state_gate(instance)
        if not choices:
            raise SessionRefusal(why)
        if state not in choices:
            raise SessionRefusal(
                f"{instance} 在这里只能设为 {'、'.join(state_zh(c) for c in choices)}，"
                f"不能设为 {state_zh(state)}")
        result = commit_assertion(self.db, self.truth, key, str(instance),
                                  str(state), self.annotator)
        if not result["changed"]:
            return result
        nxt = result["patch"].next_step
        result["restore_unseen"] = nxt is not None and not self._has_picture(nxt)
        return self._after_edit(result)

    def _has_picture(self, step: int) -> bool:
        """Does this view have a usable image of ``step``? (``_editable_frame``'s test)"""
        return int(step) in self._available and self.image_path(int(step)) is not None
