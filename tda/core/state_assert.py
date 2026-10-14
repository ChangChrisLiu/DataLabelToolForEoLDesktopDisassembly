"""Saying what state a part is in at the frame being looked at (task U8).

The state log is derived from the step table (:func:`tda.core.states.events_from_actions`)
and that is the right authority on *what was done*. It is not always the whole
truth about *what the picture shows*: on D13/scan ``ram_latch.01`` was opened at
step 14 and is visibly closed again in a few middle frames, and nobody
"closed" it -- there is no such action, and there must not be one, because a
clip that springs back is not a disassembly step. What is wrong is the log's
account of the state, so the correction is a hand-written ``StateEvent``
(``auto=False``) on top of the derived log, which the core has always merged
after the derived events of the same step (:func:`tda.core.truth_inputs.events_of`).

**"At step k this instance is in state s."** That is the whole gesture, and it
is local by construction: the state at every other step stays what it was.

* The *restore step* is the first step after ``k`` that the table does not type
  ``dupli``: a duplicate frame shows the step before it, so it takes the state
  of ``k`` along (the ``dupli`` steps in between are :attr:`StatePatch.inherits`).
* The manual state events of the target at ``k``, at the ``dupli`` steps after
  it and at the restore step are dropped.
* At ``k`` an event ``held -> s`` is written, ``held`` being the state the log
  arrives at once everything else of step ``k`` has been applied (so its
  ``old`` is exactly what :func:`~tda.core.states.validate_events` checks).
* At the restore step an event ``s -> next`` brings the state back to what it
  was there before the change -- ``next`` is read off the *current* merged log
  -- unless it is ``s`` already, or the log's own events of that step get there
  by themselves (a removal: the board takes the clip out at 42 whatever state it
  was in at 41).
* A manual event the change turned into a no-op is dropped as well, so walking
  backwards over a closed run (21, then 20, then 19) leaves exactly two events,
  ``open -> closed`` at 19 and ``closed -> open`` at 22.

Refused, with a sentence an annotator can act on (:class:`StateAssertError`):
a state the class does not have, ``removed`` (when a part leaves is the step
table's to say), a part already removed at ``k``, an assertion *at* a ``dupli``
step (it is the frame before it, which is where to say it), and anything that
would make :func:`~tda.core.states.event_problems` find a problem in the merged
log it did not find before -- a derived event that no longer holds means the
action itself is on the wrong step, which is a step-table correction.

The wording takes a ``name`` function for the state values: the core has no
display vocabulary, the session passes :func:`tda.ui.class_names.state_zh`.

:func:`assert_state` writes and returns a :class:`StatePatch`, whose
:meth:`~StatePatch.payload` and :meth:`~StatePatch.inverse` are two *states* of
the same rows (the target's manual state events at the touched steps), so one
function -- :func:`write_events` -- is both the redo and the undo.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Optional

from tda.core.db import Db
from tda.core.model import Placement, StateEvent, StepType, is_provisional
from tda.core.states import (
    ATTR_STATE,
    INITIAL_STEP,
    LEAVES_REMOVED,
    OLD_MISMATCH,
    REMOVED,
    EventProblem,
    event_problems,
    events_from_actions,
    initial_overrides,
    state_at,
)
from tda.core.taxonomy import Taxonomy
from tda.core.truth_inputs import merge_events

__all__ = [
    "OP_ASSERT_STATE",
    "StateAssertError",
    "StatePatch",
    "assert_state",
    "assertable_states",
    "event_from_dict",
    "plan_assertion",
    "write_events",
]

#: ``op_log.kind`` (and :data:`tda.ui.commands.KINDS` entry) of one assertion.
OP_ASSERT_STATE = "assert_state"
IN_CHASSIS = Placement.IN_CHASSIS.value

#: How a state value is said; the session passes the annotator's words.
Namer = Callable[[str], str]


def _same(value: str) -> str:
    return str(value)


class StateAssertError(ValueError):
    """An assertion the log cannot take; the text is ready for the status line."""


def assertable_states(tax: Taxonomy, cls: str) -> list[str]:
    """The states of ``cls`` an annotator can set at a frame, in taxonomy order.

    The mask-bearing ones -- the states a part *in the picture* can be in --
    and only when there are at least two of them: with one there is nothing to
    choose, and a state that needs no mask (an unplugged connector) is a part
    that is not drawn, which is not what a frame-level correction is for.
    ``removed`` never: when a part leaves is the step table's to say.
    """
    try:
        states = tax.states_of(cls)
    except KeyError:
        return []
    out = [s for s in states
           if s != REMOVED and tax.needs_mask(cls, s, IN_CHASSIS)]
    return out if len(out) >= 2 else []


def event_from_dict(data: dict) -> StateEvent:
    """A :class:`StateEvent` back from the op log's JSON."""
    return StateEvent(
        desktop=int(data["desktop"]), step=int(data["step"]), target=str(data["target"]),
        attr=str(data["attr"]), old=data.get("old"), new=data.get("new"),
        evidence_view=data.get("evidence_view"), auto=bool(data.get("auto", False)),
    )


@dataclass
class StatePatch:
    """What one assertion changes, as two states of the same rows.

    ``steps`` are the steps whose manual ``state`` events of ``target`` the
    patch owns; ``before`` and ``after`` are those events (stored order) on
    either side of it. ``previous`` is the state at ``step`` before the change,
    ``inherits`` the ``dupli`` steps right after it (they show the same frame,
    so they carry the same state), ``next_step`` the restore step (``None``
    when the table ends first) and ``restores`` the state from there on, which
    the change leaves alone.
    """

    desktop: int
    target: str
    step: int
    state: str
    previous: str
    next_step: Optional[int]
    restores: Optional[str]
    inherits: list[int] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    before: list[StateEvent] = field(default_factory=list)
    after: list[StateEvent] = field(default_factory=list)
    evidence_view: Optional[str] = None

    @property
    def changed(self) -> bool:
        """Does the patch change anything at all? (Asserting what holds does not.)"""
        return self.previous != self.state

    @property
    def reach(self) -> list[int]:
        """The steps whose frames have to be brought up to date after it.

        The state moves at ``step`` and at the ``dupli`` steps that follow it;
        ``next_step`` is in because its own events were rewritten, which is
        cheap to re-check and keeps a frozen frame there honest.
        """
        return sorted({self.step, *self.inherits}
                      | ({self.next_step} if self.next_step is not None else set()))

    def _rows(self, events: list[StateEvent]) -> dict:
        return {"desktop": self.desktop, "target": self.target,
                "steps": [int(s) for s in self.steps],
                "events": [asdict(e) for e in events]}

    def _facts(self, state: str, previous: str) -> dict:
        return {"step": int(self.step), "state": state, "previous": previous,
                "inherits": [int(s) for s in self.inherits],
                "next_step": self.next_step, "restores": self.restores,
                "evidence_view": self.evidence_view, "reach": self.reach}

    def payload(self) -> dict:
        """The op-log payload: the rows after, plus what was asserted."""
        return self._rows(self.after) | self._facts(self.state, self.previous)

    def inverse(self) -> dict:
        """The rows as they were: :func:`write_events` of this undoes the patch."""
        return self._rows(self.before) | self._facts(self.previous, self.state)


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
def _mine(event: StateEvent, target: str, steps: Iterable[int]) -> bool:
    """Is ``event`` one of the target's hand-written state events at ``steps``?"""
    return (not event.auto and event.target == target and event.attr == ATTR_STATE
            and event.step in set(steps))


def _held(instances, derived, manual, step: int, target: str, tax: Taxonomy) -> str:
    """The target's state at ``step`` with this manual log on top of the derived one."""
    return state_at(instances, merge_events(derived, manual), step, tax)[target].state


def _idle(instances, derived, manual: list[StateEvent], target: str,
          tax: Taxonomy) -> set[int]:
    """``id()`` of the target's manual state events that change nothing.

    Replayed in merged order: an event whose ``new`` is the state already held
    when it is applied is a no-op. Initial states (step 0) are never counted --
    they are an added part's starting state, not a correction.
    """
    held = tax.default_state(instances[target].cls)
    idle: set[int] = set()
    mine = {id(e) for e in manual}
    for event in merge_events(derived, manual):
        if event.target != target or event.attr != ATTR_STATE:
            continue
        if id(event) in mine and event.step > INITIAL_STEP and event.new == held:
            idle.add(id(event))
        held = event.new
    return idle


def _contradiction(problem: EventProblem, name: Namer) -> str:
    """One new problem of the merged log, said as what the annotator has to do."""
    event = problem.event
    if problem.code == OLD_MISMATCH and event.auto:
        return (f"日志第 {event.step} 步把 {event.target} 从「{name(event.old)}」变成"
                f"「{name(event.new)}」，那一步之前它应是「{name(event.old)}」，这样改以后"
                f"却已经是「{name(problem.held)}」了。动作发生在哪一步，请在 Steps 模式里改")
    if problem.code == OLD_MISMATCH:
        return (f"第 {event.step} 步另有一条手工状态（{name(event.old)} → "
                f"{name(event.new)}）会接不上：先到第 {event.step} 步把它改过来")
    if problem.code == LEAVES_REMOVED:
        return f"{event.target} 第 {event.step} 步已经拆下，之后的状态不能再变"
    return f"这样改会和第 {event.step} 步的状态记录对不上"


def plan_assertion(db: Db, tax: Taxonomy, desktop: int, target: str, step: int,
                   new_state: str, evidence_view: Optional[str] = None,
                   name: Namer = _same) -> StatePatch:
    """Work out :func:`assert_state`'s patch without writing anything."""
    desktop, step, new_state = int(desktop), int(step), str(new_state)
    instances = db.instances(desktop)
    rec = instances.get(target)
    if rec is None:
        raise StateAssertError(f"D{desktop:02d} 没有零件 {target}")
    if is_provisional(target):
        raise StateAssertError(f"{target} 是 Label Studio 草稿，不是零件，没有状态")
    states = tax.states_of(rec.cls)
    if new_state not in states:
        raise StateAssertError(
            f"{target} 没有「{name(new_state)}」这个状态；可选："
            f"{'、'.join(name(s) for s in states)}")
    if new_state == REMOVED:
        raise StateAssertError(
            f"「{name(REMOVED)}」不能在这里设：零件哪一步被拆下由步骤表决定（在 Steps 模式里改）")
    rows = sorted(db.steps(desktop), key=lambda r: int(r.step))
    table = [int(r.step) for r in rows]
    if step not in table:
        raise StateAssertError(f"第 {step} 步不在 D{desktop:02d} 的步骤表里")
    dupli = {int(r.step) for r in rows if r.dupli or r.step_type == StepType.DUPLI.value}
    if step in dupli:
        # a duplicate frame shows the step before it: that is where to say it
        shown = [s for s in table if s < step and s not in dupli]
        where = f"第 {shown[-1]} 步" if shown else "它前面那一帧"
        raise StateAssertError(
            f"第 {step} 步在步骤表里是重复帧（dupli），和{where}是同一个样子，状态跟着它走："
            f"到{where}去改")
    later = [s for s in table if s > step]
    inherits = []
    for s in later:
        if s not in dupli:
            break
        inherits.append(s)
    nxt = next((s for s in later if s not in dupli), None)

    manual = [e for e in db.events(desktop) if not e.auto]
    derived = events_from_actions(instances, db.actions(desktop), tax,
                                  initial=initial_overrides(manual))
    current = merge_events(derived, manual)
    cur = state_at(instances, current, step, tax)[target].state
    restores = None if nxt is None else state_at(instances, current, nxt, tax)[target].state
    if cur == REMOVED:
        raise StateAssertError(f"{target} 在第 {step} 步已经拆下，没有状态可改")
    patch = StatePatch(desktop=desktop, target=target, step=step, state=new_state,
                       previous=cur, next_step=nxt, restores=restores,
                       inherits=inherits, evidence_view=evidence_view)
    if cur == new_state:
        return patch      # asserting what already holds: nothing to write

    touched = {step, *inherits} | ({nxt} if nxt is not None else set())
    kept = [e for e in manual if not _mine(e, target, touched)]
    added: list[StateEvent] = []
    held_k = _held(instances, derived, kept, step, target, tax)
    if held_k == REMOVED:
        raise StateAssertError(f"{target} 在第 {step} 步已经拆下，没有状态可改")
    if held_k != new_state:
        added.append(StateEvent(desktop, step, target, ATTR_STATE, held_k, new_state,
                                evidence_view=evidence_view, auto=False))
    if nxt is not None:
        held_n = _held(instances, derived, kept + added, nxt, target, tax)
        if held_n != restores:
            if held_n == REMOVED:
                raise StateAssertError(
                    f"{target} 在第 {nxt} 步已经拆下，第 {step} 步的状态接不回去")
            added.append(StateEvent(desktop, nxt, target, ATTR_STATE, held_n, restores,
                                    evidence_view=evidence_view, auto=False))
    after_manual = kept + added
    # a manual event this change turned into a no-op goes with it
    was_idle = _idle(instances, derived, manual, target, tax)
    now_idle = _idle(instances, derived, after_manual, target, tax) - was_idle
    dropped = [e for e in after_manual if id(e) in now_idle]
    after_manual = [e for e in after_manual if id(e) not in now_idle]
    touched |= {e.step for e in dropped}

    def keyed(found: list[EventProblem]) -> Counter:
        return Counter((id(p.event), p.code) for p in found)

    after_problems = event_problems(instances, merge_events(derived, after_manual), tax)
    new = keyed(after_problems) - keyed(event_problems(instances, current, tax))
    if new:
        first = next(p for p in after_problems if (id(p.event), p.code) in new)
        raise StateAssertError(_contradiction(first, name))
    merged_after = merge_events(derived, after_manual)
    got = state_at(instances, merged_after, step, tax)[target].state
    back = None if nxt is None else state_at(instances, merged_after, nxt, tax)[target].state
    if got != new_state or back != restores:   # pragma: no cover - an invariant
        raise StateAssertError(
            f"{target} 第 {step} 步的状态改不过来（得到 {got}，下一步 {back}）")

    patch.steps = sorted(touched)
    patch.before = [e for e in manual if _mine(e, target, touched)]
    patch.after = [e for e in after_manual if _mine(e, target, touched)]
    return patch


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def write_events(db: Db, desktop: int, target: str, steps: Iterable[int],
                 events: Iterable) -> None:
    """Make the target's manual state events at ``steps`` exactly ``events``.

    The one writer of an assertion, both ways round: ``events`` are
    :class:`StateEvent` objects or their op-log dicts, in the order they are to
    be stored. **Joins the caller's transaction.**
    """
    rows = [e if isinstance(e, StateEvent) else event_from_dict(e) for e in events]
    with db.transaction():
        db.delete_manual_events_at(int(desktop), str(target), steps, attr=ATTR_STATE)
        db.add_events(rows)


def assert_state(db: Db, tax: Taxonomy, desktop: int, target: str, step: int,
                 new_state: str, evidence_view: Optional[str] = None,
                 name: Namer = _same) -> StatePatch:
    """"At step ``step`` ``target`` is in ``new_state``": write it, return the patch.

    Nothing is written when the state already holds (``patch.changed`` is
    false). The op log is the caller's: the session records the patch with its
    inverse in the same transaction (:mod:`tda.ui.session_state`). ``name``
    says a state value in the refusals.
    """
    with db.transaction():
        patch = plan_assertion(db, tax, desktop, target, step, new_state, evidence_view,
                               name)
        if patch.changed:
            write_events(db, patch.desktop, patch.target, patch.steps, patch.after)
    return patch
