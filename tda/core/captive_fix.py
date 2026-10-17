"""Unscrewed and taken out, or left in its part: a screw's ``captive`` (task U14).

In this tool ``captive`` means **"unscrewing does not take it out of the
chassis"** -- an ``unscrew`` leaves a captive screw ``loosened`` in place and
it goes out with its ``parent``; a plain one is ``removed`` there and then
(``taxonomy.yaml``'s conditional ``unscrew``). It is not a statement about the
screw's mechanics. The importer guesses it
(:func:`tda.core.log_identity.is_captive`: every CPU-cooler screw of a Dell),
and the guess is wrong on some machines: on D3 the five fan screws were taken
out at steps 2-6, so the card asked for them on frames 7-15, while D2's really
did stay in the fan. The sheet says the same thing about both; only the
pictures decide, so the annotator says it with the key that already corrects
the log about a part (``Shift+T``, task U9).

This module only *plans* the flip (:func:`plan_captive_fix`): which screw,
the step its truth changes from (its ``unscrew``), the part it stays in when
it becomes captive, and every reason it cannot be done, in a sentence the
annotator can act on. The write goes through the step table's own path
(:mod:`tda.ui.session_captive`), as an edit of the ``Captive`` cell in Steps
mode would.

What the flip writes on the instance:

* **not captive -> captive** (「其实留在零件里」): ``captive=True``,
  ``captive_source="manual"``; ``parent`` is the one it has, or -- when blank
  -- what it fastens (the importer's own rule,
  :func:`tda.core.graph_infer.infer_relational_fields`), and ``attached`` is
  ticked: a captive screw that does not ride out with a part would stand
  ``loosened`` in the chassis for ever.
* **captive -> not captive** (「其实拧下来了」): ``captive=False``,
  ``captive_source="manual"``, ``attached`` unticked -- it no longer leaves
  with anything; ``parent`` is kept, inert, so a flip back finds it.

Refused (:class:`CaptiveFixError`): not a screw, a Label Studio draft, a screw
the log never unscrews (the flag would change nothing), a captive screw with no
part to stay in, and anything that would make
:func:`~tda.core.states.event_problems` find a problem it did not find before
(a hand-written state after the unscrew that cannot follow a removal).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

from tda.core.captive_replay import OP_CAPTIVE_FIX as _REPLAYED_KIND
from tda.core.db import Db
from tda.core.model import ActionRec, InstanceRec, is_provisional
from tda.core.states import (
    LEAVES_REMOVED,
    event_problems,
    events_from_actions,
    initial_overrides,
)
from tda.core.taxonomy import Taxonomy
from tda.core.truth_inputs import events_of, merge_events

__all__ = [
    "CAPTIVE_FIELDS",
    "CaptiveFix",
    "CaptiveFixError",
    "OP_CAPTIVE_FIX",
    "STAYS_IN",
    "UNSCREWED",
    "captive_offers",
    "plan_captive_fix",
    "record_of",
    "unscrew_step",
]

#: ``op_log.kind`` (and :data:`tda.ui.commands.KINDS` entry) of one flip; the
#: importer's replay (:mod:`tda.core.captive_replay`) reads these rows.
OP_CAPTIVE_FIX = _REPLAYED_KIND
#: The two directions, as a row's ``verb_fix`` says them (the instance list's
#: button, the card): captive -> not, and not -> captive.
UNSCREWED = "unscrewed"      # 「其实拧下来了」
STAYS_IN = "stays_in"        # 「其实留在零件里」
SCREW = "screw"
UNSCREW = "unscrew"
SUCCESS = "success"
MANUAL = "manual"
#: What one flip may change on the instance record, and its inverse restores.
CAPTIVE_FIELDS = ("captive", "captive_source", "parent", "attached")

Namer = Callable[[str], str]


def _same(value: str) -> str:
    return str(value)


class CaptiveFixError(ValueError):
    """A flip the log cannot take; the text is ready for the status line."""


def record_of(rec: InstanceRec) -> dict:
    """The fields a flip touches, as stored (``None`` for an absent attribute)."""
    attrs = rec.attrs or {}
    return {"captive": attrs.get("captive"), "captive_source": attrs.get("captive_source"),
            "parent": rec.parent or None, "attached": bool(rec.attached)}


@dataclass(frozen=True)
class CaptiveFix:
    """One screw whose ``captive`` flips; its truth changes from ``step`` on.

    ``before`` / ``after`` are :func:`record_of` the instance; ``parent`` is
    the part a captive screw stays in (after a flip to captive) or stayed in
    (before a flip away from it), for the status line. ``at`` is the frame the
    annotator asked from.
    """

    desktop: int
    target: str
    step: int
    new: bool
    at: int
    parent: Optional[str] = None
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)

    @property
    def direction(self) -> str:
        return STAYS_IN if self.new else UNSCREWED

    @property
    def old(self) -> bool:
        return bool(self.before.get("captive"))

    @property
    def filled(self) -> bool:
        """Did the flip take its ``parent`` from ``fastens``?"""
        return not self.before.get("parent") and bool(self.after.get("parent"))

    def rows(self, record: dict) -> dict:
        """The op-log record with ``record``'s fields: both directions are one."""
        return {"desktop": int(self.desktop), "target": self.target, "step": int(self.step),
                "at": int(self.at), **{name: record.get(name) for name in CAPTIVE_FIELDS}}


def unscrew_step(actions: list[ActionRec], target: str) -> Optional[int]:
    """The step of ``target``'s first successful ``unscrew``, or ``None``."""
    steps = [a.step for a in actions
             if a.target == target and a.verb == UNSCREW and a.result == SUCCESS]
    return min(steps) if steps else None


def captive_offers(instances: dict[str, InstanceRec],
                   actions: list[ActionRec]) -> dict[str, str]:
    """``screw -> UNSCREWED | STAYS_IN`` for every screw the flip is offered on.

    The cheap gate the instance list and the card ask on every frame: a real
    screw the log unscrews. Captive ones are offered 「其实拧下来了」, the
    others 「其实留在零件里」, on any frame -- it is a property of the screw,
    not of the frame. Everything :func:`plan_captive_fix` refuses beyond that
    is said when the key is pressed.
    """
    unscrewed = {a.target for a in actions if a.verb == UNSCREW and a.result == SUCCESS}
    return {key: (UNSCREWED if (rec.attrs or {}).get("captive") else STAYS_IN)
            for key, rec in instances.items()
            if rec.cls == SCREW and key in unscrewed and not is_provisional(key)}


def _signature(problem) -> tuple:
    event = problem.event
    return (event.step, event.target, event.attr, event.old, event.new, event.auto,
            problem.code)


def _flipped(rec: InstanceRec, after: dict) -> InstanceRec:
    """A copy of ``rec`` carrying ``after``'s fields."""
    attrs = dict(rec.attrs or {})
    for name in ("captive", "captive_source"):
        if after.get(name) is None:
            attrs.pop(name, None)
        else:
            attrs[name] = after[name]
    return replace(rec, attrs=attrs, parent=after.get("parent"),
                   attached=bool(after.get("attached")))


def _contradiction(problem, name: Namer) -> str:
    """A new problem of the merged log, said as what the annotator has to do."""
    event = problem.event
    if not event.auto and problem.code == LEAVES_REMOVED:
        return (f"{event.target} 第 {event.step} 步有手动设的状态（T：{name(event.old)} → "
                f"{name(event.new)}），拧下拿走以后就接不上了：先到第 {event.step} 步按 T "
                f"把它改回去")
    return (f"这样改会和第 {event.step} 步的状态记录对不上（{problem.text}）："
            f"请在 Steps 模式里改")


def plan_captive_fix(db: Db, tax: Taxonomy, desktop: int, target: str, step: int,
                     to: Optional[str] = None, name: Namer = _same) -> CaptiveFix:
    """Flip ``target``'s ``captive``; the direction is the flag's own.

    A captive screw becomes 「其实拧下来了」 (``to=UNSCREWED``), any other
    「其实留在零件里」 (``to=STAYS_IN``); ``to`` names the direction the caller
    means and is refused when the flag says otherwise. Writes nothing.
    """
    from tda.core.graph_infer import blank, resolve_ref   # late: the graph modules are heavy

    desktop, step = int(desktop), int(step)
    instances = db.instances(desktop)
    rec = instances.get(target)
    if rec is None:
        raise CaptiveFixError(f"D{desktop:02d} 没有零件 {target}")
    if is_provisional(target):
        raise CaptiveFixError(f"{target} 是 Label Studio 草稿，不是零件，没有可改的")
    if rec.cls != SCREW:
        raise CaptiveFixError(f"{target}（{rec.cls}）不是螺丝：拧下后留不留在零件里只问螺丝")
    actions = db.actions(desktop)
    at = unscrew_step(actions, target)
    if at is None:
        raise CaptiveFixError(f"日志里没有拧下 {target} 的动作（unscrew）：它留不留在零件里"
                              f"都一样，没有可改的")
    before = record_of(rec)
    new = not bool(before["captive"])
    direction = STAYS_IN if new else UNSCREWED
    if to is not None and str(to) != direction:
        now = "留在零件里" if before["captive"] else "拧下就拿走"
        raise CaptiveFixError(f"{target} 已经是「{now}」了，没有可改的")
    after = dict(before, captive=new, captive_source=MANUAL)
    if new:
        parent = before["parent"] or resolve_ref(instances, rec.fastens)
        if not parent and not blank(rec.fastens):
            raise CaptiveFixError(
                f"{target} 的 fastens 写的是「{rec.fastens}」，这台机器上找不到这个零件：留在零件里"
                f"得知道是哪个零件，先在 Steps 模式的实例表里改 fastens 或填 parent")
        if not parent:
            raise CaptiveFixError(
                f"{target} 没写拧在哪个零件上（fastens / parent 都空着）：留在零件里得知道是"
                f"哪个零件，先在 Steps 模式的实例表里填上")
        if parent == target or parent not in instances:
            raise CaptiveFixError(f"{target} 的 parent（{parent}）不是这台机器的零件：先在 "
                                  f"Steps 模式的实例表里改")
        after.update(parent=parent, attached=True)
    else:
        parent = before["parent"] if before["attached"] else None
        after.update(attached=False)

    events = events_of(db, tax, desktop)
    manual = [e for e in events if not e.auto]
    flipped = {**instances, target: _flipped(rec, after)}
    merged = merge_events(events_from_actions(flipped, actions, tax,
                                              initial=initial_overrides(manual)), manual)
    found = event_problems(flipped, merged, tax)
    fresh = (Counter(_signature(p) for p in found)
             - Counter(_signature(p) for p in event_problems(instances, events, tax)))
    if fresh:
        first = next(p for p in found if _signature(p) in fresh)
        raise CaptiveFixError(_contradiction(first, name))
    return CaptiveFix(desktop=desktop, target=str(target), step=int(at), new=new,
                      at=step, parent=parent, before=before, after=after)
