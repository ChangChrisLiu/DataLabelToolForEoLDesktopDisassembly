"""Opened, or taken out: the verb of the action that made a part leave (task U9).

A cover and the four removable latches (``psu_latch``, ``drive_latch``,
``card_latch``, ``cooler_latch`` -- :meth:`~tda.core.taxonomy.Taxonomy.opened_or_taken_out`)
can be swung open and stay on the machine, or come off and go on the pile.
The sheet rarely says which, and the importer drafts ``open``: on D2 the card
at the last frame asked for six parts that were lying in the components pile.
Only the pictures decide, so the annotator says it in one key and the action
that opened the part becomes a ``remove`` -- or, the other way round, a
``remove`` becomes an ``open`` for a part that never left.

This is a **step-table correction**, not a state assertion (task U8): the
action itself is what was wrong, so the action is rewritten and the derived
log follows it everywhere -- every view, every frame from that step on. This
module only *plans* it (:func:`plan_verb_fix`): which action, and every reason
it cannot be done, in a sentence the annotator can act on. The write goes
through the step table's own path (:mod:`tda.ui.session_verb`), so it is
validated and saved exactly as an edit in Steps mode would be.

Refused (:class:`VerbFixError`): a class that cannot be both opened and taken
out, a part that is neither open nor removed at the frame, an open (or a
removal) that no action of the part's own made -- a hand-written state
(``T``), a part that went out inside its parent --, a part the log still acts
on after that step, and anything that would make
:func:`~tda.core.states.event_problems` find a problem it did not find before
(a hand-written state after the step that cannot follow a removal).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable, Optional

from tda.core.db import Db
from tda.core.model import ActionRec, InstanceRec, is_provisional
from tda.core.states import (
    ATTR_STATE,
    INITIAL_STEP,
    LEAVES_REMOVED,
    REMOVED,
    EventProblem,
    FrameState,
    event_problems,
    events_from_actions,
    initial_overrides,
    state_at,
)
from tda.core.taxonomy import Taxonomy
from tda.core.truth_inputs import events_of, merge_events

__all__ = [
    "OPEN",
    "OP_CORRECT_VERB",
    "REMOVE",
    "VerbFix",
    "VerbFixError",
    "offers",
    "plan_verb_fix",
]

#: ``op_log.kind`` (and :data:`tda.ui.commands.KINDS` entry) of one correction.
OP_CORRECT_VERB = "correct_verb"
OPEN = "open"
REMOVE = "remove"
#: Each verb and the one it becomes.
FLIP = {OPEN: REMOVE, REMOVE: OPEN}
SUCCESS = "success"

Namer = Callable[[str], str]


def _same(value: str) -> str:
    return str(value)


class VerbFixError(ValueError):
    """A correction the log cannot take; the text is ready for the status line."""


@dataclass(frozen=True)
class VerbFix:
    """One action whose verb flips: ``old`` -> ``new`` at ``step`` (its ``idx``).

    ``at`` is the frame the annotator asked from; ``target`` is the part.
    """

    desktop: int
    target: str
    step: int
    idx: int
    old: str
    new: str
    at: int

    def rows(self, verb: str) -> dict:
        """The op-log record of the action with ``verb``: both directions are one."""
        return {"desktop": int(self.desktop), "target": self.target, "step": int(self.step),
                "idx": int(self.idx), "verb": str(verb), "at": int(self.at)}


def _state_of_verb(tax: Taxonomy, rec: InstanceRec, verb: str) -> Optional[str]:
    """The state ``verb`` leaves a part of this class in, or ``None``."""
    effect = tax.apply_verb(rec.cls, rec.attrs, verb)
    return effect[1] if effect and effect[0] == ATTR_STATE else None


def _own_actions(actions: list[ActionRec], step: int) -> dict[str, ActionRec]:
    """``target -> its latest successful open/remove`` at or before ``step``."""
    found: dict[str, ActionRec] = {}
    for action in sorted(actions, key=lambda a: (a.step, a.idx)):
        if action.step > step:
            break
        if action.result == SUCCESS and action.verb in FLIP:
            found[action.target] = action
    return found


def offers(tax: Taxonomy, instances: dict[str, InstanceRec], state: FrameState,
           actions: list[ActionRec], step: int) -> dict[str, str]:
    """``instance -> the verb it would become`` for the parts offered the correction.

    The cheap gate the instance list and the task card ask on every frame: a
    part of a class that may be opened *or* taken out, whose state at ``step``
    is the one its own latest ``open`` / ``remove`` left it in. Everything
    :func:`plan_verb_fix` refuses beyond that is said when the key is pressed.
    """
    two_way = tax.opened_or_taken_out()
    own = _own_actions(actions, step)
    out: dict[str, str] = {}
    for key, held in state.items():
        rec = instances.get(key)
        action = own.get(key)
        if rec is None or action is None or rec.cls not in two_way or is_provisional(key):
            continue
        if _state_of_verb(tax, rec, action.verb) == held.state:
            out[key] = FLIP[action.verb]
    return out


def _signature(problem: EventProblem) -> tuple:
    event = problem.event
    return (event.step, event.target, event.attr, event.old, event.new, event.auto,
            problem.code)


def _contradiction(problem: EventProblem, name: Namer) -> str:
    """A new problem of the merged log, said as what the annotator has to do."""
    event = problem.event
    if not event.auto and problem.code == LEAVES_REMOVED:
        return (f"{event.target} 第 {event.step} 步有手动设的状态（T：{name(event.old)} → "
                f"{name(event.new)}），拆下以后就接不上了：先到第 {event.step} 步按 T 把它改回去，"
                f"或在 Steps 模式里改拆下的那一步")
    return (f"这样改会和第 {event.step} 步的状态记录对不上（{problem.text}）："
            f"请在 Steps 模式里改")


def _not_two_way(target: str, rec: InstanceRec, tax: Taxonomy) -> str:
    opens = (tax.verbs.get(OPEN) or {}).get("applies_to") or ()
    if rec.cls in opens:
        return f"{target}（{rec.cls}）只能打开、不会单独拆下（它长在主板上，跟着主板走）"
    return f"{target}（{rec.cls}）不是能打开也能拆下的盖板或卡扣，这里没有可改的"


def plan_verb_fix(db: Db, tax: Taxonomy, desktop: int, target: str, step: int,
                  to: Optional[str] = None, name: Namer = _same) -> VerbFix:
    """Which action to flip so that ``target`` is open / taken out from its step on.

    The direction is read off the state at ``step``: ``open`` there flips the
    ``open`` that made it so into ``remove``; ``removed`` there flips the part's
    own ``remove`` into ``open``. ``to`` names the direction the caller means
    (``"remove"`` / ``"open"``) and is refused when the state says otherwise.
    Writes nothing.
    """
    desktop, step = int(desktop), int(step)
    instances = db.instances(desktop)
    rec = instances.get(target)
    if rec is None:
        raise VerbFixError(f"D{desktop:02d} 没有零件 {target}")
    if is_provisional(target):
        raise VerbFixError(f"{target} 是 Label Studio 草稿，不是零件，没有动作可改")
    if rec.cls not in tax.opened_or_taken_out():
        raise VerbFixError(_not_two_way(target, rec, tax))
    actions = db.actions(desktop)
    events = events_of(db, tax, desktop)
    held = state_at(instances, events, step, tax)[target].state
    old = next((verb for verb in FLIP if _state_of_verb(tax, rec, verb) == held), None)
    if old is None:
        raise VerbFixError(f"{target} 在第 {step} 步是「{name(held)}」：只有「{name(OPEN)}」"
                           f"或「{name(REMOVED)}」的零件能改")
    new = FLIP[old]
    if to is not None and str(to) != new:
        raise VerbFixError(f"{target} 在第 {step} 步已经是「{name(held)}」了，没有可改的")

    mine = [e for e in events if e.target == target and e.attr == ATTR_STATE
            and e.step <= step and e.new == held]
    derived = [e for e in mine if e.auto]
    if not derived:
        if mine and mine[-1].step <= INITIAL_STEP:
            # a part added in S1: its state is where it started, and T cannot
            # say "removed" -- only an action can
            raise VerbFixError(
                f"{target} 是 S1 添加的零件，「{name(held)}」是添加时设的初始状态，日志里没有它的"
                f"动作：它被拆下了，就到 Steps 模式给它加一步 remove 动作")
        if mine:
            raise VerbFixError(
                f"{target} 的「{name(held)}」是第 {mine[-1].step} 步手动设的状态（T），不是日志里"
                f"的动作：没有动作可改；其实关着就在那一帧按 T 改，拆下了就到 Steps 模式"
                f"给它加一步 remove 动作")
        raise VerbFixError(f"{target} 的「{name(held)}」不是日志里哪个动作做的：没有动作可改")
    at = derived[-1].step
    action = next((a for a in actions if a.step == at and a.target == target
                   and a.verb == old and a.result == SUCCESS), None)
    if action is None:
        parent = rec.parent if rec.attached and rec.parent else None
        if parent and old == REMOVE:
            raise VerbFixError(f"{target} 是第 {at} 步跟着 {parent} 一起拆下的，不是它自己的"
                               f"动作：要改请在 Steps 模式里改")
        raise VerbFixError(f"第 {at} 步没有 {target} 自己的 {old} 动作：请在 Steps 模式里改")
    later = [a for a in actions if a.target == target and a.step > at and a.result == SUCCESS]
    if new == REMOVE and later:
        raise VerbFixError(
            f"日志第 {later[0].step} 步还有它的动作（{later[0].verb}）：到那一步它还在机箱里；"
            f"它是哪一步拆下的，请在 Steps 模式里改")

    manual = [e for e in events if not e.auto]
    flipped = [replace(a, verb=new) if a is action else a for a in actions]
    after = merge_events(events_from_actions(instances, flipped, tax,
                                             initial=initial_overrides(manual)), manual)
    found = event_problems(instances, after, tax)
    fresh = (Counter(_signature(p) for p in found)
             - Counter(_signature(p) for p in event_problems(instances, events, tax)))
    if fresh:
        first = next(p for p in found if _signature(p) in fresh)
        raise VerbFixError(_contradiction(first, name))
    return VerbFix(desktop=desktop, target=str(target), step=int(action.step),
                   idx=int(action.idx), old=old, new=new, at=step)
