"""Setting a part's state at the frame in hand, as an undoable op (task U8).

:mod:`tda.core.state_assert` decides what "at step k this part is in state s"
writes; this module is the session's half of it: which parts the gesture is
offered for on the open frame (:meth:`StateMixin.state_choices`), the op-log
row with its inverse, the undo/redo handler, and :func:`settle` -- the same
rule every other write goes through, so the frame in hand is recompiled and a
frozen frame the change reaches is queued for a re-check.

**A cut keyframe follows its boundary** (task U11): the version a commit cut at
the run's lower edge (U8 round 3, ``source="cut"``) moves when an assertion
moves that edge -- :func:`tda.core.cut_follow.plan_cut_moves` says where, in
every view, and the moves are written in the assertion's own op, so one
``Ctrl+Z`` takes back both.

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

from tda.core.cut_follow import CutMove, cut_views, mask_states, plan_cut_moves
from tda.core.db import Db
from tda.core.model import FrameKey, Placement, is_context, is_provisional
from tda.core.state_assert import (
    OP_ASSERT_STATE,
    StateAssertError,
    assert_range,
    assert_state,
    assertable_states,
    write_events,
)
from tda.core.split_rows import cut_copy_at, open_split_steps, reverse_walk
from tda.core.states import (
    FrameState,
    events_from_actions,
    initial_overrides,
    is_cage_latch_event,
    needs_geom,
    state_at,
)
from tda.core.truth import TruthService
from tda.core.truth_inputs import InputCache, annotatable_steps, instances_of, state_of
from tda.ui.class_names import state_zh
from tda.ui.commands import Op
from tda.ui.context_text import NO_STATE as CONTEXT_NO_STATE
from tda.ui.gallery import NO_STATE
from tda.ui.session_api import SessionRefusal
from tda.ui.session_ops import FORWARD as FORWARD_WALK
from tda.ui.session_ops import _apply_keyframe, frozen_among, keyframe_state, settle

__all__ = ["OP_ASSERT_STATE", "StateMixin", "apply_assert_state", "commit_assertion",
           "gate", "move_cuts", "recheck_other_views", "run_around"]

IN_CHASSIS = Placement.IN_CHASSIS.value


def derived_log(db: Db, tax, desktop: int) -> list:
    """The state log the step table alone gives (no hand-written correction on top)."""
    instances = db.instances(desktop)
    manual = [e for e in db.events(desktop) if not e.auto]
    return events_from_actions(instances, db.actions(desktop), tax,
                               initial=initial_overrides(manual))


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
    if is_context(rec):
        return [], CONTEXT_NO_STATE.format(instance=instance)   # task U18
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

    A ✂ row below the run that is open only because its frame shows a cut copy
    (U11 round 2) is the other state's work, not this run's: the copy is what
    keeps the run's version off that frame, so it does not count here.
    """
    walk = reverse_walk(db, key.desktop, key.view)
    under = [s for s in walk if s < first]
    low = under[-1] if under else first
    opened = set(open_split_steps(db, tax, key.desktop, key.view, instance, low, last + 1,
                                  walk=walk))
    below = (bool(under) and low in opened
             and not cut_copy_at(db, tax, FrameKey(key.desktop, low, key.view), instance))
    return {"top": last in opened, "below": below}


def move_cuts(db: Db, desktop: int, instance: str,
              moves: list[CutMove]) -> tuple[list[dict], list[dict], dict[str, list[int]]]:
    """Write :func:`~tda.core.cut_follow.plan_cut_moves`'s answer.

    Returns the keyframe states before and after (the rows of the op, shared
    ``ref`` holders as in a ``commit_keyframe`` op, so a redo re-inserts a
    dropped cut under its own id) and each view's frames to re-check. A cut
    that is ``kept`` writes nothing. **Joins the caller's transaction.**
    """
    before: list[dict] = []
    after: list[dict] = []
    reach: dict[str, set[int]] = {}
    with db.transaction():
        for move in moves:
            if move.kept or not (move.moved or move.dropped):
                continue
            kf = next((k for k in db.keyframes(desktop, move.view, instance)
                       if k.id == move.keyframe_id), None)
            if kf is None:
                continue
            ref = {"keyframe_id": kf.id}
            before.append(keyframe_state(kf, ref))
            if move.dropped:
                db.delete_keyframe(kf.id)
                after.append(keyframe_state(None, ref, template=kf))
            else:
                kf.anchor_step = int(move.new)
                db.update_keyframe(kf)
                after.append(keyframe_state(kf, ref))
            reach.setdefault(move.view, set()).update(move.reach)
    return before, after, {view: sorted(steps) for view, steps in reach.items()}


def _flat(payload: dict) -> dict:
    """The op-log copy of a payload: the shared ``ref`` holders flattened away."""
    return payload | {"keyframes": [
        {k: v for k, v in state.items() if k != "ref"}
        | {"keyframe_id": state["ref"].get("keyframe_id")}
        for state in payload.get("keyframes", ())]}


def recheck_other_views(db: Db, truth: TruthService, desktop: int, view: str,
                        cut_reach: dict) -> None:
    """Queue the frozen frames a moved cut reaches in the views not on screen.

    Their compiled rows are a cache every reader rebuilds; a frozen one is the
    only thing that has to be compared, and the persisted queue is drained when
    that view is opened (or by the sweeper).
    """
    for other, steps in (cut_reach or {}).items():
        if other == view or not steps:
            continue
        wanted = annotatable_steps(db, desktop, other, steps)
        truth.queue_rechecks(desktop, other, frozen_among(db, desktop, other, wanted))


def apply_assert_state(db: Db, truth: TruthService, payload: dict,
                       current: Optional[int] = None) -> dict:
    """Both directions of an ``assert_state`` op: write its rows, then settle.

    The rows are the target's manual state events and, since U11, the cut
    keyframes that followed the boundary (``keyframes``, in any view).
    """
    with db.transaction():
        write_events(db, payload["desktop"], payload["target"], payload["steps"],
                     payload["events"])
        for state in payload.get("keyframes", ()):
            _apply_keyframe(db, state)
    cut_reach = payload.get("cut_reach") or {}
    steps = set(payload["reach"]) | set(cut_reach.get(payload["view"], ()))
    stats = settle(db, truth, payload["desktop"], payload["view"], steps, current)
    recheck_other_views(db, truth, payload["desktop"], payload["view"], cut_reach)
    return stats


def commit_assertion(db: Db, truth: TruthService, key: FrameKey, instance: str,
                     state: str, annotator: str, last: Optional[int] = None) -> dict:
    """Write "at this frame ``instance`` is ``state``" and log it with its inverse.

    ``last`` makes it a span (task U11, ``Ctrl+T``): "from this frame to step
    ``last``", either side of it.

    Returns what every edit returns (``affected``, ``rechecks``, ``problems``,
    ``frame``, ``op``) plus ``changed``, the :class:`StatePatch` as ``patch``,
    :func:`run_around`'s answer as ``run`` / ``after`` and ``cuts``, the
    :class:`~tda.core.cut_follow.CutMove` of every cut keyframe that followed
    the boundary (or stayed, ``kept``). Asserting the state that already holds
    writes nothing and has no op. A refusal of the core is a
    :class:`SessionRefusal` here, so the window shows its sentence instead of a
    traceback.
    """
    try:
        with db.transaction():
            # a part with no cut keyframe anywhere has nothing to follow: the
            # states are read (twice, every step) only when there is one
            follow = bool(cut_views(db, key.desktop, instance))
            before = mask_states(db, truth.tax, key.desktop, instance) if follow else {}
            if last is None:
                patch = assert_state(db, truth.tax, key.desktop, instance, key.step, state,
                                     evidence_view=key.view, name=state_zh)
            else:
                patch = assert_range(db, truth.tax, key.desktop, instance, key.step,
                                     int(last), state, evidence_view=key.view,
                                     name=state_zh)
            if not patch.changed:
                return {"changed": False, "patch": patch, "affected": [], "problems": {},
                        "rechecks": [], "conflicts": 0, "frame": None, "cuts": []}
            moves = [] if not follow else plan_cut_moves(
                db, truth.tax, key.desktop, instance, before,
                mask_states(db, truth.tax, key.desktop, instance))
            kf_before, kf_after, cut_reach = move_cuts(db, key.desktop, instance, moves)
            common = {"view": key.view, "cut_reach": cut_reach}
            payload = patch.payload() | common | {"keyframes": kf_after}
            inverse = patch.inverse() | common | {"keyframes": list(reversed(kf_before))}
            db.log_op(key.desktop, key.view, OP_ASSERT_STATE, _flat(payload),
                      _flat(inverse), annotator)
    except StateAssertError as refused:
        raise SessionRefusal(str(refused)) from refused
    affected = annotatable_steps(db, key.desktop, key.view,
                                 set(patch.reach) | set(cut_reach.get(key.view, ())))
    stats = settle(db, truth, key.desktop, key.view, affected, key.step)
    recheck_other_views(db, truth, key.desktop, key.view, cut_reach)
    first, last_step, after = run_around(db, truth.tax, key.desktop, instance, key.step)
    return {"changed": True, "patch": patch, "affected": affected,
            "run": (first, last_step), "after": after,
            "redraw": redraw_needed(db, truth.tax, key, instance, first, last_step),
            "cuts": moves,
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
        if getattr(self, "gallery", False):
            return [], NO_STATE   # a components-gallery shot has no state (U10)
        key = self.current()
        cache = InputCache()
        return gate(instances_of(self.db, key.desktop, cache),
                    state_of(self.db, self.tax, key.desktop, key.step, cache),
                    self.tax, str(instance), key.step)

    def _with_state_choices(self, rows: list[dict], state: FrameState,
                            instances: dict) -> list[dict]:
        """Add ``state_choices`` to instance-list rows (``[]`` where not offered)."""
        step = self.current().step
        if getattr(self, "gallery", False):
            for row in rows:
                row["state_choices"] = []   # no state on a gallery shot (U10)
            return rows
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
        key = self._state_gate_here(instance, state)
        return self._commit_state(key, instance, state, None)

    def assert_state_range(self, instance: str, state: str, to_step: int) -> dict:
        """"From this frame to step ``to_step`` ``instance`` is ``state``" (U11, ``Ctrl+T``).

        The span runs either way from the frame on screen (the annotator walks
        backwards, so usually down to an earlier step). One op: the events at
        its two ends, the part's hand-written state events inside it dropped,
        and the cut keyframes that follow its lower edge -- one ``Ctrl+Z``.
        Refused where ``T`` would be, and where the span crosses a step at
        which the log itself changes the part's state (that is a Steps-mode
        correction).
        """
        key = self._state_gate_here(instance, state)
        return self._commit_state(key, instance, state, int(to_step))

    def state_range_default(self, instance: str) -> tuple[Optional[str], int]:
        """``(state, step)`` the ``Ctrl+T`` dialog opens with.

        The state: the frame's own when it was set by hand (the run being
        walked is to be extended), else the one ``T`` would set. The step: as
        far as the span can go from here in the browsing direction without
        crossing a step where the log changes the part's state -- walking
        backwards, the step after the last such event at or below this frame
        (the table's first step when there is none); forwards, the step before
        the next one (the last step).

        A drive cage's implied latch (task U20) on a frame at or before the
        step it opens at by default proposes what its card row says --
        「默认这里是打开的；更早就开着的话到那一帧 Ctrl+T」 -- ``open``, to this very
        frame: the annotator says how far. That default is no action of the
        log, so it never bounds a span either.
        """
        key = self.current()
        rows = sorted(self.db.steps(key.desktop), key=lambda r: int(r.step))
        table = [int(r.step) for r in rows]
        if not table:
            return None, key.step
        dupli = {int(r.step) for r in rows if r.dupli or r.step_type == "dupli"}
        log = derived_log(self.db, self.tax, key.desktop)
        default = [e for e in log if e.target == instance and is_cage_latch_event(e)]
        mine = [e.step for e in log
                if e.target == instance and e.step >= 1 and not is_cage_latch_event(e)]
        if self.browsing == FORWARD_WALK:
            above = [s for s in mine if s > key.step]
            step = (min(above) - 1) if above else table[-1]
        else:
            below = [s for s in mine if s <= key.step]
            step = min((max(below) + 1) if below else table[0], key.step)
            while step in dupli and step < key.step:
                step += 1                  # a duplicate frame follows the one before it
        held = state_of(self.db, self.tax, key.desktop, key.step).get(instance)
        logged = state_at(self.db.instances(key.desktop), log, key.step,
                          self.tax).get(instance)
        if held is not None and logged is not None and held.state != logged.state:
            return held.state, int(step)
        if default and key.step <= default[0].step:
            # the cage latch's card row (U20 review, N1): open, from here
            return str(default[0].new), int(key.step)
        return self.next_state(instance), int(step)

    def _state_gate_here(self, instance: str, state: str) -> FrameKey:
        """The frame both state gestures stand on, or the refusal of ``T`` there."""
        key = self.current()
        if getattr(self, "gallery", False):
            raise SessionRefusal(NO_STATE)   # task U10: refused before anything else
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
        return key

    def _commit_state(self, key: FrameKey, instance: str, state: str,
                      last: Optional[int]) -> dict:
        result = commit_assertion(self.db, self.truth, key, str(instance),
                                  str(state), self.annotator, last=last)
        if not result["changed"]:
            return result
        nxt = result["patch"].next_step
        result["restore_unseen"] = nxt is not None and not self._has_picture(nxt)
        return self._after_edit(result)

    def _has_picture(self, step: int) -> bool:
        """Does this view have a usable image of ``step``? (``_editable_frame``'s test)"""
        return int(step) in self._available and self.image_path(int(step)) is not None
