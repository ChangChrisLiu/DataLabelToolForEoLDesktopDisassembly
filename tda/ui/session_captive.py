"""「其实拧下来了」 / 「其实留在零件里」 as an undoable op (task U14).

:mod:`tda.core.captive_fix` decides what flips; this module is the session's
half: the gate the instance list and the card show
(:meth:`CaptiveMixin.captive_offers`), the write -- through the step table's
own path, ``StepTableData.load`` → ``apply_instance_edit(key, "captive", ...)``
→ ``save``, what the ``Captive`` cell of Steps mode does, so the S1 issues are
asked and the automatic event log is recompiled the same way; the graph meta is
re-stamped as a merge does it (:func:`tda.core.extra.restamp_graph`) -- the
op-log row with its inverse, and the undo/redo handler.

**Every view.** A screw's truth changes from its ``unscrew`` on in all four
views and in the components gallery (a captive screw goes out *inside* its
part; a plain one is a part of the pile), so every frozen frame of every view
in that reach is queued for a re-check, and the frame in hand is compiled
(:func:`~tda.ui.session_ops.settle`). A confirmed frame whose signed rows no
longer match -- the screw drawn where it is now out of the machine -- opens a
conflict when it is re-checked; one that now lacks it goes back to review.
The commit counts both beforehand, for the status line.

Both directions of the op are the same record with the other values, so one
handler (:func:`apply_captive_fix`) is the redo and the undo.
"""
from __future__ import annotations

from typing import Optional

from tda.core.captive_fix import (
    OP_CAPTIVE_FIX,
    SCREW,
    UNSCREWED,
    CaptiveFixError,
    captive_offers,
    plan_captive_fix,
)
from tda.core.db import Db
from tda.core.model import ALL_VIEWS, FrameKey, is_component_view, is_context_key
from tda.core.states import needs_geom
from tda.core.truth import TruthService
from tda.core.truth_inputs import InputCache, annotatable_steps, instances_of, state_of
from tda.ui.class_names import state_zh
from tda.ui.commands import Op
from tda.ui.gallery import NO_VERB
from tda.ui.session_api import KIND_CONFIRM, VERB_NEEDS_CLEAN_LAYER, SessionRefusal
from tda.ui.session_ops import settle
from tda.ui.session_tasks import bench_roi_of
from tda.ui.steps_model import ANNOTATOR, EditError, StepTableData

__all__ = ["OP_CAPTIVE_FIX", "CaptiveMixin", "apply_captive_fix", "commit_captive_fix",
           "reach_of", "write_captive"]

VERIFIED = "verified"
LOOSENED = "loosened"
#: A flip that would leave the step table with a question it did not have.
NEW_ISSUE = "这样改 Steps 表会多一个问题（{issue}）：请在 Steps 模式里改"


def write_captive(db: Db, tax, rows: dict, annotator: Optional[str] = None,
                  check: bool = False) -> list[str]:
    """Give one screw the record ``rows`` names through the S1 path; joins a transaction.

    ``rows`` carries ``captive``, ``captive_source``, ``parent`` and
    ``attached`` (``None`` for an attribute that is absent). ``check`` refuses
    a write that gives the instance table an S1 issue it did not have -- the
    forward flip; an undo puts back a record that was there. Returns what
    :meth:`StepTableData.save` returns. The constraint graph is not re-derived
    (``save(graph=False)``: no edge is derived from ``captive``); its meta is
    re-stamped, because the deadlock count reads it.
    """
    from tda.core.extra import restamp_graph     # late: tda.pipeline is heavy

    who = str(annotator or rows.get("annotator") or ANNOTATOR)
    desktop, target = int(rows["desktop"]), str(rows["target"])
    data = StepTableData.load(db, desktop, tax, who)
    if target not in data.instances:
        raise SessionRefusal(f"{target} 已经不在实例表里了（在别处改过）")
    before = set(data.issues) if check else set()
    if rows.get("captive") is not None:
        data.apply_instance_edit(target, "captive", bool(rows["captive"]))
    rec = data.instances[target]
    for name in ("captive", "captive_source"):
        if rows.get(name) is None:
            rec.attrs.pop(name, None)
        else:
            rec.attrs[name] = rows[name]
    if (rec.parent or None) != (rows.get("parent") or None):
        data.apply_instance_edit(target, "parent", rows.get("parent") or "")
    if bool(rec.attached) != bool(rows.get("attached")):
        data.apply_instance_edit(target, "attached", bool(rows.get("attached")))
    data.refresh_issues()
    if check:
        fresh = [issue for issue in data.issues if issue not in before]
        if fresh:
            raise SessionRefusal(NEW_ISSUE.format(issue=fresh[0]))
    messages = data.save(db, graph=False)
    restamp_graph(db, desktop, tax)
    return messages


def reach_of(db: Db, desktop: int, step: int) -> dict[str, list[int]]:
    """``view -> the frames whose truth a flip from ``step`` on can change``.

    Every view with frames: the annotatable steps from ``step`` on, and every
    shot of a components-gallery view (it shows the end of the teardown).
    """
    table = sorted({int(r.step) for r in db.steps(desktop)})
    out: dict[str, list[int]] = {}
    for view in ALL_VIEWS:
        frames = [int(r["step"]) for r in db.frames_for(desktop, view)]
        if not frames:
            continue
        wanted = frames if is_component_view(view) else [s for s in table if s >= int(step)]
        steps = annotatable_steps(db, desktop, view, wanted)
        if steps:
            out[view] = steps
    return out


def _signed(db: Db, desktop: int, target: str) -> dict[tuple[str, int], str]:
    """``(view, step) -> placement`` of every row of ``target`` a human has signed."""
    return {(str(r["view"]), int(r["step"])): str(r["placement"] or "")
            for r in db.conn.execute(
                "SELECT view, step, placement FROM compiled_mask WHERE desktop=? "
                "AND instance=? AND status=?", (int(desktop), str(target), VERIFIED))}


def _conflicted(db: Db, desktop: int, target: str) -> set[tuple[str, int]]:
    """``(view, step)`` of every open conflict about ``target``."""
    return {(str(r["view"]), int(r["step"])) for r in db.conn.execute(
        "SELECT view, step FROM conflict WHERE desktop=? AND instance=? AND status='open'",
        (int(desktop), str(target)))}


def _agrees(held: tuple, placement: str) -> bool:
    """Does the frame ask for the screw where its signed row has it?"""
    inst, needed = held
    return bool(needed) and inst is not None and inst[1] == placement


def _held(db: Db, tax, desktop: int, target: str,
          frames: dict[str, list[int]]) -> dict[tuple[str, int], tuple]:
    """``(view, step) -> (state, placement, left_with, needed)`` of ``target``."""
    cache = InputCache()
    instances = instances_of(db, desktop, cache)
    out: dict[tuple[str, int], tuple] = {}
    for view, steps in frames.items():
        for step in steps:
            key = FrameKey(int(desktop), int(step), view)
            state = state_of(db, tax, desktop, int(step), cache, view=view)
            inst = state.get(target)
            needed = target in needs_geom(instances, state, tax,
                                          bench_roi=bench_roi_of(db, key, cache))
            out[(view, int(step))] = (None if inst is None else
                                      (inst.state, inst.placement, inst.left_with), needed)
    return out


def _queue_other_views(db: Db, truth: TruthService, desktop: int, view: str,
                       reach: dict) -> dict[str, list[int]]:
    """Queue the frozen frames of every view but ``view`` in ``reach``."""
    queued: dict[str, list[int]] = {}
    for other, steps in sorted((reach or {}).items()):
        if other == view:
            continue
        frozen = db.frozen_steps_in(int(desktop), other, steps)
        if frozen:
            queued[other] = truth.queue_rechecks(int(desktop), other, frozen)
    return queued


def apply_captive_fix(db: Db, truth: TruthService, payload: dict,
                      current: Optional[int] = None) -> dict:
    """Both directions of a ``captive_fix`` op: write the record, then settle."""
    desktop, view = int(payload["desktop"]), str(payload["view"])
    reach = payload.get("reach") or {}
    with db.transaction():
        write_captive(db, truth.tax, payload)
        _queue_other_views(db, truth, desktop, view, reach)
    return settle(db, truth, desktop, view, reach.get(view) or [], current)


def commit_captive_fix(db: Db, truth: TruthService, key: FrameKey, instance: str,
                       to: Optional[str], annotator: str) -> dict:
    """Flip ``instance``'s ``captive``, logged with its inverse.

    Returns what every edit returns (``affected``, ``rechecks``, ``problems``,
    ``frame``, ``op``) plus the :class:`~tda.core.captive_fix.CaptiveFix` as
    ``fix``, ``reach`` (every view's frames it can change), ``frozen`` (the
    confirmed frames among them, re-checked), ``clashes`` -- the confirmed
    frames whose signed row of the screw no longer matches (a conflict when
    re-checked) --, ``closes`` -- the ones whose signed row matches again and
    hold an open conflict about it (a flip taken back by a second ``Shift+T``
    rather than ``Ctrl+Z``: the re-check closes it) -- and ``owed``, the
    confirmed frames that now ask for it and have no row of it (back to
    review). A signed row matches when the frame asks for the screw in the
    placement the row has it.
    """
    try:
        fix = plan_captive_fix(db, truth.tax, key.desktop, instance, key.step, to,
                               name=state_zh)
    except CaptiveFixError as refused:
        raise SessionRefusal(str(refused)) from refused
    reach = reach_of(db, key.desktop, fix.step)
    frozen = {view: found for view, steps in reach.items()
              if (found := db.frozen_steps_in(key.desktop, view, steps))}
    signed = _signed(db, key.desktop, instance)
    conflicted = _conflicted(db, key.desktop, instance)
    before = _held(db, truth.tax, key.desktop, instance, frozen)
    payload = fix.rows(fix.after) | {"view": key.view, "reach": reach,
                                     "annotator": annotator}
    inverse = fix.rows(fix.before) | {"view": key.view, "reach": reach,
                                      "annotator": annotator}
    try:
        with db.transaction():
            messages = write_captive(db, truth.tax, payload, annotator, check=True)
            db.log_op(key.desktop, key.view, OP_CAPTIVE_FIX, payload, inverse, annotator)
            queued = _queue_other_views(db, truth, key.desktop, key.view, reach)
    except EditError as refused:
        raise SessionRefusal(str(refused)) from refused
    after = _held(db, truth.tax, key.desktop, instance, frozen)
    changed = sorted(frame for frame in before if before[frame] != after[frame])
    resigned = [f for f in changed if f in signed]
    stats = settle(db, truth, key.desktop, key.view, reach.get(key.view) or [], key.step)
    return {"changed": True, "fix": fix, "reach": reach, "affected": reach.get(key.view, []),
            "frozen": frozen, "queued": queued,
            "clashes": [f for f in resigned if not _agrees(after[f], signed[f])],
            "closes": [f for f in resigned if _agrees(after[f], signed[f]) and f in conflicted],
            "owed": [f for f in changed if f not in signed and after[f][1]],
            "messages": list(messages), "compiled": stats["compiled"],
            "rechecks": stats["rechecks"], "conflicts": stats["conflicts"],
            "problems": stats["problems"], "frame": stats["frame"],
            "op": Op(OP_CAPTIVE_FIX, payload, inverse)}


class CaptiveMixin:
    """The captive flip of :class:`~tda.ui.session.AnnotationSession`."""

    def captive_offers(self, state=None, instances=None) -> dict[str, str]:
        """``screw -> "unscrewed" | "stays_in"`` for the screws offered the flip here.

        :func:`tda.core.captive_fix.captive_offers`: a screw the log
        unscrews. Nothing on a components-gallery shot (its number is a shot,
        not a step). ``state`` is unused (the offer is the screw's, not the
        frame's) and accepted so callers can pass what they pass to
        :meth:`verb_offers`.
        """
        if not self.is_open or getattr(self, "gallery", False):
            return {}
        key = self.current()
        if instances is None:
            instances = instances_of(self.db, key.desktop)
        return captive_offers(instances, self.db.actions(key.desktop))

    def _mark_loosened_screws(self, card: list[dict], key: FrameKey) -> None:
        """Mark the card rows of loosened captive screws with ``verb_fix`` (U14).

        A captive screw stands ``loosened`` in the chassis from its unscrew
        until its part comes out, and the card asks for it there; when the
        picture does not have it, it was taken out after all (D3 7-15). Its
        row says which key: :func:`tda.ui.panels.taskcard.captive_hint`.
        """
        rows = [row for row in card
                if str(row.get("cls", "")) == SCREW and row.get("kind") != KIND_CONFIRM]
        if not rows or getattr(self, "gallery", False):
            return
        offered = self.captive_offers()
        rows = [row for row in rows if offered.get(str(row.get("instance", ""))) == UNSCREWED]
        if not rows:
            return
        state = state_of(self.db, self.tax, key.desktop, key.step, view=key.view)
        for row in rows:
            held = state.get(str(row.get("instance", "")))
            if held is not None and held.state == LOOSENED:
                row["verb_fix"] = UNSCREWED

    def is_screw(self, instance: str) -> bool:
        """Is ``instance`` a screw of the open desktop? (``Shift+T``'s two meanings.)"""
        if not self.is_open or not instance or is_context_key(instance):
            return False        # a 框外零件 screw is no screw of the machine (U18)
        rec = instances_of(self.db, self.current().desktop).get(str(instance))
        return rec is not None and rec.cls == SCREW

    def captive_gate(self, instance: str) -> str:
        """Why :meth:`correct_captive` would refuse ``instance`` here, ``""`` if not.

        The core's plan as a dry run, writing nothing: what the palette's
        greyed 「其实拆下」 button says for a screw.
        """
        if not self.is_open:
            return "没有打开的帧"
        if getattr(self, "gallery", False):
            return NO_VERB
        key = self.current()
        try:
            plan_captive_fix(self.db, self.tax, key.desktop, str(instance), key.step,
                             name=state_zh)
        except CaptiveFixError as refused:
            return str(refused)
        return ""

    def correct_captive(self, instance: str, to: Optional[str] = None) -> dict:
        """「其实拧下来了」 (``to="unscrewed"``) / 「其实留在零件里」 (``"stays_in"``); undoable.

        Without ``to`` the direction is the screw's flag: captive becomes
        unscrewed-and-out, anything else stays in its part. A screw being
        drawn with pixels not yet committed is refused; one whose edit is
        untouched has it dropped, since the frame may no longer ask for it.
        """
        if not self.is_open:
            raise SessionRefusal("没有打开的帧")
        if getattr(self, "gallery", False):
            raise SessionRefusal(NO_VERB)
        editing = self.layer.instance
        if editing == instance and self.layer.changed():
            raise SessionRefusal(VERB_NEEDS_CLEAN_LAYER.format(instance=instance))
        result = commit_captive_fix(self.db, self.truth, self.current(), str(instance), to,
                                    self.annotator)
        if editing == instance:
            self.clear_edit()
        return self._after_edit(result)
