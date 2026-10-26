"""「这台没有这个零件」: a logged part the machine does not have (task U34).

D34 has no optical drive, yet its sheet logs one -- step 37 "Try to remove the
optical drive" (``failed``), two "optical drive screw" rows -- so the importer
made ``optical_drive.01``, the state machine keeps it ``installed`` from frame 1
and every frame's task card asks for its mask. Nothing let the annotator say
otherwise:

* ``T`` sets the state *the picture shows* among the states a part in the
  picture can be in; ``removed`` is not one of them by design
  (:func:`tda.core.state_assert.assertable_states`: "when a part leaves is the
  step table's to say"), and :func:`tda.ui.session_state.gate` only offers it
  for a part in the picture anyway;
* deleting the instance in Steps is refused while an action targets it, and
  the action is the sheet's own record of what the operator tried.

So this is its own op, not a state assertion: **the part is ``removed`` from
before step 1** -- a hand-written ``state`` event ``<initial> -> removed`` at
:data:`~tda.core.states.INITIAL_STEP`, the shape an added part's initial
state takes (:func:`tda.core.extra.plan_extras`), plus ``placement
in_chassis -> elsewhere`` beside it, both marked :data:`EVIDENCE`. The state
machine reads it as :func:`~tda.core.states.absent_keys`: the part starts
``removed`` / ``elsewhere`` (not ``on_bench``: a view with a bench ROI would ask
for its box there), needs no geometry anywhere, never joins the components
gallery, takes no attached child out with it, and the sheet's own actions on
it derive nothing -- so the compiler, the task card, ``cli check``, the exports
and the state timeline see a part that was never there. Its screws and
connectors are left as the sheet says: they are the annotator's to decide.

The op touches only the target's hand-written events at or before step 0:
:class:`AbsentPatch` holds them before and after, so one writer
(:func:`write_rows`) is the do, the undo and the redo, and 「其实有」
(:func:`plan_present`) is the same write the other way round. Every write
recompiles the stored automatic events, re-stamps the graph meta and queues
the frozen frames whose work it changes, in its transaction.

Refused (:class:`AbsentError`, a sentence the status line can show): a part
that has a shape in any view (a drawn part *is* in the machine), a Label
Studio draft, a 框外零件, a part added in S1 (delete it instead), the chassis,
a class without a ``removed`` state, a part already said absent.

``import-logs --force`` keeps it: hand-written events survive the import and
its derived log starts from :func:`~tda.core.states.initial_overrides`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

from tda.core.db import Db
from tda.core.log_identity import CHASSIS_KEY
from tda.core.model import (
    ALL_VIEWS,
    GALLERY_EXCLUDED_CLASSES,
    Placement,
    StateEvent,
    is_component_view,
    is_context,
    is_extra,
    is_provisional,
)
from tda.core.states import (
    ABSENT_PLACEMENT,
    ATTR_PLACEMENT,
    ATTR_STATE,
    INITIAL_STEP,
    REMOVED,
    absent_keys,
    events_from_actions,
    gallery_state,
    initial_overrides,
    needs_geom,
    state_at,
)
from tda.core.taxonomy import Taxonomy
from tda.core.truth_inputs import merge_events

__all__ = [
    "CAPTION",
    "CAPTION_BACK",
    "CHOICE",
    "EVIDENCE",
    "OP_KIND",
    "OP_VIEW",
    "SHORT",
    "AbsentError",
    "AbsentPatch",
    "absent_events",
    "is_absent",
    "latest_mark",
    "mark_absent",
    "mark_present",
    "plan_absent",
    "plan_present",
    "refusal",
    "shaped_instances",
    "write_rows",
]

#: ``op_log.kind`` of both directions (and the session's undo kind).
OP_KIND = "absent_from_start"
#: ``op_log.view`` when the op was not made on a frame (Steps, the script).
OP_VIEW = "-"
#: ``StateEvent.evidence_view`` of the two events the op writes.
EVIDENCE = "s1:absent"
#: The pseudo-state the ``T`` picker offers on a part's first frame (task U34).
CHOICE = "absent_from_start"
#: How the picker, the Steps menu and the rows say it.
CAPTION = "一开始就不在（这台没有它）"
#: The Steps row's short word, and the inverse's menu entry.
SHORT = "一开始就不在"
CAPTION_BACK = "其实有"
IN_CHASSIS = Placement.IN_CHASSIS.value


class AbsentError(ValueError):
    """The op cannot be done; the text is ready for the status line."""


# --------------------------------------------------------------------------- #
# reads
# --------------------------------------------------------------------------- #
def is_absent(db: Db, desktop: int, key: str) -> bool:
    """Is ``key`` said to be a part this machine does not have?"""
    return str(key) in absent_keys(e for e in db.events(int(desktop)) if not e.auto)


def shaped_instances(db: Db, desktop: int) -> set[str]:
    """Every instance of ``desktop`` with a shape in any view: two queries.

    A shape is a keyframe (a Label Studio draft adopted, a cut copy, anything
    drawn) or a single frame's pixels (a frame override with a mask).
    """
    desktop = int(desktop)
    out = {str(row[0]) for row in db.conn.execute(
        "SELECT DISTINCT instance FROM shape_keyframe WHERE desktop=?", (desktop,))}
    out |= {str(row[0]) for row in db.conn.execute(
        "SELECT DISTINCT instance FROM frame_override WHERE desktop=? "
        "AND visible_rle_json IS NOT NULL", (desktop,))}
    return out


def _shapes(db: Db, desktop: int, key: str) -> dict[str, int]:
    """``view -> how many shapes`` of ``key`` (keyframes, single-frame pixels)."""
    out: dict[str, int] = {}
    for view in ALL_VIEWS:
        n = len(db.keyframes(desktop, view, key))
        if n:
            out[view] = n
    for view, n in db.conn.execute(
            "SELECT view, COUNT(*) FROM frame_override WHERE desktop=? AND instance=? "
            "AND visible_rle_json IS NOT NULL GROUP BY view", (int(desktop), str(key))):
        out[str(view)] = out.get(str(view), 0) + int(n)
    return out


def refusal(db: Db, tax: Taxonomy, desktop: int, key: str) -> str:
    """Why ``key`` cannot be said to be absent from the start; ``""`` if it can."""
    desktop, key = int(desktop), str(key)
    rec = db.instances(desktop).get(key)
    if rec is None:
        return f"D{desktop:02d} 没有零件 {key}"
    if is_provisional(key):
        return f"{key} 是 Label Studio 草稿，不是零件"
    if is_context(rec):
        return f"{key} 是框外零件，不在机器里，没有「在不在」可说"
    if key == CHASSIS_KEY or rec.cls == CHASSIS_KEY:
        return "机箱就是这台机器，不能说它不在"
    if is_extra(rec):
        return (f"{key} 是手动添加的零件：这台没有它就直接删掉（Steps → Instances 右键 "
                f"Delete）")
    if REMOVED not in tax.states_of(rec.cls):
        return f"{key}（{rec.cls}）没有「拆下」这个状态，不能说它一开始就不在"
    if is_absent(db, desktop, key):
        return f"{key} 已经是「{SHORT}」了"
    shapes = _shapes(db, desktop, key)
    if shapes:
        where = "、".join(f"{view} {n} 个" for view, n in sorted(shapes.items()))
        return (f"{key} 已经画过形状（{where}）：画过的零件就是在这台机器里。真没有它就先把"
                f"形状删掉（Ctrl+Z 撤回那次提交，或在那一帧擦掉后提交）")
    return ""


def absent_events(desktop: int, key: str, held: str) -> list[StateEvent]:
    """The two events of 「这台没有这个零件」: ``held -> removed``, ``in_chassis -> elsewhere``."""
    return [
        StateEvent(desktop=int(desktop), step=INITIAL_STEP, target=str(key),
                   attr=ATTR_STATE, old=str(held), new=REMOVED, evidence_view=EVIDENCE,
                   auto=False),
        StateEvent(desktop=int(desktop), step=INITIAL_STEP, target=str(key),
                   attr=ATTR_PLACEMENT, old=IN_CHASSIS, new=ABSENT_PLACEMENT,
                   evidence_view=EVIDENCE, auto=False),
    ]


def _initial_rows(db: Db, desktop: int, key: str) -> list[StateEvent]:
    """The target's hand-written events at or before step 0, stored order."""
    return [e for e in db.events(int(desktop))
            if e.target == key and not e.auto and e.step <= INITIAL_STEP]


def _row(event) -> tuple:
    data = event if isinstance(event, dict) else asdict(event)
    return (int(data["step"]), str(data["target"]), str(data["attr"]), data.get("old"),
            data.get("new"), data.get("evidence_view"))


def _event(data) -> StateEvent:
    if isinstance(data, StateEvent):
        return data
    return StateEvent(desktop=int(data["desktop"]), step=int(data["step"]),
                      target=str(data["target"]), attr=str(data["attr"]),
                      old=data.get("old"), new=data.get("new"),
                      evidence_view=data.get("evidence_view"),
                      auto=bool(data.get("auto", False)))


# --------------------------------------------------------------------------- #
# the patch
# --------------------------------------------------------------------------- #
@dataclass
class AbsentPatch:
    """One write of the op: the target's step-0 hand-written events, before and after.

    ``absent`` is what the write says (``True``: 「这台没有这个零件」, ``False``:
    「其实有」); ``rechecks`` (``view -> steps``) are the frozen frames whose
    work it changes, filled in by :func:`write_rows`.
    """

    desktop: int
    key: str
    absent: bool
    before: list[StateEvent] = field(default_factory=list)
    after: list[StateEvent] = field(default_factory=list)
    rechecks: dict[str, list[int]] = field(default_factory=dict)

    def _rows(self, absent: bool, events: list[StateEvent]) -> dict:
        return {"desktop": int(self.desktop), "instance": self.key, "absent": bool(absent),
                "events": [asdict(e) for e in events]}

    def payload(self) -> dict:
        return self._rows(self.absent, self.after) | {
            "rechecks": {v: list(s) for v, s in self.rechecks.items()}}

    def inverse(self) -> dict:
        return self._rows(not self.absent, self.before)

    @property
    def frames(self) -> int:
        """How many frozen frames the write re-checks, every view together."""
        return sum(len(steps) for steps in self.rechecks.values())


def plan_absent(db: Db, tax: Taxonomy, desktop: int, key: str) -> AbsentPatch:
    """「这台没有这个零件」 for ``key``, worked out without writing anything."""
    why = refusal(db, tax, desktop, key)
    if why:
        raise AbsentError(why)
    desktop, key = int(desktop), str(key)
    rec = db.instances(desktop)[key]
    before = _initial_rows(db, desktop, key)
    held = initial_overrides(before).get(key) or tax.default_state(rec.cls)
    return AbsentPatch(desktop, key, True, before, absent_events(desktop, key, held))


def latest_mark(db: Db, desktop: int, key: str) -> Optional[dict]:
    """The newest ``absent_from_start`` op that said ``key`` absent, or ``None``."""
    for op in reversed(db.ops_of_kind(int(desktop), OP_KIND)):
        payload = op.get("payload") or {}
        if payload.get("instance") == key and payload.get("absent"):
            return op
    return None


def plan_present(db: Db, tax: Taxonomy, desktop: int, key: str) -> AbsentPatch:
    """「其实有」: take 「这台没有这个零件」 back, worked out without writing.

    The step-0 events go; what stood there before the op that wrote them is
    put back when the stored events are still exactly what it wrote (an
    initial state written by hand), and nothing otherwise.
    """
    desktop, key = int(desktop), str(key)
    if key not in db.instances(desktop):
        raise AbsentError(f"D{desktop:02d} 没有零件 {key}")
    if not is_absent(db, desktop, key):
        raise AbsentError(f"{key} 没有标成「{SHORT}」，不用改")
    before = _initial_rows(db, desktop, key)
    restore: list[StateEvent] = []
    op = latest_mark(db, desktop, key)
    if op is not None:
        wrote = [_row(e) for e in (op.get("payload") or {}).get("events") or []]
        if wrote == [_row(e) for e in before]:
            restore = [_event(e) for e in (op.get("inverse") or {}).get("events") or []]
    return AbsentPatch(desktop, key, False, before, restore)


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def _reach(db: Db, tax: Taxonomy, desktop: int, key: str) -> tuple[set[int], bool]:
    """``(steps where key needs geometry, is it a gallery part)`` as stored now."""
    instances = db.instances(desktop)
    manual = [e for e in db.events(desktop) if not e.auto]
    derived = events_from_actions(instances, db.actions(desktop), tax,
                                  initial=initial_overrides(manual))
    merged = merge_events(derived, manual)
    steps = {int(r.step) for r in db.steps(desktop)}
    needed = {s for s in steps
              if key in needs_geom(instances, state_at(instances, merged, s, tax), tax)}
    gallery = key in needs_geom(instances, gallery_state(instances, merged, tax,
                                                         GALLERY_EXCLUDED_CLASSES), tax)
    return needed, gallery


def write_rows(db: Db, tax: Taxonomy, desktop: int, key: str,
               rows: Iterable) -> dict[str, list[int]]:
    """Make ``key``'s hand-written events at or before step 0 exactly ``rows``.

    The one writer of the op, every direction: ``rows`` are
    :class:`StateEvent` objects or their op-log dicts. In one transaction (the
    caller's, when it has one) it also recompiles the stored automatic events
    (the sheet's actions on an absent part derive nothing), re-stamps the
    graph meta and queues the frozen frames whose work changed -- every frame
    the part needed geometry on before or after the write, in every view, and
    every components-gallery shot when its membership there changed. Returns
    those as ``view -> steps``.
    """
    from tda.core.extra import recompile_events, restamp_graph

    desktop, key = int(desktop), str(key)
    events = [_event(e) for e in rows]
    with db.transaction():
        before, gallery_before = _reach(db, tax, desktop, key)
        db.delete_manual_events(desktop, key, up_to_step=INITIAL_STEP)
        db.add_events(events)
        recompile_events(db, desktop, tax)
        restamp_graph(db, desktop, tax)
        after, gallery_after = _reach(db, tax, desktop, key)
        reach = before | after
        queued: dict[str, list[int]] = {}
        for view in ALL_VIEWS:
            if is_component_view(view):
                if gallery_before == gallery_after:
                    continue
                wanted = [int(r["step"]) for r in db.frames_for(desktop, view)]
            else:
                wanted = sorted(reach)
            frozen = db.frozen_steps_in(desktop, view, wanted) if wanted else []
            if frozen:
                db.add_rechecks(desktop, view, frozen)
                queued[view] = list(frozen)
    return queued


def _commit(db: Db, tax: Taxonomy, patch: AbsentPatch, annotator: str,
            view: str) -> int:
    """Write ``patch`` and log it with its inverse; returns the op-log id."""
    with db.transaction():
        patch.rechecks = write_rows(db, tax, patch.desktop, patch.key, patch.after)
        return db.log_op(patch.desktop, str(view or OP_VIEW), OP_KIND, patch.payload(),
                         patch.inverse(), str(annotator or ""))


def mark_absent(db: Db, tax: Taxonomy, desktop: int, key: str, annotator: str,
                view: str = OP_VIEW) -> tuple[int, AbsentPatch]:
    """「这台没有这个零件」: write it; returns ``(op id, patch)``."""
    with db.transaction():
        patch = plan_absent(db, tax, desktop, key)
        return _commit(db, tax, patch, annotator, view), patch


def mark_present(db: Db, tax: Taxonomy, desktop: int, key: str, annotator: str,
                 view: str = OP_VIEW) -> tuple[int, AbsentPatch]:
    """「其实有」: take it back; returns ``(op id, patch)``."""
    with db.transaction():
        patch = plan_present(db, tax, desktop, key)
        return _commit(db, tax, patch, annotator, view), patch
