"""「其实是硬盘架」 / 「其实是卡扣」: a logged part reclassed, everything drawn kept (task U21).

The sheets name the HDD/SSD cage and its latch inconsistently, and the
importer's name patterns made the cage a latch on three annotated desktops:
D2's "remove SSD locker shield" (``drive_latch.02``, a 106 k px metal cage), D3's
"HDD Locker Shield" (``drive_latch.02``), D4's "HDD Shield Locker" (the cage and
its blue latch drawn as one mask). All three carry shapes, confirmed rows,
relations and ``of`` references on the misclassified key.

A reclass **is a merge into a fresh instance of the other class**: the next
key of the new class (one no part ever had: the instance table and every key
the op log names are skipped) is written with the record
:func:`tda.core.reclass_replay.reclassed_record` gives -- ``of`` and the raw
names carried, the columns that no longer make sense cleared -- and
:mod:`tda.core.merge_instances` folds the old key into it with the class check
lifted: keyframes, confirmed rows (renamed, byte for byte), overrides, flags,
conflicts, z-order, pair overrides, actions (the verb stays, at the same
step), hand-written events (one the new class has no state for is dropped and
said), relations and every pointer, ``attrs["of"]`` included.

**The cage rule follows** (:mod:`tda.core.implied`, task U20), in the same
transaction and the same op: a latch that became the cage gets its latch --
the desktop's logged locker opened before the cage comes out and never taken
out on its own is hung on it (``parent``, ``attached``; D2's "SSD Locker",
D3's "HDD Locker" once their verbs say opened), else the cage's implied latch
is made (closed, open on the frame before, out with the cage; D4). A cage that
was a latch after all lets go of its latches: an untouched implied one is
deleted, one with work refuses the reclass (it would be a latch's latch), the
others are unhung.

Refused (:class:`ReclassError`): a class other than the two, a Label Studio
draft, a 框外零件, an implied part (delete it instead), an action whose verb the
new class does not take (a latch only opened is no cage -- 「其实拆下来了」 first),
a cage whose implied latch has work when it is to become a latch, and whatever
the merge itself refuses. One ``reclass_instance`` row in the op log; its
inverse puts everything back (:func:`undo_reclass`), :func:`redo_reclass` runs
it again; a forced re-import replays it (:mod:`tda.core.reclass_replay`).
"""
from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

from tda.core import cage_patch as P
from tda.core.db import Db
from tda.core.implied import (
    CAGE_CLASS,
    LATCH_CLASS,
    OP_DELETE_IMPLIED,
    cage_rule_for,
    is_cage_latch,
    is_implied,
)
from tda.core.merge_instances import (
    THEN_MERGE,
    MergeError,
    MergePlan,
    _confirmed,
    _confirmed_since,
    _frames_text,
    _inverse,
    _settle,
    _undo_frames,
    plan_merge,
    undo_rows,
    write_merge,
)
from tda.core.merge_replay import OP_VIEW, ranges
from tda.core.model import (
    ALL_VIEWS,
    FrameKey,
    InstanceRec,
    is_component_view,
    is_context,
    is_provisional,
)
from tda.core.reclass_replay import (
    CAPTION,
    CLASS_ZH,
    OP_RECLASS,
    OP_UNDO_RECLASS,
    RECLASS_TO,
    reclass_target,
    reclassed_record,
    verb_applies,
)
from tda.core.states import needs_geom
from tda.core.taxonomy import Taxonomy
from tda.core.truth_inputs import InputCache, instances_of, pose_segment_of, state_of

__all__ = [
    "CAPTION", "CLASS_ZH", "OP_RECLASS", "OP_UNDO_RECLASS", "ReclassError", "ReclassPlan",
    "ReclassReport", "fresh_key", "latch_of", "latest_reclass", "plan_reclass", "reclass_gate",
    "reclass_instance", "reclass_target", "redo_reclass", "undo_reclass",
]

VERIFIED = "verified"


class ReclassError(MergeError):
    """A reclass (or its undo) that is refused; the text is UI-ready Chinese."""


def _reworded(error: Exception) -> str:
    """A merge refusal said as the reclass it is."""
    return (str(error).replace(THEN_MERGE, "，再改类别").replace("合并后", "改类别后")
            .replace("合并", "改类别"))


# --------------------------------------------------------------------------- #
# keys
# --------------------------------------------------------------------------- #
def _history_keys(db: Db, desktop: int, cls: str) -> set[str]:
    """Every ``<cls>.NN`` key the op log of ``desktop`` names, payload or inverse."""
    pattern = re.compile(rf"\b{re.escape(cls)}\.\d+\b")
    found: set[str] = set()
    like = f"%{cls}.%"
    for row in db.conn.execute(
            "SELECT payload_json, inverse_json FROM op_log WHERE desktop=? "
            "AND (payload_json LIKE ? OR inverse_json LIKE ?)", (int(desktop), like, like)):
        for text in row:
            found.update(pattern.findall(str(text or "")))
    return found


def _ordinal(key: str, cls: str) -> int:
    head, _, tail = str(key).rpartition(".")
    return int(tail) if head == cls and tail.isdigit() else 0


def fresh_key(db: Db, desktop: int, cls: str, also: Iterable[str] = ()) -> str:
    """The next key of ``cls`` no part ever had on ``desktop``.

    Past the instance table, every key the op log names (a merged-away
    ``gone``, a reclassed-away ``old``, a deleted implied latch...) and
    ``also``: a merge or a replay that finds its old key again would read it as
    undone, and a correction made on the old part would follow to the new one.
    """
    keys = set(db.instances(int(desktop))) | _history_keys(db, desktop, cls) | set(also)
    top = max((_ordinal(k, cls) for k in keys), default=0)
    return f"{cls}.{top + 1:02d}"


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def _own_actions(db: Db, desktop: int, key: str) -> list:
    return [a for a in db.actions(desktop) if a.target == key]


def reclass_gate(db: Db, tax: Taxonomy, desktop: int, key: str,
                 new_cls: Optional[str] = None) -> str:
    """Why ``key`` cannot be reclassed (to ``new_cls``), ``""`` when it may. Writes nothing.

    The cheap refusals, said before the confirmation: the merge's own are
    found by :func:`plan_reclass`.
    """
    desktop = int(desktop)
    instances = db.instances(desktop)
    rec = instances.get(key)
    if rec is None:
        return f"D{desktop:02d} 没有零件 {key}"
    if is_provisional(key):
        return f"{key} 是 Label Studio 草稿：草稿要采纳（Shift+A），不是改类别"
    if is_context(rec):
        return f"{key} 是框外零件：没有类别可改（不要了就在实例表右键删除）"
    target = reclass_target(rec.cls)
    if target is None:
        return (f"{key}（{rec.cls}）不是硬盘架或它的卡扣：只有这两类会被日志弄混，"
                f"别的类别改不了")
    if new_cls is not None and str(new_cls) != target:
        return f"{key} 是 {rec.cls}，只能改成 {target}"
    if is_implied(rec):
        return (f"{key} 是隐含零件（日志里没有）：不用改类别，不要就到 Steps 右键删除"
                f"（删除会被记住）")
    for action in _own_actions(db, desktop, key):
        if not verb_applies(tax, action.verb, target):
            if action.verb == "open":
                return (f"第 {action.step} 步 {key} 是「打开」：{CLASS_ZH[target]}不会只是打开。"
                        f"它其实拆下来了就先按 Shift+T（其实拆下来了），再改类别")
            return (f"第 {action.step} 步 {key} 的动作是 {action.verb}：{target} 没有这个动作，"
                    f"先在 Steps 把那一步改对，再改类别")
    if rec.cls == CAGE_CLASS:
        busy = [other for other, held in sorted(instances.items())
                if is_cage_latch(held) and held.parent == key and P.has_work(db, desktop, other)]
        if busy:
            return (f"{key} 的隐含卡扣 {'、'.join(busy)} 已经画过形状或手设过状态：它改成卡扣以后"
                    f"那就是卡扣的卡扣。先到 Steps 右键删掉 {'、'.join(busy)}（或撤销那些修改），"
                    f"再改类别")
    return ""


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
@dataclass
class ReclassPlan:
    """Everything one reclass writes -- read off the write itself, in a transaction."""

    desktop: int
    old: str
    new: str
    old_cls: str
    new_cls: str
    old_record: InstanceRec
    record: InstanceRec
    #: ``{"attrs": {...}, "columns": {...}}`` the new record does not keep.
    left_out: dict
    merge: MergePlan
    #: The cage patch (:mod:`tda.core.cage_patch`), ``None`` when the rule asks nothing.
    cage: Optional[dict] = None
    #: ``view -> steps`` of the frozen frames queued for a re-check.
    rechecks: dict[str, list[int]] = field(default_factory=dict)
    #: ``{"gained" | "lost": {view: {step: [keys]}}}``: confirmed frames whose
    #: parts change, which the re-check will send back to 待复核.
    demotes: dict = field(default_factory=dict)
    steps: list[int] = field(default_factory=list)
    verbs: list[list] = field(default_factory=list)

    # -- what the confirmation and the status line say -------------------- #
    def _name(self, rec: InstanceRec) -> str:
        return f"{rec.key}（{'、'.join(rec.raw_names)}）" if rec.raw_names else rec.key

    def demoted(self) -> dict[str, list[int]]:
        """``view -> steps`` of the confirmed frames that will go back to 待复核."""
        out: dict[str, set[int]] = {}
        for kind in ("gained", "lost"):
            for view, steps in (self.demotes.get(kind) or {}).items():
                out.setdefault(view, set()).update(int(s) for s in steps)
        return {view: sorted(steps) for view, steps in sorted(out.items()) if steps}

    def cage_lines(self, done: bool = False) -> list[str]:
        """What happens to the cage's latches, one line each (``done``: it happened)."""
        patch = self.cage or {}
        cage = patch.get("cage") or self.new
        did = "已" if done else ""
        lines = []
        for key in patch.get("adopted") or ():
            lines.append(f"{key} {did}挂到 {cage} 上，随机架一起拆下（第 {patch.get('out_at')} 步）")
        for move in patch.get("repoint") or ():
            if move["key"] not in (patch.get("adopted") or ()):
                lines.append(f"{move['key']} {did}不再挂在 {cage} 上（parent 清空）")
        for record in patch.get("add") or ():
            opens = patch.get("opens")
            when = (f"关着，默认第 {opens} 帧打开，第 {patch.get('out_at')} 步随机架拆下"
                    if opens is not None else "关着，随机架拆下")
            lines.append(f"机架 {cage} {'因此' if done else ''}自动得到隐含卡扣 {record['key']}"
                         f"（{when}）")
        for record in patch.get("drop") or ():
            lines.append(f"{self.old} 的隐含卡扣 {record['key']} 用不着了，{did}删掉（没画过）")
        return lines

    def summary_lines(self, undo_hint: str = "（标注模式里 Ctrl+Z 撤销）") -> list[str]:
        """The confirmation: the sentence first, then what moves, the cage, the frames."""
        to = CLASS_ZH.get(self.new_cls, self.new_cls)
        acts = "、".join(f"第 {s} 步「{v}」" for s, v in self.verbs) or "没有动作"
        lines = [f"把 {self._name(self.old_record)} 改成{to} {self.new}：形状、确认过的行、"
                 f"动作、关系和指向它的字段都搬过去，动作不变（{acts}）。",
                 "搬过去 / moves:"]
        lines.extend(self.merge.moved_lines())
        drops = self.merge.dropped_lines()
        if drops:
            lines.append("丢掉 / dropped:")
            lines.extend(drops)
        left = {**self.left_out.get("attrs", {}), **self.left_out.get("columns", {})}
        if left:
            lines.append(f"{self.new} 不带的字段：{left}（记在操作日志里）")
        cage = self.cage_lines()
        if cage:
            lines.append("机架的卡扣 / the cage's latch:")
            lines.extend(f"  · {text}" for text in cage)
        n = sum(len(v) for v in self.rechecks.values())
        if n:
            demoted = self.demoted()
            out = (f"，其中 {sum(len(v) for v in demoted.values())} 帧会退回待复核"
                   f"（{_frames_text(demoted)}：{self._demote_reason()}）" if demoted else "")
            lines.append(f"{n} 个已确认帧会在后台重新核对{out}。")
        lines.append(f"立即写入，不经 Apply{undo_hint}。")
        return lines

    def _demote_reason(self) -> str:
        said = []
        for kind, word in (("gained", "多了"), ("lost", "少了")):
            keys = sorted({k for steps in (self.demotes.get(kind) or {}).values()
                           for ks in steps.values() for k in ks})
            if keys:
                said.append(f"{word} {'、'.join(keys)}")
        return "，".join(said)

    def status_line(self) -> str:
        """What the status line says after the reclass."""
        to = CLASS_ZH.get(self.new_cls, self.new_cls)
        merge = self.merge
        kfs = sum(len(v) for v in merge.keyframes.values())
        parts = [f"形状 {kfs}"] if kfs else []
        if merge.compiled_renamed:
            parts.append(f"已确认的行 {len(merge.compiled_renamed)}")
        if merge.actions:
            parts.append(f"动作 {len(merge.actions)}")
        moved = "、".join(parts) or "没有形状和动作"
        text = f"{self.old} 其实是{to}：现在是 {self.new}（{moved} 搬过去）"
        cage = self.cage_lines(done=True)
        if cage:
            text += "；" + "；".join(cage)
        n = sum(len(v) for v in self.rechecks.values())
        if n:
            demoted = sum(len(v) for v in self.demoted().values())
            text += f"；{n} 个已确认帧在后台重新核对" + (
                f"，其中 {demoted} 帧会退回待复核" if demoted else "")
        return text


@dataclass
class ReclassReport:
    """One reclass written: the op-log row and the plan it carried out."""

    op_id: int
    plan: ReclassPlan

    def status_line(self) -> str:
        return self.plan.status_line()


# --------------------------------------------------------------------------- #
# what changes on the confirmed frames
# --------------------------------------------------------------------------- #
def _views(db: Db, desktop: int) -> list[str]:
    return sorted({str(r["view"]) for r in db.conn.execute(
        "SELECT DISTINCT view FROM frame WHERE desktop=?", (int(desktop),))})


def _needs(db: Db, tax: Taxonomy, desktop: int) -> dict[tuple[str, int], set[str]]:
    """``(view, step) -> the parts it needs`` of every frozen frame, as the compiler asks it."""
    cache = InputCache()
    instances = instances_of(db, desktop, cache)
    out: dict[tuple[str, int], set[str]] = {}
    for view in _views(db, desktop):
        for step in db.frozen_steps(desktop, view):
            key = FrameKey(desktop, int(step), view)
            state = state_of(db, tax, desktop, int(step), cache, view=view)
            roi = (None if is_component_view(view)
                   else db.bench_roi(desktop, view, pose_segment_of(db, key, cache)))
            out[(view, int(step))] = set(needs_geom(instances, state, tax, bench_roi=roi))
    return out


def _changes(before: dict, after: dict, old: str, new: str, verified: set
             ) -> tuple[dict, dict[str, list[int]]]:
    """``(demotes, rechecks)``: the frozen frames whose parts change, ``old`` read as ``new``."""
    demotes: dict = {"gained": {}, "lost": {}}
    rechecks: dict[str, list[int]] = {}
    for (view, step), had in sorted(before.items()):
        had = {new if k == old else k for k in had}
        has = after.get((view, step), set())
        gained, lost = sorted(has - had), sorted(had - has)
        if not (gained or lost):
            continue
        rechecks.setdefault(view, []).append(step)
        if (view, step) not in verified:
            continue
        if gained:
            demotes["gained"].setdefault(view, {})[step] = gained
        if lost:
            demotes["lost"].setdefault(view, {})[step] = lost
    return {k: v for k, v in demotes.items() if v}, rechecks


def _verified(db: Db, desktop: int) -> set[tuple[str, int]]:
    return {(str(r["view"]), int(r["step"])) for r in db.conn.execute(
        "SELECT view, step FROM frame WHERE desktop=? AND review_status=?",
        (int(desktop), VERIFIED))}


# --------------------------------------------------------------------------- #
# staging: the write itself
# --------------------------------------------------------------------------- #
def _cage_patch(db: Db, tax: Taxonomy, desktop: int, plan: ReclassPlan,
                before: dict[str, InstanceRec],
                latch_key: Optional[str] = None) -> Optional[dict]:
    """What the cage rule asks for after the merge (read on the merged table).

    ``latch_key`` is the key a redo gives the implied latch: the one the
    reclass gave it the first time.
    """
    instances = db.instances(desktop)
    new = plan.new
    if plan.new_cls == CAGE_CLASS:
        fix = cage_rule_for(instances, db.actions(desktop), tax, new,
                            stored={**before, **instances},
                            deleted=db.ops_of_kind(desktop, OP_DELETE_IMPLIED))
        if fix is None:
            return None
        add = []
        for made in fix.add:
            # a key no part ever had (the old one included), as for the new one
            key = (latch_key if latch_key and latch_key not in instances
                   else fresh_key(db, desktop, LATCH_CLASS, also=set(before) | {plan.old}))
            add.append(asdict(InstanceRec(**{**asdict(made), "key": key})))
        repoint = [{"key": key, "before": [instances[key].parent, bool(instances[key].attached)],
                    "after": [new, True]} for key in fix.adopt]
        drop = [asdict(instances[key]) for key in fix.drop
                if not P.has_work(db, desktop, key)]
        patch = {"desktop": int(desktop), "cage": new, "add": add, "drop": drop,
                 "repoint": repoint, "adopted": list(fix.adopt), "kept": [],
                 "opens": fix.opens, "out_at": fix.out_at}
    else:
        # a cage that was a latch: its implied latch goes (one with work was
        # refused by the gate), its other latches are unhung
        drop, repoint = [], []
        for key, rec in sorted(instances.items()):
            if rec.cls != LATCH_CLASS or rec.parent != new or key == new:
                continue
            if is_cage_latch(rec):
                drop.append(asdict(rec))
            else:
                repoint.append({"key": key, "before": [rec.parent, bool(rec.attached)],
                                "after": [None, False]})
        patch = {"desktop": int(desktop), "cage": plan.old, "add": [], "drop": drop,
                 "repoint": repoint, "adopted": [], "kept": [], "opens": None,
                 "out_at": None}
    if not (patch["add"] or patch["drop"] or patch["repoint"]):
        return None
    return patch


def _stage(db: Db, tax: Taxonomy, desktop: int, key: str, new_cls: Optional[str],
           new_key: Optional[str], latch_key: Optional[str] = None) -> ReclassPlan:
    """Write the reclass; **inside the caller's transaction**. Raises :class:`ReclassError`."""
    desktop, key = int(desktop), str(key).strip()
    why = reclass_gate(db, tax, desktop, key, new_cls)
    if why:
        raise ReclassError(why)
    before = db.instances(desktop)
    rec = before[key]
    target = RECLASS_TO[rec.cls]
    new = str(new_key).strip() if new_key else fresh_key(db, desktop, target)
    if new in before:
        raise ReclassError(f"{new} 已经是另一个零件的键了：改不成它")
    record, left_out = reclassed_record(rec, target, new, tax)
    own = sorted(_own_actions(db, desktop, key), key=lambda a: (a.step, a.idx))
    needs_before = _needs(db, tax, desktop)
    db.upsert_instance(record)
    try:
        merge = plan_merge(db, tax, desktop, key, new, reclass=True)
    except MergeError as refused:
        raise ReclassError(_reworded(refused)) from refused
    write_merge(db, merge)
    plan = ReclassPlan(desktop=desktop, old=key, new=new, old_cls=rec.cls, new_cls=target,
                       old_record=copy.deepcopy(rec), record=record, left_out=left_out,
                       merge=merge, steps=sorted({int(a.step) for a in own}),
                       verbs=[[int(a.step), str(a.verb)] for a in own])
    plan.cage = _cage_patch(db, tax, desktop, plan, before, latch_key)
    try:
        P.apply_cage_patch(db, plan.cage)
    except P.CagePatchError as refused:
        raise ReclassError(str(refused)) from refused
    _settle(db, tax, desktop, merge.frames, {})
    plan.demotes, changed = _changes(needs_before, _needs(db, tax, desktop), key, new,
                                     _verified(db, desktop))
    rechecks = {view: set(steps) for view, steps in merge.rechecks.items()}
    for view, steps in changed.items():
        rechecks.setdefault(view, set()).update(steps)
    plan.rechecks = {view: sorted(steps) for view, steps in sorted(rechecks.items()) if steps}
    for view, steps in plan.rechecks.items():
        db.add_rechecks(desktop, view, steps)
    return plan


class _Rehearsal(Exception):
    """Raised to roll a planning transaction back once the plan is in hand."""

    def __init__(self, planned: ReclassPlan) -> None:
        super().__init__("rehearsal")
        self.planned = planned


def plan_reclass(db: Db, tax: Taxonomy, desktop: int, key: str,
                 new_cls: Optional[str] = None, new_key: Optional[str] = None) -> ReclassPlan:
    """What reclassing ``key`` would do; **nothing is written**.

    The reclass is written in a transaction that is rolled back once the plan
    is read, so the confirmation says exactly what the write will do. Refused
    inside another transaction, whose rollback this would be.
    """
    if getattr(db, "_tx_depth", 0):
        raise ReclassError("改类别的预演不能在另一个事务里进行 / not inside another transaction")
    try:
        with db.transaction():
            raise _Rehearsal(_stage(db, tax, desktop, key, new_cls, new_key))
    except _Rehearsal as done:
        return done.planned


def _payload(plan: ReclassPlan) -> dict:
    return {
        "old": plan.old, "new": plan.new, "old_cls": plan.old_cls, "new_cls": plan.new_cls,
        "steps": list(plan.steps), "raw_names": list(plan.old_record.raw_names),
        "verbs": [list(v) for v in plan.verbs], "record": asdict(plan.record),
        "gone_record": asdict(plan.old_record), "left_out": plan.left_out,
        "moved": plan.merge.moved(), "dropped": plan.merge.dropped(),
        "dropped_events": [{"id": r["id"], "step": r["step"], "attr": r["attr"],
                            "old": r["old"], "new": r["new"], "why": why}
                           for r, why in plan.merge.events_dropped],
        P_CAGE: plan.cage, "rechecks": plan.rechecks,
        "demotes": {kind: {view: {str(s): ks for s, ks in steps.items()}
                           for view, steps in views.items()}
                    for kind, views in plan.demotes.items()},
    }


#: ``payload`` / ``inverse`` key of the cage patch (as ``Shift+T``'s).
P_CAGE = "cage"


def reclass_instance(db: Db, tax: Taxonomy, desktop: int, key: str,
                     new_cls: Optional[str] = None, annotator: str = "",
                     new_key: Optional[str] = None,
                     latch_key: Optional[str] = None) -> ReclassReport:
    """Reclass ``key`` into the other bay class, atomically, with one op-log row.

    ``new_key`` and ``latch_key`` are for a redo: the keys the reclass gave the
    new part and the cage's implied latch the first time.
    """
    with db.transaction():
        plan = _stage(db, tax, desktop, key, new_cls, new_key, latch_key)
        inverse = _inverse(plan.merge)
        inverse.update({"new": plan.new, "record": asdict(plan.record),
                        P_CAGE: P.mirror(plan.cage)})
        inverse["confirmed"] = _confirmed(db, plan.desktop, _undo_frames(inverse))
        op_id = db.log_op(plan.desktop, OP_VIEW, OP_RECLASS, _payload(plan), inverse,
                          annotator or "ui")
    return ReclassReport(op_id=op_id, plan=plan)


# --------------------------------------------------------------------------- #
# undo / redo
# --------------------------------------------------------------------------- #
def latest_reclass(db: Db, desktop: int, new: str) -> Optional[dict]:
    """The newest reclass row of ``desktop`` that made ``new`` and is not undone."""
    undone = {int((op.get("payload") or {}).get("undid") or 0)
              for op in db.ops_of_kind(int(desktop), OP_UNDO_RECLASS)}
    found = [op for op in db.ops_of_kind(int(desktop), OP_RECLASS)
             if str((op.get("payload") or {}).get("new")) == str(new)
             and int(op["id"]) not in undone]
    return found[-1] if found else None


def _left_on(db: Db, desktop: int, key: str) -> dict[str, int]:
    """What a human put on ``key`` that an undo would have to throw away.

    An open conflict is not one: it is a re-check's question about a confirmed
    row the undo moves back, and the re-check after the undo asks it again.
    """
    counts = {table: n for table, n in db.instance_reference_counts(desktop, key).items()
              if table != "conflict"}
    shapes = sum(len(db.keyframes(desktop, view, key)) for view in ALL_VIEWS)
    if shapes:
        counts["shape_keyframe"] = shapes
    edges = [r for r in db.relations(desktop)
             if key in (r["target"], r["blocker"]) and r["source"] != "rule"]
    if edges:
        counts["relation"] = len(edges)
    if any(a.target == key for a in db.actions(desktop)):
        counts["action"] = sum(1 for a in db.actions(desktop) if a.target == key)
    return counts


def undo_reclass(db: Db, tax: Taxonomy, op: dict, annotator: str = "") -> int:
    """Apply the inverse of one :data:`OP_RECLASS` row; returns the undo's op id.

    The cage's latches go back first (an implied latch the reclass made is
    deleted -- refused once it has work of its own --, a latch it hung on the
    cage is unhung, one it deleted comes back), then every row the merge moved
    (:func:`tda.core.merge_instances.undo_rows`), then the new key is deleted.
    Refused once the old key exists again, the new one is gone, a frame the
    undo would rewrite was confirmed since, or something was put on the new key
    since (a shape, an override, an edge...).
    """
    if str(op.get("kind")) != OP_RECLASS:
        raise ReclassError(f"op {op.get('id')} 不是一次改类别")
    inv = op.get("inverse") or {}
    payload = op.get("payload") or {}
    d = int(op["desktop"])
    old, new = str(inv["gone"]), str(inv["keep"])
    stored = db.instances(d)
    if old in stored:
        raise ReclassError(f"撤销不了：{old} 又存在了")
    if new not in stored or stored[new].cls != payload.get("new_cls"):
        raise ReclassError(f"撤销不了：{new} 已经不在了（或不是 {payload.get('new_cls')}）")
    since = _confirmed_since(db, d, inv)
    if since:
        raise ReclassError(f"撤销不了：改类别后又确认过这些帧（{_frames_text(since)}），撤销会把"
                           f"改类别前的行写回去、盖掉那次确认")
    try:
        with db.transaction():
            try:
                P.apply_cage_patch(db, inv.get(P_CAGE))
            except P.CagePatchError as refused:
                raise ReclassError(str(refused)) from refused
            frames = undo_rows(db, tax, d, inv)
            left = _left_on(db, d, new)
            if left:
                what = "、".join(f"{table} {n}" for table, n in sorted(left.items()))
                raise ReclassError(f"撤销不了：改类别后 {new} 上又有了修改（{what}）。先撤销那些，"
                                   f"再撤销改类别")
            # what is left on it is machinery: derived edges, closed conflicts,
            # cached rows, derived events
            db.conn.execute("DELETE FROM relation WHERE desktop=? AND (target=? OR blocker=?)",
                            (d, new, new))
            db.conn.execute("DELETE FROM conflict WHERE desktop=? AND instance=?", (d, new))
            db.delete_auto_events(d, new)
            db.delete_instance(d, new)
            _settle(db, tax, d, frames, {})
            views = {view for view, _step in frames} | set(payload.get("rechecks") or {})
            views |= {str(r["view"]) for r in db.conn.execute(
                "SELECT DISTINCT view FROM shape_keyframe WHERE desktop=? AND instance=?",
                (d, old))}
            if inv.get(P_CAGE):
                views |= set(_views(db, d))
            for view in sorted(views):
                db.queue_rechecks_for_view(d, view)
            return db.log_op(d, OP_VIEW, OP_UNDO_RECLASS,
                             {"undid": int(op["id"]), "old": old, "new": new},
                             {"redo": {"old": old, "new": new,
                                       "new_cls": payload.get("new_cls"),
                                       "latch": latch_of(payload)}},
                             annotator or str(op.get("annotator") or ""))
    except ReclassError:
        raise
    except Exception as error:  # the transaction rolled back
        raise ReclassError(f"撤销不了：{type(error).__name__}: {error}") from error


def redo_reclass(db: Db, tax: Taxonomy, op: dict, annotator: str = "") -> ReclassReport:
    """Run the reclass an :data:`OP_RECLASS` or :data:`OP_UNDO_RECLASS` row names once more.

    Under the key it gave the first time, the preconditions checked again.
    """
    payload = op.get("payload") or {}
    if str(op.get("kind")) == OP_UNDO_RECLASS:
        names = (op.get("inverse") or {}).get("redo") or {}
    elif str(op.get("kind")) == OP_RECLASS:
        names = {**payload, "latch": latch_of(payload)}
    else:
        raise ReclassError(f"op {op.get('id')} 不是一次改类别")
    return reclass_instance(db, tax, int(op["desktop"]), str(names["old"]),
                            names.get("new_cls"), annotator or str(op.get("annotator") or ""),
                            new_key=str(names["new"]), latch_key=names.get("latch"))


def latch_of(payload: dict) -> Optional[str]:
    """The implied latch a reclass row made for its cage, or ``None``."""
    added = ((payload or {}).get(P_CAGE) or {}).get("add") or []
    return str(added[0]["key"]) if added else None


def ranges_of(steps: Iterable[int]) -> str:
    """``[10, 11, 12]`` -> ``"10–12"`` (re-exported for the UI)."""
    return ranges(steps)
