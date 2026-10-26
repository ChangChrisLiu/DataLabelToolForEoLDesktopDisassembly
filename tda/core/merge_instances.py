"""Merging two instances that are one physical part (task U12, spec 3.2a).

The capture sheets name parts inconsistently, so the log importer sometimes
makes two instances of one part: desktop 2 had ``cover.01`` ("Screw Cover",
opened at step 38) and ``cover.02`` ("Front Cover", taken out at step 46) --
one cover. The task card asked for both, the annotator drew the cover under
``cover.02``, and S1 could retarget an action but never delete an instance with
shapes. This module folds one instance (``gone``) into another (``keep``) of
the same class, in every table, as **one** transaction and **one** ``op_log``
row whose inverse restores everything (:func:`undo_merge`).

What moves (all views)
----------------------
* the shape keyframes (their parts ride along: ``shape_part`` hangs off the
  keyframe id, which does not change);
* the **verified** compiled rows -- a human's signature on the shape --
  renamed. ``gone``'s other rows and ``keep``'s unverified rows on the frames
  ``gone`` reached are a cache and are dropped, so the next refresh compiles
  them from the merged inputs (the frames' digests are cleared);
* frame overrides, per-frame flags and conflicts (renamed); the occluder
  masks belong to no instance;
* the z-order entries (renamed; where both appear in one order, ``gone``'s
  position is kept and ``keep``'s entry dropped -- ``gone`` is the one that was
  drawn) and the pair overrides (renamed; a pair with itself or a duplicate is
  dropped, and so is a ``keep`` pair the renamed one contradicts);
* the actions (``target`` -> ``keep``);
* the constraint edges (re-pointed; self-edges dropped; of two that become
  one, the one a human made wins over a ``rule`` one) and the other instances'
  ``parent``/``mounted_on``/``fastens``/``socket_host`` and ``attrs["of"]``
  pointing at ``gone`` (a pointer of ``keep`` itself at ``gone`` is cleared);
* ``gone``'s hand-written state events, each only where ``keep`` has no
  hand-written event of that attribute at that step and the merged log still
  validates (otherwise dropped and listed). An initial state (step 0) is never
  moved: ``keep`` keeps its own start, as it keeps its attributes. A
  hand-written event of ``keep`` that the merge turns into a no-op (the cover
  the annotator opened by hand at 38 is opened by the retargeted action
  anyway) is dropped too.

``keep`` keeps its record; ``gone``'s whole record -- its provenance if it was
added in S1 (``added_by``...) -- is in the op payload and in the inverse.

Refusals (:class:`MergeError`, Chinese, UI-ready): not the same class, the same
key, a Label Studio draft on either side (a draft is adopted, not merged), both
present **with a shape** on one frame of one view (two keyframe chains
selecting that frame: two parts), both with an action at one step, both with a
frame override / flag / verified row on one frame, a merged state log with a
problem the two logs did not have, a confirmed frame on which the merged part
would no longer match what was confirmed, a ``parent`` loop or a constraint
deadlock the merge would create.

**Overlap** (task U21 round 2): the refusals that only say "both are drawn,
confirmed, overridden or flagged on these frames" are :class:`OverlapError`,
and ``overlap="keep"`` merges anyway -- D5's locker and the latch added by hand
for it are one blue tab drawn twice on frames 1-9. On those frames ``keep``'s
stays; ``gone``'s compiled rows, overrides, flags and conflicts there are
dropped, and so is every keyframe of ``gone``'s the merged chain would still
pick on a frame both are drawn on (anchored there, or reaching back over it);
``gone``'s other shapes move as always. The dropped rows are in the inverse.

Afterwards the automatic event log is recompiled, ``graph_version`` re-stamped,
the digests of the frames ``gone`` reached cleared and every frozen frame the
merged part touches queued for a re-check. :func:`undo_merge` applies the
inverse -- refused once a frame it would rewrite has been confirmed since the
merge -- and :func:`redo_merge` runs the merge again from the op.
"""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from typing import Iterable, Optional

from tda.core import dbrows as R
from tda.core.compiler import select_keyframe
from tda.core.db import Db
from tda.core.model import (
    VIEWS,
    FrameKey,
    InstanceRec,
    StateEvent,
    is_context,
    is_provisional,
)
from tda.core.states import (
    INITIAL_STEP,
    LEAVES_REMOVED,
    OLD_MISMATCH,
    UNKNOWN_VALUE,
    absent_keys,
    event_problems,
    events_from_actions,
    initial_overrides,
    needs_geom,
    state_at,
)
from tda.core.taxonomy import Taxonomy
from tda.core.truth_inputs import events_of, merge_events, pose_segment_of
# the vocabulary the importer shares, kept in the light module it imports
from tda.core.merge_replay import (
    OF_ATTR,
    OP_MERGE,
    OP_UNDO,
    OP_VIEW,
    POINTERS,
    ranges,
    repointed,
)

__all__ = [
    "OP_MERGE", "OP_UNDO", "OP_VIEW", "OVERLAP_KEEP", "OVERLAP_REFUSE", "MergeError",
    "MergePlan", "MergeReport", "OverlapError", "apply_merge", "merge_instances", "plan_merge",
    "ranges", "redo_merge", "undo_merge",
]

VERIFIED = "verified"
#: ``relation.source`` of a derived edge (:data:`tda.core.graph_derive.RULE`,
#: which this module does not import at load time).
RULE = "rule"
#: The instance columns a merge does not carry over from ``gone`` (``keep``
#: keeps its own); the confirmation names the ones that differ.
GONE_COLUMNS = ("parent", "attached", "mounted_on", "fastens", "socket_host",
                "removal_direction")
#: How every refusal that wants the step table fixed first ends (round 3): a
#: separate ``Apply`` would demote the confirmed frames the merge is about to
#: keep (D2: 38-48), 「Apply 并合并」 writes both at once.
THEN_MERGE = "（不用先 Apply），再「合并到…」：确认框会提供「Apply 并合并」，一次写入"


#: How a merge treats the frames on which **both** instances are drawn (or both
#: carry a confirmed row, an override, a flag): refuse, as U12 always did, or
#: keep ``keep``'s there and drop ``gone``'s (task U21 round 2: D5's locker and
#: the latch added by hand for it are one part drawn twice on frames 1-9).
OVERLAP_REFUSE = "refuse"
OVERLAP_KEEP = "keep"


class MergeError(ValueError):
    """A merge (or its undo) that is refused; the text is UI-ready Chinese."""


class OverlapError(MergeError):
    """Refused only because both are drawn or confirmed on the same frames.

    ``overlap="keep"`` merges anyway, keeping ``keep``'s on those frames.
    """


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _frames_text(frames: dict[str, list[int]]) -> str:
    """``{"scan": [38, 39]}`` -> ``"scan 第 38–39 帧"``."""
    return "；".join(f"{view} 第 {ranges(steps)} 帧"
                    for view, steps in sorted(frames.items()) if steps)


def _rows(db: Db, sql: str, *args) -> list[dict]:
    return [dict(row) for row in db.conn.execute(sql, args).fetchall()]


def _insert(db: Db, table: str, row: dict, replace: bool = False) -> None:
    """Put one stored row back exactly as it was read (``id`` included)."""
    cols = list(row)
    verb = "INSERT OR REPLACE" if replace else "INSERT"
    db.conn.execute(
        f"{verb} INTO {table} ({', '.join(f'{chr(34)}{c}{chr(34)}' for c in cols)}) "
        f"VALUES ({', '.join('?' for _ in cols)})",
        [row[c] for c in cols],
    )


def _event(row: dict, target: Optional[str] = None) -> StateEvent:
    return StateEvent(desktop=int(row["desktop"]), step=int(row["step"]),
                      target=str(target or row["target"]), attr=str(row["attr"]),
                      old=row["old"], new=row["new"], evidence_view=row["evidence_view"],
                      auto=bool(row["auto"]))


def _views(db: Db, desktop: int, keys: Iterable[str]) -> list[str]:
    """Every view the desktop has frames in or the two keys have shapes in."""
    keys = list(keys)
    marks = ",".join("?" for _ in keys)
    found = {r["view"] for r in db.conn.execute(
        "SELECT DISTINCT view FROM frame WHERE desktop=?", (desktop,))}
    found |= {r["view"] for r in db.conn.execute(
        f"SELECT DISTINCT view FROM shape_keyframe WHERE desktop=? AND instance IN ({marks})",
        (desktop, *keys))}
    return sorted(found)


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
@dataclass
class MergePlan:
    """Everything one merge would write, read before anything is."""

    desktop: int
    gone: str
    keep: str
    cls: str
    gone_record: InstanceRec
    keep_record: InstanceRec
    #: Stored records of the instances whose pointers change, before / after.
    neighbours_before: list[InstanceRec] = field(default_factory=list)
    neighbours_after: list[InstanceRec] = field(default_factory=list)
    actions: list[tuple[int, int]] = field(default_factory=list)
    #: ``view -> [(keyframe id, anchor step)]``.
    keyframes: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    compiled_renamed: list[tuple[int, str]] = field(default_factory=list)
    compiled_dropped: list[dict] = field(default_factory=list)
    overrides: list[tuple[int, str]] = field(default_factory=list)
    flags: list[tuple[int, str]] = field(default_factory=list)
    conflicts: list[int] = field(default_factory=list)
    #: ``{"view", "seg", "before", "version", "after", "dropped"}`` per order.
    zorder: list[dict] = field(default_factory=list)
    pairs_renamed: list[tuple[dict, dict]] = field(default_factory=list)
    pairs_dropped: list[tuple[dict, str]] = field(default_factory=list)
    relations_renamed: list[tuple[dict, str, str]] = field(default_factory=list)
    relations_dropped: list[tuple[dict, str]] = field(default_factory=list)
    events_moved: list[dict] = field(default_factory=list)
    events_dropped: list[tuple[dict, str]] = field(default_factory=list)
    #: ``(view, step)`` of every frame the merge reaches: digests cleared.
    frames: list[tuple[str, int]] = field(default_factory=list)
    #: ``view -> steps`` of the frozen frames to queue for a re-check.
    rechecks: dict[str, list[int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    #: :data:`OVERLAP_REFUSE` or :data:`OVERLAP_KEEP` (task U21 round 2).
    overlap: str = OVERLAP_REFUSE
    #: ``view -> steps`` where both are drawn or confirmed and ``keep``'s stays.
    overlap_frames: dict[str, list[int]] = field(default_factory=dict)
    #: Of those, the ones where both are **drawn** (both keyframe chains pick a shape).
    overlap_shapes: dict[str, list[int]] = field(default_factory=dict)
    #: What of ``gone`` goes on those frames, as stored rows (the inverse puts
    #: them back): keyframes (with their parts), overrides, flags, conflicts.
    #: Its compiled rows there are in :attr:`compiled_dropped`.
    keyframes_dropped: list[dict] = field(default_factory=list)
    parts_dropped: list[dict] = field(default_factory=list)
    overrides_dropped: list[dict] = field(default_factory=list)
    flags_dropped: list[dict] = field(default_factory=list)
    conflicts_dropped: list[dict] = field(default_factory=list)
    #: ``gone``'s confirmed rows among :attr:`compiled_dropped` (overlap frames).
    verified_dropped: int = 0

    # -- counts ---------------------------------------------------------- #
    def moved(self) -> dict[str, int]:
        """What moves to ``keep``, per table (zero counts left out)."""
        counts = {
            "shape_keyframe": sum(len(v) for v in self.keyframes.values()),
            "compiled_mask": len(self.compiled_renamed),
            "frame_override": len(self.overrides),
            "instance_frame_flags": len(self.flags),
            "conflict": len(self.conflicts),
            "zorder": len(self.zorder),
            "pair_override": len(self.pairs_renamed),
            "relation": len(self.relations_renamed),
            "action": len(self.actions),
            "state_event": len(self.events_moved),
            "instance": len(self.neighbours_after),
        }
        return {k: n for k, n in counts.items() if n}

    def dropped(self) -> dict[str, int]:
        """What goes, per table (zero counts left out)."""
        counts = {
            "compiled_mask": len(self.compiled_dropped),
            "zorder": sum(len(z["dropped"]) for z in self.zorder),
            "pair_override": len(self.pairs_dropped),
            "relation": len(self.relations_dropped),
            "state_event": len(self.events_dropped),
            "shape_keyframe": len(self.keyframes_dropped),
            "frame_override": len(self.overrides_dropped),
            "instance_frame_flags": len(self.flags_dropped),
            "conflict": len(self.conflicts_dropped),
        }
        return {k: n for k, n in counts.items() if n}

    # -- what the confirmation says -------------------------------------- #
    def summary_lines(self, with_apply: bool = False) -> list[str]:
        """The confirmation dialog, one line per thing, in the annotator's words.

        ``with_apply``: the merge is written together with the staged S1 edits
        (「Apply 并合并」), which the last line says instead of "不经 Apply".
        """
        name = lambda rec: (f"{rec.key}（{'、'.join(rec.raw_names)}）"   # noqa: E731
                            if rec.raw_names else rec.key)
        keeping = f"（重叠处保留 {self.keep}）" if self.overlap == OVERLAP_KEEP else ""
        lines = [f"把 {name(self.gone_record)} 合并到 {name(self.keep_record)}{keeping}：",
                 "搬过去 / moves:"]
        lines.extend(self.moved_lines())
        drops = self.dropped_lines()
        if drops:
            lines.append("丢掉 / dropped:")
            lines.extend(drops)
        rest = {k: v for k, v in self.gone_record.attrs.items()
                if self.keep_record.attrs.get(k) != v}
        tail = f"属性 {rest} " if rest else ""
        lines.append(f"{self.gone} 这一行会删除；它的{tail}不带过去（记在操作日志里）。"
                     f"{self.keep} 的属性不变。")
        columns = [f"{name}={getattr(self.gone_record, name)}（{self.keep} 是 "
                   f"{getattr(self.keep_record, name) or '空'}）"
                   for name in GONE_COLUMNS
                   if getattr(self.gone_record, name) not in (None, False, "")
                   and getattr(self.gone_record, name) != getattr(self.keep_record, name)]
        if columns:
            lines.append(f"{self.gone} 的 {'、'.join(columns)} 也不带过去。")
        for text in self.warnings:
            lines.append(f"注意：{text}")
        if self.rechecks:
            n = sum(len(v) for v in self.rechecks.values())
            lines.append(f"{n} 个已确认帧会在后台重新核对。")
        lines.append("和 Apply 一起立即写入（Revert 撤不回）。" if with_apply
                     else "立即写入，不经 Apply（Revert 撤不回）。")
        return lines

    def moved_lines(self) -> list[str]:
        """The 「搬过去」 lines of the confirmation, one per table that moves."""
        lines: list[str] = []
        for view, kfs in sorted(self.keyframes.items()):
            if kfs:
                lines.append(f"  · 形状 {len(kfs)} 个（{view} 锚在第 "
                             f"{ranges(a for _id, a in kfs)} 帧）")
        if self.compiled_renamed:
            by_view: dict[str, list[int]] = {}
            for step, view in self.compiled_renamed:
                by_view.setdefault(view, []).append(step)
            lines.append(f"  · 已确认的行 {len(self.compiled_renamed)} 行（"
                         f"{_frames_text(by_view)}）")
        if self.actions:
            lines.append(f"  · 动作 {len(self.actions)}（第 {ranges(s for s, _i in self.actions)} "
                         f"步改成 {self.keep}）")
        simple = [("单帧覆盖", len(self.overrides)),
                  ("逐帧标记", len(self.flags)), ("冲突记录", len(self.conflicts)),
                  ("层级", len(self.zorder)), ("成对层级", len(self.pairs_renamed)),
                  ("约束边", len(self.relations_renamed)),
                  ("手写状态事件", len(self.events_moved)),
                  ("指向它的实例字段", len(self.neighbours_after))]
        for label, n in simple:
            if n:
                lines.append(f"  · {label} {n}")
            if label == "约束边" and n:
                rules = sum(1 for row, _t, _b in self.relations_renamed if row["source"] == RULE)
                if rules:
                    lines.append(f"    （其中 {rules} 条规则边将按保留实例 {self.keep} 的属性重算）")
        return lines

    def dropped_lines(self) -> list[str]:
        """The 「丢掉」 lines of the confirmation (empty when nothing goes)."""
        drops: list[str] = []
        if self.overlap_frames:
            what = [f"形状 {len(self.keyframes_dropped)} 个"] if self.keyframes_dropped else []
            if self.verified_dropped:
                what.append(f"已确认的行 {self.verified_dropped} 行")
            for label, rows in (("单帧覆盖", self.overrides_dropped),
                                ("逐帧标记", self.flags_dropped),
                                ("冲突记录", self.conflicts_dropped)):
                if rows:
                    what.append(f"{label} {len(rows)}")
            drops.append(f"  · 重叠处（{_frames_text(self.overlap_frames)}）保留 {self.keep} 的，"
                         f"{self.gone} 在这些帧上的{'、'.join(what) or '缓存行'}不要了")
        cache = len(self.compiled_dropped) - self.verified_dropped
        if cache:
            drops.append(f"  · 缓存行 {cache} 行（会重新编译）")
        for z in self.zorder:
            for inst, part in z["dropped"]:
                drops.append(f"  · {z['view']} 位姿段 {z['seg']} 层级里 {inst}/{part} 原来的位置"
                             f"（保留 {self.gone} 的位置）")
        for row, why in self.pairs_dropped:
            drops.append(f"  · 成对层级 {row['above']} 盖住 {row['below']}（{why}）")
        for row, why in self.relations_dropped:
            drops.append(f"  · 约束边 {row['type']}({row['target']}, {row['blocker']})（{why}）")
        for row, why in self.events_dropped:
            drops.append(f"  · 手写状态事件 {row['target']} 第 {row['step']} 步 "
                         f"{row['old']}→{row['new']}：{why}")
        return drops


# -- pointers ------------------------------------------------------------ #
def _repoint(instances: dict[str, InstanceRec], gone: str, keep: str
             ) -> tuple[dict[str, InstanceRec], list[InstanceRec], list[InstanceRec]]:
    """The instance table without ``gone``, every pointer at it moved to ``keep``.

    Returns ``(table, changed before, changed after)``. The pointers are the
    relational columns and a latch's ``attrs["of"]``
    (:func:`tda.core.merge_replay.repointed`, which the forced re-import uses
    too). ``keep``'s own pointer at ``gone`` would point at itself and is
    cleared (``parent`` takes ``attached`` with it, as everywhere else).
    """
    table: dict[str, InstanceRec] = {}
    before: list[InstanceRec] = []
    after: list[InstanceRec] = []
    for key, rec in instances.items():
        if key == gone:
            continue
        new = repointed(rec, gone, keep)
        if new is None:
            table[key] = copy.deepcopy(rec)
            continue
        before.append(copy.deepcopy(rec))
        after.append(new)
        table[key] = new
    return table, before, after


def _parent_loop(instances: dict[str, InstanceRec]) -> str:
    """``a → b → a`` for the first ``parent`` loop, or ``""``."""
    for key in sorted(instances):
        seen = [key]
        current = instances[key].parent
        while current:
            if current == key:
                return " → ".join([*seen, current])
            if current in seen:
                break
            seen.append(current)
            rec = instances.get(current)
            current = rec.parent if rec else None
    return ""


# -- events -------------------------------------------------------------- #
def _problem_key(problem, rename: Optional[dict] = None) -> tuple:
    event = problem.event
    target = (rename or {}).get(event.target, event.target)
    return (target, event.step, event.attr, event.old, event.new, problem.code)


class _Log:
    """The merged event log under trial: problems beyond the two logs' own."""

    def __init__(self, instances, actions, tax, baseline: Counter):
        self.instances = instances
        self.actions = actions
        self.tax = tax
        self.baseline = baseline

    def check(self, manual: list[StateEvent]) -> tuple[list, Counter, list[StateEvent]]:
        derived = events_from_actions(self.instances, self.actions, self.tax,
                                      initial=initial_overrides(manual))
        merged = merge_events(derived, manual)
        problems = event_problems(self.instances, merged, self.tax)
        extra = Counter(_problem_key(p) for p in problems) - self.baseline
        return problems, extra, merged


def _why(problem, candidate: StateEvent, cls: str = "") -> str:
    """Why a hand-written event cannot follow the merge, in the annotator's words.

    ``cls`` is the class of the part it would move to: a reclass (task U21)
    moves a latch's events to a cage, which has no ``open``.
    """
    event = problem.event
    if event is candidate:
        if problem.code == UNKNOWN_VALUE and cls:
            return f"{cls} 没有「{event.new}」这个状态"
        if problem.code == OLD_MISMATCH and problem.held == event.new:
            return f"合并后第 {event.step} 步它已经是 {event.new}，这条多余"
        if problem.code == LEAVES_REMOVED:
            return f"合并后它在第 {event.step} 步之前已拆下"
        return f"合并后第 {event.step} 步它是 {problem.held}，对不上"
    return (f"会让第 {event.step} 步 {event.target} 的 {event.old}→{event.new} 对不上"
            f"（那时是 {problem.held}）")


def _plan_events(db: Db, tax: Taxonomy, plan: MergePlan, instances_post: dict,
                 actions_post: list) -> list[StateEvent]:
    """Decide the fate of every hand-written event of ``gone`` (and ``keep``).

    Returns the merged log the rest of the plan reads states from. Raises
    :class:`MergeError` when the merged log keeps a problem the two logs did not
    have even with every movable event left behind.
    """
    desktop, gone, keep = plan.desktop, plan.gone, plan.keep
    rows = _rows(db, "SELECT * FROM state_event WHERE desktop=? AND auto=0 ORDER BY step, id",
                 desktop)
    instances_pre = db.instances(desktop)
    baseline = Counter(_problem_key(p, {gone: keep}) for p in event_problems(
        instances_pre, events_of(db, tax, desktop), tax))
    log = _Log(instances_post, actions_post, tax, baseline)

    others = [_event(r) for r in rows if r["target"] not in (gone, keep)]
    keep_rows = [r for r in rows if r["target"] == keep]
    gone_rows = [r for r in rows if r["target"] == gone]

    # 1. keep's own events the merge turns into no-ops
    keep_events = [(r, _event(r)) for r in keep_rows]
    problems, extra, _merged = log.check(others + [e for _r, e in keep_events])
    noop = {id(p.event) for p in problems
            if p.event.target == keep and not p.event.auto and p.code == OLD_MISMATCH
            and p.held == p.event.new and extra[_problem_key(p)] > 0}
    kept_keep: list[tuple[dict, StateEvent]] = []
    for row, event in keep_events:
        if id(event) in noop:
            plan.events_dropped.append(
                (row, f"合并后第 {row['step']} 步它已经是 {row['new']}，这条多余"))
        else:
            kept_keep.append((row, event))
    current = others + [e for _r, e in kept_keep]
    _problems, current_extra, _merged = log.check(current)

    # 2. gone's events, one at a time in log order
    held_at = {(r["step"], r["attr"]) for r, _e in kept_keep}
    for row in gone_rows:
        if int(row["step"]) <= INITIAL_STEP:
            plan.events_dropped.append(
                (row, f"初始状态跟着 {keep}（它有自己的起始状态）"))
            continue
        if (row["step"], row["attr"]) in held_at:
            plan.events_dropped.append(
                (row, f"{keep} 在第 {row['step']} 步已有自己的手写事件"))
            continue
        candidate = _event(row, target=keep)
        trial = current + [candidate]
        problems, extra, _merged = log.check(trial)
        worse = extra - current_extra
        if worse:
            hits = [p for p in problems if _problem_key(p) in worse]
            # a state the class does not have says the most (a reclass, U21)
            first = next((p for p in hits if p.code == UNKNOWN_VALUE and p.event is candidate),
                         hits[0])
            plan.events_dropped.append((row, _why(first, candidate, plan.keep_record.cls)))
            continue
        current, current_extra = trial, extra
        plan.events_moved.append(row)
        held_at.add((row["step"], row["attr"]))

    problems, extra, merged = log.check(current)
    if extra:
        first = next(p for p in problems if _problem_key(p) in extra)
        raise MergeError(
            f"合并后状态日志会出错：{first.text}。先在 Steps 把步骤改对{THEN_MERGE}")
    return merged


# -- frames -------------------------------------------------------------- #
def _shape_at(kfs, seg: int, step: int, placement: Optional[str]):
    chain = [kf for kf in kfs if kf.pose_segment == seg
             and (placement is None or kf.placement == placement)]
    return select_keyframe(chain, step)


def _clash_frames(db: Db, tax: Taxonomy, desktop: int, gone: str, keep: str
                  ) -> dict[str, dict[int, tuple[int, Optional[str]]]]:
    """Where both are present **with a shape**: ``view -> {step: (segment, keep's chain)}``.

    In a sequence view both have to be asked for there (their states) and both
    keyframe chains -- each in its own placement -- have to select a shape; in a
    components-gallery view two shapes on one shot are two parts whatever the
    log says (``keep``'s chain is then ``None``: any placement).
    """
    instances = db.instances(desktop)
    events = events_of(db, tax, desktop)
    states: dict[int, dict] = {}
    rois: dict[tuple[str, int], object] = {}
    out: dict[str, dict[int, tuple[int, Optional[str]]]] = {}
    for view in _views(db, desktop, (gone, keep)):
        kf_gone = db.keyframes(desktop, view, gone)
        kf_keep = db.keyframes(desktop, view, keep)
        if not (kf_gone and kf_keep):
            continue
        sequence = view in VIEWS
        for frame in db.frames_for(desktop, view):
            step = int(frame["step"])
            seg = pose_segment_of(db, FrameKey(desktop, step, view))
            if not sequence:
                # no state semantics outside the sequence views: two shapes on
                # one shot are two parts, whatever the log says
                if _shape_at(kf_gone, seg, step, None) and _shape_at(kf_keep, seg, step, None):
                    out.setdefault(view, {})[step] = (seg, None)
                continue
            if (view, seg) not in rois:
                rois[(view, seg)] = db.bench_roi(desktop, view, seg)
            if step not in states:
                states[step] = state_at(instances, events, step, tax)
            pre = states[step]
            needs = needs_geom(instances, pre, tax, bench_roi=rois[(view, seg)])
            if (gone in needs and keep in needs
                    and _shape_at(kf_gone, seg, step, pre[gone].placement)
                    and _shape_at(kf_keep, seg, step, pre[keep].placement)):
                out.setdefault(view, {})[step] = (seg, pre[keep].placement)
    return out


def _overlap_keyframes(db: Db, plan: MergePlan,
                       clashes: dict[str, dict[int, tuple[int, Optional[str]]]]) -> set[int]:
    """``gone``'s keyframes that would still be the shape on a frame ``keep`` keeps.

    A keyframe covers its anchor and the frames back to the one before it, so
    one of ``gone``'s that the merged chain would select on an overlap frame --
    anchored there, or reaching back over it -- goes; until none is selected
    (dropping one can hand the frame to an older one of ``gone``'s).
    """
    dropped: set[int] = set()
    for view, frames in sorted(clashes.items()):
        kf_gone = db.keyframes(plan.desktop, view, plan.gone)
        kf_keep = db.keyframes(plan.desktop, view, plan.keep)
        mine = {int(kf.id) for kf in kf_gone}
        changed = True
        while changed:
            changed = False
            chain = kf_keep + [kf for kf in kf_gone if int(kf.id) not in dropped]
            for step, (seg, placement) in sorted(frames.items()):
                picked = _shape_at(chain, seg, step, placement)
                if picked is not None and int(picked.id) in mine and int(picked.id) not in dropped:
                    dropped.add(int(picked.id))
                    changed = True
    return dropped


def _plan_frames(db: Db, tax: Taxonomy, plan: MergePlan, instances_post: dict,
                 events_post: list, gone_rows: dict, keep_rows: dict) -> None:
    """The two shape checks, the confirmed-frame check and the re-check queue.

    With ``overlap="keep"`` the frames both are drawn on are no refusal: ``gone``'s
    rows there are dropped, so they count as not there.
    """
    desktop, gone, keep = plan.desktop, plan.gone, plan.keep
    instances_pre = db.instances(desktop)
    events_pre = events_of(db, tax, desktop)
    clashes = {view: sorted(steps) for view, steps in
               _clash_frames(db, tax, desktop, gone, keep).items()}
    kept = {(view, int(step)) for view, steps in plan.overlap_frames.items() for step in steps}
    breaks: dict[str, dict[str, list[int]]] = {}
    stale: dict[str, list[int]] = {}
    states_pre: dict[int, dict] = {}
    states_post: dict[int, dict] = {}
    rois: dict[tuple[str, int], object] = {}

    def state(cache, instances, events, step):
        if step not in cache:
            cache[step] = state_at(instances, events, step, tax)
        return cache[step]

    for view in _views(db, desktop, (gone, keep)):
        frozen = set(db.frozen_steps(desktop, view))
        sequence = view in VIEWS
        for frame in db.frames_for(desktop, view):
            step = int(frame["step"])
            key = FrameKey(desktop, step, view)
            seg = pose_segment_of(db, key)
            if (view, seg) not in rois:
                rois[(view, seg)] = db.bench_roi(desktop, view, seg)
            roi = rois[(view, seg)]
            if not sequence:
                if step in frozen and ((step, view) in gone_rows or (step, view) in keep_rows):
                    plan.rechecks.setdefault(view, []).append(step)
                continue
            pre = state(states_pre, instances_pre, events_pre, step)
            needs_pre = needs_geom(instances_pre, pre, tax, bench_roi=roi)
            if step not in frozen:
                continue
            post = state(states_post, instances_post, events_post, step)
            needs_post = needs_geom(instances_post, post, tax, bench_roi=roi)
            flag = frame.get("review_status") == VERIFIED
            g_row, k_row = gone_rows.get((step, view)), keep_rows.get((step, view))
            if (view, step) in kept:
                # keep's stays here; gone's row is dropped -- read as not there
                # after the merge (the check before it still sees both)
                g_pre, g_row = g_row, None
            else:
                g_pre = g_row
            g_ver = bool(g_row and g_row["status"] == VERIFIED)
            g_pre_ver = bool(g_pre and g_pre["status"] == VERIFIED)
            k_ver = bool(k_row and k_row["status"] == VERIFIED)

            def off(has_row: bool, row_verified: bool, needed: bool) -> bool:
                return (flag and has_row != needed) or (row_verified and not needed)

            pre_off = (off(g_pre is not None, g_pre_ver, gone in needs_pre)
                       or off(k_row is not None, k_ver, keep in needs_pre))
            has_keep = g_ver or k_ver or (k_row is not None and g_row is None)
            post_off = off(has_keep, g_ver or k_ver, keep in needs_post)
            if post_off and not pre_off:
                kind = "lost" if has_keep else "gained"
                breaks.setdefault(kind, {}).setdefault(view, []).append(step)
            elif post_off:
                stale.setdefault(view, []).append(step)
            if (g_pre is not None or k_row is not None or keep in needs_post
                    or gone in needs_pre or keep in needs_pre):
                plan.rechecks.setdefault(view, []).append(step)
    if clashes and plan.overlap != OVERLAP_KEEP:
        raise OverlapError(
            f"{gone} 和 {keep} 在同一帧上都在、都有形状（{_frames_text(clashes)}）："
            f"是两个零件，不能合并。真是同一个，先删掉其中一个的形状")
    if breaks:
        said = []
        if breaks.get("lost"):
            said.append(f"{_frames_text(breaks['lost'])}：确认时有它，合并后按步骤表它已不在画面里")
        if breaks.get("gained"):
            said.append(f"{_frames_text(breaks['gained'])}：确认时没有它，合并后按步骤表却要画它")
        raise MergeError(
            f"合并后 {keep} 和这些已确认帧对不上（{'；'.join(said)}）。先在 Steps 把步骤改对"
            f"，例如那一步的动词{THEN_MERGE}")
    if stale:
        plan.warnings.append(
            f"这些已确认帧本来就和步骤表对不上，合并后仍要复核：{_frames_text(stale)}")


# -- layers and edges ---------------------------------------------------- #
def _plan_layers(db: Db, plan: MergePlan) -> None:
    desktop, gone, keep = plan.desktop, plan.gone, plan.keep
    for row in _rows(db, "SELECT * FROM zorder WHERE desktop=? ORDER BY view, pose_segment",
                     desktop):
        try:
            order = R.loads(row["order_json"]) or []
        except (TypeError, ValueError):
            continue
        if not isinstance(order, list) or not any(
                isinstance(p, (list, tuple)) and p and p[0] == gone for p in order):
            continue
        from_gone = {(keep, p[1]) for p in order
                     if isinstance(p, (list, tuple)) and len(p) > 1 and p[0] == gone}
        after: list[list] = []
        dropped: list[tuple[str, str]] = []
        for p in order:
            if isinstance(p, (list, tuple)) and len(p) > 1 and p[0] == keep \
                    and (keep, p[1]) in from_gone:
                dropped.append((keep, str(p[1])))
                continue
            if isinstance(p, (list, tuple)) and p and p[0] == gone:
                after.append([keep, *list(p)[1:]])
            else:
                after.append(list(p) if isinstance(p, (list, tuple)) else p)
        plan.zorder.append({"view": row["view"], "seg": int(row["pose_segment"]),
                            "before": row["order_json"], "version": row["version"],
                            "after": after, "dropped": dropped})

    pairs = _rows(db, "SELECT * FROM pair_override WHERE desktop=? "
                      "ORDER BY view, pose_segment, above, below", desktop)
    standing = {(r["view"], r["pose_segment"], r["above"], r["below"])
                for r in pairs if gone not in (r["above"], r["below"])}
    contradicted: set[tuple] = set()
    for row in pairs:
        if gone not in (row["above"], row["below"]):
            continue
        above = keep if row["above"] == gone else row["above"]
        below = keep if row["below"] == gone else row["below"]
        triple = (row["view"], row["pose_segment"], above, below)
        if above == below:
            plan.pairs_dropped.append((row, f"合并后是 {keep} 和它自己"))
            continue
        if triple in standing:
            plan.pairs_dropped.append((row, "合并后与已有的一条重复"))
            continue
        reverse = (row["view"], row["pose_segment"], below, above)
        if reverse in standing and reverse not in contradicted:
            contradicted.add(reverse)
        plan.pairs_renamed.append((row, {**row, "above": above, "below": below}))
        standing.add(triple)
    for row in pairs:
        triple = (row["view"], row["pose_segment"], row["above"], row["below"])
        if triple in contradicted:
            plan.pairs_dropped.append((row, f"与 {plan.gone} 的那条相反，保留 {plan.gone} 的"))

    relations = _rows(db, "SELECT * FROM relation WHERE desktop=? ORDER BY id", desktop)
    # triple -> the standing row (None once a renamed row of gone holds it)
    taken: dict[tuple, Optional[dict]] = {
        (r["type"], r["target"], r["blocker"]): r for r in relations
        if gone not in (r["target"], r["blocker"])}
    for row in relations:
        if gone not in (row["target"], row["blocker"]):
            continue
        target = keep if row["target"] == gone else row["target"]
        blocker = keep if row["blocker"] == gone else row["blocker"]
        triple = (row["type"], target, blocker)
        if target == blocker:
            plan.relations_dropped.append((row, f"合并后是 {keep} 和它自己"))
            continue
        if triple in taken:
            other = taken[triple]
            if (other is not None and other["source"] == RULE
                    and row["source"] != RULE):
                # the same edge twice: a human's (gone's) beats a derived one
                # (round 3) -- its necessity, mode and reason are a decision
                plan.relations_dropped.append(
                    (other, f"与 {gone} 的{row['source']}边重复，保留那条"))
            else:
                plan.relations_dropped.append((row, "合并后与已有的一条重复"))
                continue
        plan.relations_renamed.append((row, target, blocker))
        taken[triple] = None


def _deadlocks(db: Db, plan: MergePlan, instances_post: dict, tax: Taxonomy) -> None:
    """Refuse a merge that makes the constraint graph deadlock where it did not."""
    from tda.core.graph import edges_from_db
    from tda.core.graph_derive import settled_instances
    from tda.core.graph_plan import find_deadlocks
    from tda.core.graph_rules import HARD_TYPES

    if not (plan.relations_renamed or plan.relations_dropped):
        return
    edges = [e for e in edges_from_db(db, plan.desktop) if e.type in HARD_TYPES]
    manual = [e for e in db.events(plan.desktop) if not e.auto]
    before = find_deadlocks(edges, settled_instances(db.instances(plan.desktop)), tax,
                            initial=initial_overrides(manual))
    renamed = {(r["type"], r["target"], r["blocker"]): (t, b)
               for r, t, b in plan.relations_renamed}
    dropped = {(r["type"], r["target"], r["blocker"]) for r, _why in plan.relations_dropped}
    after_edges = []
    for edge in edges:
        triple = (edge.type, edge.target, edge.blocker)
        if triple in dropped:
            continue
        if triple in renamed:
            target, blocker = renamed[triple]
            edge = replace(edge, target=target, blocker=blocker)
        after_edges.append(edge)
    keep_manual = [e for e in manual if e.target != plan.gone]
    after = find_deadlocks(after_edges, settled_instances(instances_post), tax,
                           initial=initial_overrides(keep_manual))
    if len(after) > len(before):
        raise MergeError(f"合并后约束图会死锁：{after[0].label()}。先在 Relations 页改那条边")


# -- the plan ------------------------------------------------------------ #
def plan_merge(db: Db, tax: Taxonomy, desktop: int, gone: str, keep: str,
               reclass: bool = False, overlap: str = OVERLAP_REFUSE) -> MergePlan:
    """Read everything a merge of ``gone`` into ``keep`` would write; nothing is.

    Raises :class:`MergeError` with the reason when the merge is refused --
    :class:`OverlapError` when only because both are drawn (or confirmed, or
    overridden, or flagged) on the same frames. ``reclass`` lifts the class
    check for one caller: 「其实是硬盘架」 / 「其实是卡扣」 (:mod:`tda.core.reclass`,
    task U21) merges a part into a fresh instance of the other class, which it
    has just written.

    ``overlap="keep"`` (task U21 round 2) merges those anyway: on the frames
    both are drawn on, both carry a confirmed row, an override or a flag,
    ``keep``'s stays and ``gone``'s compiled rows, overrides, flags and
    conflicts there are dropped, and so is every keyframe of ``gone``'s the
    merged chain would still pick on a frame both are drawn on; ``gone``'s
    shapes elsewhere move as always. Everything dropped is in the inverse.
    """
    if overlap not in (OVERLAP_REFUSE, OVERLAP_KEEP):
        raise ValueError(f"overlap must be {OVERLAP_REFUSE!r} or {OVERLAP_KEEP!r}, got {overlap!r}")
    desktop = int(desktop)
    gone, keep = str(gone).strip(), str(keep).strip()
    if gone == keep:
        raise MergeError(f"不能把 {gone} 合并到它自己")
    instances = db.instances(desktop)
    for key in (gone, keep):
        if key not in instances:
            raise MergeError(f"D{desktop:02d} 没有实例 {key}")
        if is_provisional(key):
            raise MergeError(f"{key} 是 Label Studio 草稿：草稿要采纳（Shift+A），不是合并")
        if is_context(instances[key]):
            # task U18: not a part of the machine; it is deleted, not merged
            raise MergeError(f"{key} 是框外零件：不能合并（不要了就在实例表右键删除）")
    # task U34 round 2 (N8): a part said absent from the start was never in the
    # machine -- neither half of a merge until it is taken back
    absent = absent_keys(e for e in db.events(desktop) if not e.auto)
    for key in (gone, keep):
        if key in absent:
            raise MergeError(f"{key} 标着「一开始就不在」（这台没有它）：先在 Steps → Instances "
                             f"右键「其实有…」，再合并")
    g_rec, k_rec = instances[gone], instances[keep]
    if g_rec.cls != k_rec.cls and not reclass:
        raise MergeError(f"{gone} 是 {g_rec.cls}，{keep} 是 {k_rec.cls}：只有同一类的两个实例"
                         f"才能合并")
    plan = MergePlan(desktop=desktop, gone=gone, keep=keep, cls=k_rec.cls,
                     gone_record=copy.deepcopy(g_rec), keep_record=copy.deepcopy(k_rec),
                     overlap=overlap)

    # actions: never two of the merged part at one step
    actions = db.actions(desktop)
    gone_steps = {a.step for a in actions if a.target == gone}
    both = sorted(gone_steps & {a.step for a in actions if a.target == keep})
    if both:
        raise MergeError(
            f"第 {ranges(both)} 步 {gone} 和 {keep} 都有动作：合并后一步里会有同一个零件的两个"
            f"动作。先在 Steps 删掉或改掉其中一个{THEN_MERGE}")
    plan.actions = [(a.step, a.idx) for a in actions if a.target == gone]
    actions_post = [copy.copy(a) for a in actions]
    for action in actions_post:
        if action.target == gone:
            action.target = keep

    instances_post, before, after = _repoint(instances, gone, keep)
    plan.neighbours_before, plan.neighbours_after = before, after
    loop = _parent_loop(instances_post)
    if loop and not _parent_loop(instances):
        raise MergeError(f"合并后 parent 会成环：{loop}。先在实例表改 parent")

    # rows keyed by frame
    def by_frame(table: str, key: str) -> dict:
        return {(int(r["step"]), str(r["view"])): r for r in _rows(
            db, f"SELECT * FROM {table} WHERE desktop=? AND instance=?", desktop, key)}

    g_rows, k_rows = by_frame("compiled_mask", gone), by_frame("compiled_mask", keep)
    # overlap="keep" (U21 round 2): the frames both are drawn or confirmed on
    # keep keep's; gone's rows, overrides, flags, conflicts and the keyframes
    # that would still be the shape there go, recorded for the undo
    kept: set[tuple[str, int]] = set()
    clashes: dict[str, dict[int, tuple[int, Optional[str]]]] = {}
    if plan.overlap == OVERLAP_KEEP:
        clashes = _clash_frames(db, tax, desktop, gone, keep)
        kept = {(view, step) for view, steps in clashes.items() for step in steps}
        kept |= {(view, step) for (step, view), row in g_rows.items()
                 if row["status"] == VERIFIED
                 and (k_rows.get((step, view)) or {}).get("status") == VERIFIED}
        for table in ("frame_override", "instance_frame_flags"):
            kept |= {(view, step) for step, view in
                     set(by_frame(table, gone)) & set(by_frame(table, keep))}
        by_view: dict[str, set[int]] = {}
        for view, step in kept:
            by_view.setdefault(view, set()).add(step)
        plan.overlap_frames = {view: sorted(steps) for view, steps in sorted(by_view.items())}
        plan.overlap_shapes = {view: sorted(steps) for view, steps in sorted(clashes.items())}
    both_verified = {}
    for (step, view), row in sorted(g_rows.items()):
        other = k_rows.get((step, view))
        if (view, step) in kept:
            plan.compiled_dropped.append(row)        # keep's row stays as it is
            plan.verified_dropped += int(row["status"] == VERIFIED)
            continue
        if row["status"] == VERIFIED:
            if other is not None and other["status"] == VERIFIED:
                both_verified.setdefault(view, []).append(step)
                continue
            plan.compiled_renamed.append((step, view))
            if other is not None:
                plan.compiled_dropped.append(other)
        else:
            plan.compiled_dropped.append(row)
            if other is not None and other["status"] != VERIFIED:
                plan.compiled_dropped.append(other)
    if both_verified:
        raise OverlapError(f"{gone} 和 {keep} 在同一帧上各有一行已确认的结果"
                           f"（{_frames_text(both_verified)}）：是两个零件，不能合并")
    for table, label, target, gone_rows in (
            ("frame_override", "单帧覆盖", plan.overrides, plan.overrides_dropped),
            ("instance_frame_flags", "逐帧标记", plan.flags, plan.flags_dropped)):
        mine, theirs = by_frame(table, gone), by_frame(table, keep)
        clash: dict[str, list[int]] = {}
        for step, view in sorted(set(mine) & set(theirs)):
            clash.setdefault(view, []).append(step)
        if clash and plan.overlap != OVERLAP_KEEP:
            raise OverlapError(f"{gone} 和 {keep} 在同一帧上都有{label}（{_frames_text(clash)}）："
                               f"是两个零件，不能合并")
        for step, view in sorted(mine):
            if (view, step) in kept:
                gone_rows.append(mine[(step, view)])
            else:
                target.append((step, view))
    for row in _rows(db, "SELECT * FROM conflict WHERE desktop=? AND instance=? ORDER BY id",
                     desktop, gone):
        if (str(row["view"]), int(row["step"])) in kept:
            plan.conflicts_dropped.append(row)
        else:
            plan.conflicts.append(int(row["id"]))
    dropped_kfs = _overlap_keyframes(db, plan, clashes) if clashes else set()
    for kf_id in sorted(dropped_kfs):
        plan.keyframes_dropped.extend(_rows(db, "SELECT * FROM shape_keyframe WHERE id=?", kf_id))
        plan.parts_dropped.extend(_rows(db, "SELECT * FROM shape_part WHERE keyframe_id=? "
                                            "ORDER BY idx", kf_id))
    for view in _views(db, desktop, (gone,)):
        kfs = [kf for kf in db.keyframes(desktop, view, gone) if int(kf.id) not in dropped_kfs]
        if kfs:
            plan.keyframes[view] = sorted(((int(kf.id), int(kf.anchor_step)) for kf in kfs),
                                          key=lambda item: (item[1], item[0]))

    events_post = _plan_events(db, tax, plan, instances_post, actions_post)
    _plan_frames(db, tax, plan, instances_post, events_post, g_rows, k_rows)
    _plan_layers(db, plan)
    _deadlocks(db, plan, instances_post, tax)

    reached = {(view, step) for step, view in g_rows}
    reached |= {(row["view"], int(row["step"])) for row in plan.compiled_dropped}
    reached |= {(view, step) for step, view in plan.overrides + plan.flags}
    reached |= {(r["view"], int(r["step"])) for r in _rows(
        db, "SELECT view, step FROM conflict WHERE desktop=? AND instance=?", desktop, gone)}
    reached |= kept
    reached |= {(str(r["view"]), int(r["step"]))
                for r in plan.overrides_dropped + plan.flags_dropped}
    plan.frames = sorted(reached)
    by_view: dict[str, set[int]] = {}
    for view, step in plan.frames:
        by_view.setdefault(view, set()).add(step)
    for view, steps in by_view.items():
        plan.rechecks.setdefault(view, []).extend(db.frozen_steps_in(desktop, view, steps))
    plan.rechecks = {v: sorted(set(s)) for v, s in sorted(plan.rechecks.items()) if s}
    return plan


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
@dataclass
class MergeReport:
    """What one merge did: the plan's counts plus the op-log row."""

    op_id: int
    plan: MergePlan

    @property
    def moved(self) -> dict[str, int]:
        return self.plan.moved()

    @property
    def dropped(self) -> dict[str, int]:
        return self.plan.dropped()

    def status_line(self) -> str:
        """The panel's status line after the merge."""
        plan = self.plan
        kfs = sum(len(v) for v in plan.keyframes.values())
        parts = [f"形状 {kfs}"] if kfs else []
        if plan.compiled_renamed:
            parts.append(f"已确认的行 {len(plan.compiled_renamed)}")
        if plan.actions:
            parts.append(f"动作 {len(plan.actions)}")
        if plan.events_moved:
            parts.append(f"手写事件 {len(plan.events_moved)}")
        moved = "、".join(parts) or "没有形状和动作"
        dropped = f"；丢掉手写事件 {len(plan.events_dropped)} 条" if plan.events_dropped else ""
        if plan.overlap_frames:
            lost = [f"形状 {len(plan.keyframes_dropped)}"] if plan.keyframes_dropped else []
            if plan.verified_dropped:
                lost.append(f"已确认的行 {plan.verified_dropped}")
            dropped += (f"；重叠处（{_frames_text(plan.overlap_frames)}）保留 {plan.keep} 的"
                        + (f"，丢掉 {plan.gone} 的{'、'.join(lost)}" if lost else ""))
        queued = sum(len(v) for v in plan.rechecks.values())
        rechecks = f"；{queued} 个已确认帧在后台重新核对" if queued else ""
        return (f"已把 {plan.gone} 合并到 {plan.keep}（{moved}{dropped}{rechecks}）"
                f" / merged {plan.gone} into {plan.keep}")


def _payload(plan: MergePlan) -> dict:
    return {
        "gone": plan.gone, "keep": plan.keep, "cls": plan.cls,
        "gone_record": asdict(plan.gone_record),
        "moved": plan.moved(), "dropped": plan.dropped(),
        "keyframes": {view: [a for _i, a in kfs] for view, kfs in plan.keyframes.items()},
        "actions": [list(a) for a in plan.actions],
        "dropped_events": [{"id": r["id"], "step": r["step"], "target": r["target"],
                            "attr": r["attr"], "old": r["old"], "new": r["new"], "why": why}
                           for r, why in plan.events_dropped],
        "zorder_dropped": [{"view": z["view"], "seg": z["seg"], "entries": z["dropped"]}
                           for z in plan.zorder if z["dropped"]],
        "warnings": list(plan.warnings),
        "frames": len(plan.frames),
        "rechecks": {view: steps for view, steps in plan.rechecks.items()},
        "overlap": plan.overlap,
        "overlap_frames": {view: steps for view, steps in plan.overlap_frames.items()},
        "keyframes_dropped": [[r["view"], int(r["anchor_step"])] for r in plan.keyframes_dropped],
        "verified_dropped": plan.verified_dropped,
    }


def _inverse(plan: MergePlan) -> dict:
    return {
        "gone": plan.gone, "keep": plan.keep,
        "gone_record": asdict(plan.gone_record),
        "neighbours": [asdict(rec) for rec in plan.neighbours_before],
        "actions": [list(a) for a in plan.actions],
        "keyframes": sorted(i for kfs in plan.keyframes.values() for i, _a in kfs),
        "compiled_renamed": [[s, v] for s, v in plan.compiled_renamed],
        "compiled_dropped": list(plan.compiled_dropped),
        "overrides": [[s, v] for s, v in plan.overrides],
        "flags": [[s, v] for s, v in plan.flags],
        "conflicts": list(plan.conflicts),
        "zorder": [{"view": z["view"], "seg": z["seg"], "order_json": z["before"],
                    "version": z["version"]} for z in plan.zorder],
        "pairs_renamed": [{"old": old, "new": new} for old, new in plan.pairs_renamed],
        "pairs_dropped": [row for row, _why in plan.pairs_dropped],
        "relations_renamed": [{"id": r["id"], "target": r["target"], "blocker": r["blocker"]}
                              for r, _t, _b in plan.relations_renamed],
        "relations_dropped": [row for row, _why in plan.relations_dropped],
        "events_moved": [int(r["id"]) for r in plan.events_moved],
        "events_dropped": [row for row, _why in plan.events_dropped],
        "frames": [[v, s] for v, s in plan.frames],
        # overlap="keep" (U21 round 2): what of gone's went where keep's stays
        "overlap": plan.overlap,
        "keyframes_dropped": list(plan.keyframes_dropped),
        "parts_dropped": list(plan.parts_dropped),
        "overrides_dropped": list(plan.overrides_dropped),
        "flags_dropped": list(plan.flags_dropped),
        "conflicts_dropped": list(plan.conflicts_dropped),
    }


def _settle(db: Db, tax: Taxonomy, desktop: int, frames, rechecks: dict) -> None:
    """Recompile the derived log, re-stamp the graph, clear digests, queue re-checks."""
    from tda.core.extra import recompile_events, restamp_graph

    recompile_events(db, desktop, tax)
    restamp_graph(db, desktop, tax)
    for view, step in frames:
        db.clear_frame_digest(FrameKey(desktop, int(step), str(view)))
    for view, steps in rechecks.items():
        db.add_rechecks(desktop, view, steps)


def apply_merge(db: Db, tax: Taxonomy, plan: MergePlan, annotator: str = "") -> MergeReport:
    """Write ``plan`` in one transaction with one ``op_log`` row; returns the report."""
    with db.transaction():
        write_merge(db, plan)
        _settle(db, tax, plan.desktop, plan.frames, plan.rechecks)
        inverse = _inverse(plan)
        # what the frames an undo would rewrite were confirmed with right
        # after the merge: an undo finding more refuses (round 3)
        inverse["confirmed"] = _confirmed(db, plan.desktop, _undo_frames(inverse))
        op_id = db.log_op(plan.desktop, OP_VIEW, OP_MERGE, _payload(plan), inverse,
                          annotator or "ui:steps")
    return MergeReport(op_id=op_id, plan=plan)


def write_merge(db: Db, plan: MergePlan) -> None:
    """The rows of ``plan``, ``gone``'s record deleted; **joins the caller's transaction**.

    No settling and no op row: :func:`apply_merge` adds those, and so does the
    reclass (:mod:`tda.core.reclass`, task U21), which writes its cage patch in
    between and logs one op of its own.
    """
    d, gone, keep = plan.desktop, plan.gone, plan.keep
    c = db.conn
    with db.transaction():
        for rec in plan.neighbours_after:
            db.upsert_instance(rec)
        c.execute("UPDATE action SET target=? WHERE desktop=? AND target=?", (keep, d, gone))
        # overlap="keep": gone's that keep's replaces go first (U21 round 2)
        for row in plan.keyframes_dropped:
            c.execute("DELETE FROM shape_part WHERE keyframe_id=?", (row["id"],))
            c.execute("DELETE FROM shape_keyframe WHERE id=?", (row["id"],))
        for table, rows in (("frame_override", plan.overrides_dropped),
                            ("instance_frame_flags", plan.flags_dropped)):
            for row in rows:
                c.execute(f"DELETE FROM {table} WHERE desktop=? AND step=? AND view=? "
                          f"AND instance=?", (d, row["step"], row["view"], gone))
        for row in plan.conflicts_dropped:
            c.execute("DELETE FROM conflict WHERE id=?", (row["id"],))
        c.execute("UPDATE shape_keyframe SET instance=? WHERE desktop=? AND instance=?",
                  (keep, d, gone))
        for row in plan.compiled_dropped:
            c.execute("DELETE FROM compiled_mask WHERE desktop=? AND step=? AND view=? "
                      "AND instance=?", (d, row["step"], row["view"], row["instance"]))
        c.execute("UPDATE compiled_mask SET instance=? WHERE desktop=? AND instance=?",
                  (keep, d, gone))
        for table in ("frame_override", "instance_frame_flags", "conflict"):
            c.execute(f"UPDATE {table} SET instance=? WHERE desktop=? AND instance=?",
                      (keep, d, gone))
        for z in plan.zorder:
            c.execute("UPDATE zorder SET order_json=? WHERE desktop=? AND view=? "
                      "AND pose_segment=?", (R.dumps(z["after"]), d, z["view"], z["seg"]))
        for row in [old for old, _new in plan.pairs_renamed] + \
                [row for row, _why in plan.pairs_dropped]:
            c.execute("DELETE FROM pair_override WHERE desktop=? AND view=? AND pose_segment=? "
                      "AND above=? AND below=?",
                      (d, row["view"], row["pose_segment"], row["above"], row["below"]))
        for _old, new in plan.pairs_renamed:
            _insert(db, "pair_override", new)
        for row, _why in plan.relations_dropped:
            c.execute("DELETE FROM relation WHERE id=?", (row["id"],))
        for row, target, blocker in plan.relations_renamed:
            c.execute("UPDATE relation SET target=?, blocker=? WHERE id=?",
                      (target, blocker, row["id"]))
        for row, _why in plan.events_dropped:
            c.execute("DELETE FROM state_event WHERE id=?", (row["id"],))
        for row in plan.events_moved:
            c.execute("UPDATE state_event SET target=? WHERE id=?", (keep, row["id"]))
        c.execute("DELETE FROM state_event WHERE desktop=? AND target=? AND auto=1", (d, gone))
        c.execute('DELETE FROM instance WHERE desktop=? AND "key"=?', (d, gone))


def _undo_frames(inverse: dict) -> list[tuple[str, int]]:
    """``(view, step)`` of every frame an undo rewrites rows of."""
    frames = {(str(v), int(s)) for v, s in inverse.get("frames") or []}
    frames |= {(str(v), int(s)) for s, v in inverse.get("compiled_renamed") or []}
    frames |= {(str(r["view"]), int(r["step"])) for r in inverse.get("compiled_dropped") or []}
    return sorted(frames)


def _confirmed(db: Db, desktop: int, frames) -> list[list]:
    """``[view, step, review_status, [[instance, verified_at], ...]]`` per frame."""
    out: list[list] = []
    for view, step in frames:
        row = db.get_frame(FrameKey(desktop, step, view)) or {}
        signed = sorted([str(r["instance"]), r["verified_at"]] for r in db.conn.execute(
            "SELECT instance, verified_at FROM compiled_mask WHERE desktop=? AND step=? "
            "AND view=? AND status=?", (desktop, step, view, VERIFIED)))
        out.append([view, step, row.get("review_status"), signed])
    return out


def _confirmed_since(db: Db, desktop: int, inverse: dict) -> dict[str, list[int]]:
    """The frames an undo would rewrite that a human confirmed after the merge.

    A frame whose flag went to ``verified``, or that holds a signed row the
    merge did not leave there (a re-trace accepted with ``N``, a ``Space``).
    An undo would put the pre-merge rows back over that signature.
    """
    since: dict[str, list[int]] = {}
    for view, step, review, signed in inverse.get("confirmed") or []:
        now = _confirmed(db, desktop, [(view, step)])[0]
        then = {tuple(item) for item in signed}
        if ((now[2] == VERIFIED and review != VERIFIED)
                or any(tuple(item) not in then for item in now[3])):
            since.setdefault(str(view), []).append(int(step))
    return since


def merge_instances(db: Db, tax: Taxonomy, desktop: int, gone: str, keep: str,
                    annotator: str = "", overlap: str = OVERLAP_REFUSE) -> MergeReport:
    """Fold ``gone`` into ``keep`` (same class), atomically; see the module docstring.

    ``overlap="keep"``: where both are drawn or confirmed, ``keep``'s stays
    (:func:`plan_merge`); the default refuses there, as U12 always did.
    """
    return apply_merge(db, tax, plan_merge(db, tax, desktop, gone, keep, overlap=overlap),
                       annotator)


# --------------------------------------------------------------------------- #
# undo / redo
# --------------------------------------------------------------------------- #
def undo_merge(db: Db, tax: Taxonomy, op: dict, annotator: str = "") -> int:
    """Apply the inverse of one :data:`OP_MERGE` row; returns the undo's op id.

    ``op`` is a row as :meth:`tda.core.db.Db.get_op` returns it. ``gone`` comes
    back with its record, its rows, its events and every pointer and edge
    that named it; the derived log is recompiled, the graph re-stamped and
    every frozen frame of the views involved queued again. Refused while
    ``gone`` exists again, and once any frame the undo would rewrite has been
    confirmed since the merge (round 3: the restored rows would replace a
    signature given to the merged part).
    """
    if str(op.get("kind")) != OP_MERGE:
        raise MergeError(f"op {op.get('id')} 不是一次合并")
    inv = op.get("inverse") or {}
    d = int(op["desktop"])
    gone, keep = str(inv["gone"]), str(inv["keep"])
    stored = db.instances(d)
    if gone in stored:
        raise MergeError(f"撤销不了：{gone} 又存在了")
    if keep not in stored:
        raise MergeError(f"撤销不了：{keep} 已经不在了")
    since = _confirmed_since(db, d, inv)
    if since:
        raise MergeError(f"撤销不了：合并后又确认过这些帧（{_frames_text(since)}），撤销会把"
                         f"合并前的行写回去、盖掉那次确认")
    try:
        with db.transaction():
            frames = undo_rows(db, tax, d, inv)
            # every frozen frame the restored instance reaches, in every view
            # it or keep is drawn or compiled in -- not just the merge-time
            # list: frames were confirmed and demoted since (round 3)
            views = {view for view, _step in frames}
            views |= set((op.get("payload") or {}).get("rechecks") or {})
            views |= {str(r["view"]) for r in db.conn.execute(
                "SELECT DISTINCT view FROM shape_keyframe WHERE desktop=? AND instance IN (?, ?)",
                (d, gone, keep))}
            for view in sorted(views):
                db.queue_rechecks_for_view(d, view)
            return db.log_op(d, OP_VIEW, OP_UNDO,
                             {"undid": int(op["id"]), "gone": gone, "keep": keep},
                             {"redo": {"gone": gone, "keep": keep,
                                       "overlap": inv.get("overlap") or OVERLAP_REFUSE}},
                             annotator or str(op.get("annotator") or ""))
    except MergeError:
        raise
    except Exception as error:  # the transaction rolled back
        raise MergeError(f"撤销不了：{type(error).__name__}: {error}") from error


def undo_rows(db: Db, tax: Taxonomy, d: int, inv: dict) -> list[tuple[str, int]]:
    """Put back every row a merge's ``inv`` names; **joins the caller's transaction**.

    ``gone`` comes back with its record, its rows and events and every
    pointer and edge that named it; the derived log is recompiled, the graph
    re-stamped and the digests of the frames involved cleared. Returns those
    frames. ``keep``'s own record is left alone -- the reclass undo deletes it
    (:mod:`tda.core.reclass`, task U21).
    """
    gone, keep = str(inv["gone"]), str(inv["keep"])
    c = db.conn
    with db.transaction():
        db.upsert_instance(InstanceRec(**inv["gone_record"]))
        for data in inv.get("neighbours") or []:
            db.upsert_instance(InstanceRec(**data))
        for step, idx in inv.get("actions") or []:
            c.execute("UPDATE action SET target=? WHERE desktop=? AND step=? AND idx=? "
                      "AND target=?", (gone, d, step, idx, keep))
        for kf_id in inv.get("keyframes") or []:
            c.execute("UPDATE shape_keyframe SET instance=? WHERE id=? AND instance=?",
                      (gone, kf_id, keep))
        for step, view in inv.get("compiled_renamed") or []:
            c.execute("UPDATE compiled_mask SET instance=? WHERE desktop=? AND step=? "
                      "AND view=? AND instance=?", (gone, d, step, view, keep))
        for row in inv.get("compiled_dropped") or []:
            _insert(db, "compiled_mask", row, replace=True)
        # overlap="keep" (U21 round 2): gone's that keep's replaced come back
        for row in inv.get("keyframes_dropped") or []:
            _insert(db, "shape_keyframe", row)
        for row in inv.get("parts_dropped") or []:
            _insert(db, "shape_part", row)
        for table, key in (("frame_override", "overrides_dropped"),
                           ("instance_frame_flags", "flags_dropped"),
                           ("conflict", "conflicts_dropped")):
            for row in inv.get(key) or []:
                _insert(db, table, row)
        for table, key in (("frame_override", "overrides"),
                           ("instance_frame_flags", "flags")):
            for step, view in inv.get(key) or []:
                c.execute(f"UPDATE {table} SET instance=? WHERE desktop=? AND step=? "
                          f"AND view=? AND instance=?", (gone, d, step, view, keep))
        for cid in inv.get("conflicts") or []:
            c.execute("UPDATE conflict SET instance=? WHERE id=?", (gone, cid))
        for z in inv.get("zorder") or []:
            c.execute("UPDATE zorder SET order_json=?, version=? WHERE desktop=? AND view=? "
                      "AND pose_segment=?",
                      (z["order_json"], z["version"], d, z["view"], z["seg"]))
        for pair in inv.get("pairs_renamed") or []:
            new = pair["new"]
            c.execute("DELETE FROM pair_override WHERE desktop=? AND view=? "
                      "AND pose_segment=? AND above=? AND below=?",
                      (d, new["view"], new["pose_segment"], new["above"], new["below"]))
        for row in [p["old"] for p in inv.get("pairs_renamed") or []] + \
                list(inv.get("pairs_dropped") or []):
            _insert(db, "pair_override", row, replace=True)
        for rel in inv.get("relations_renamed") or []:
            c.execute("UPDATE relation SET target=?, blocker=? WHERE id=?",
                      (rel["target"], rel["blocker"], rel["id"]))
        for row in inv.get("relations_dropped") or []:
            _insert(db, "relation", row)
        ids = [int(i) for i in inv.get("events_moved") or []]
        for event_id in ids:
            c.execute("UPDATE state_event SET target=? WHERE id=?", (gone, event_id))
        for row in inv.get("events_dropped") or []:
            _insert(db, "state_event", row)
        frames = _undo_frames(inv)
        _settle(db, tax, d, frames, {})
    return frames


def redo_merge(db: Db, tax: Taxonomy, op: dict, annotator: str = "") -> MergeReport:
    """Run the merge an :data:`OP_MERGE` or :data:`OP_UNDO` row names once more.

    The preconditions are checked again: whatever happened since is what the
    merge has to fit now.
    """
    payload = op.get("payload") or {}
    if str(op.get("kind")) == OP_UNDO:
        names = (op.get("inverse") or {}).get("redo") or payload
    elif str(op.get("kind")) == OP_MERGE:
        names = payload
    else:
        raise MergeError(f"op {op.get('id')} 不是一次合并")
    return merge_instances(db, tax, int(op["desktop"]), str(names["gone"]),
                           str(names["keep"]), annotator or str(op.get("annotator") or ""),
                           overlap=str(names.get("overlap") or OVERLAP_REFUSE))
