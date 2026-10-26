"""「合并到…」 in stage S1: the Qt-free half (task U12).

The importer sometimes makes two instances of one part (D2: ``cover.01``
"Screw Cover" and ``cover.02`` "Front Cover", one cover). Steps mode merges
them with :mod:`tda.core.merge_instances`, and like
:func:`tda.ui.steps_delete.delete_instance` the merge is **written through**:
it moves shapes, confirmed rows and events that ``Apply`` knows nothing about,
so it cannot wait for it. A plain merge is therefore refused while the panel
holds anything not applied yet (:func:`staged_edits`): the panel reloads from
the database afterwards, and an edit made a minute ago would be lost without a
word.

**「Apply 并合并」** (round 2) is the way to merge with edits staged: the table
is saved and the merge planned and written in **one** transaction
(``with_staged=True``). The D2 case needs it: the verb at step 38 has to be
``open`` before the merge fits, and an ``Apply`` on its own reopens the session,
whose background re-check finds ``cover.01`` open without a shape on frames
38-48 and demotes all eleven -- the merge a minute later finds nothing to undo
that. In one transaction nobody ever sees that state. The save runs on a
**copy** of the panel's table, so a refusal anywhere -- the save, the plan, the
write -- rolls the database back and leaves the panel holding its edits as they
were.

This module is what the panel calls: the choices offered, the refusal for
staged edits, the plan for the confirmation and the merge itself. Refusals come
back as :class:`~tda.ui.steps_values.EditError`, like every other S1 refusal --
:class:`MergeOverlap` when the plain merge is refused only because both are
drawn or confirmed on the same frames: the confirmation then offers
「合并，重叠处保留 <keep>」 (``overlap="keep"``, task U21 round 2).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tda.core.db import Db
from tda.core.merge_instances import (
    OVERLAP_KEEP,
    OVERLAP_REFUSE,
    MergeError,
    MergePlan,
    MergeReport,
    OverlapError,
    apply_merge,
    plan_merge,
)
from tda.core.model import is_provisional
from tda.ui.steps_values import EditError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tda.ui.steps_model import StepTableData

__all__ = ["APPLY_AND_MERGE", "KEEP_BUTTON", "MENU_TEXT", "OVERLAP_KEEP", "STAGED",
           "AppliedMerge", "MergeOverlap", "choice_label", "merge", "merge_choices",
           "overlap_question", "plan", "staged_edits"]

#: The second button of the confirmation when the plain merge is refused only
#: because both are drawn or confirmed on the same frames (task U21 round 2).
KEEP_BUTTON = "合并，重叠处保留 {keep}"


class MergeOverlap(EditError):
    """The plain merge is refused only for overlap; ``overlap="keep"`` would take it."""


def overlap_question(planned: MergePlan) -> str:
    """The first sentence of the overlap confirmation (U21 review, M4).

    Not U12's 「…是两个零件，不能合并」 followed by a merge button: it says
    what is usually true, and what merging anyway keeps and drops. A merge is
    undone from the op log, not with ``Ctrl+Z`` (Steps writes it through).
    """
    from tda.core.merge_instances import _frames_text

    shapes = bool(planned.overlap_shapes)
    frames = _frames_text(planned.overlap_frames)
    what = "都有形状" if shapes else "都有已确认的行（或单帧覆盖、逐帧标记）"
    kept = "形状和确认行" if shapes else "确认行"
    return (f"两者在 {frames}{what}：通常是两个零件。确定是同一个的话，可以合并并在重叠处保留 "
            f"{planned.keep} 的{kept}（{planned.gone} 在这些帧上的会被丢弃，原样记在操作日志里，"
            f"能整个撤回，但不是 Ctrl+Z）")


def _refusal(error: MergeError) -> EditError:
    """A core refusal as the panel's: an overlap one says so by its type."""
    return MergeOverlap(str(error)) if isinstance(error, OverlapError) else EditError(str(error))

#: The context-menu entry on an instance row.
MENU_TEXT = "合并到… / Merge into…"
#: The refusal of a plain merge while the panel holds unapplied edits.
STAGED = ("先 Apply 或 Revert：Steps 里还有没写入的修改（{what}）。合并会立即写入并重新读入这张表 / "
          "apply or revert the staged edits first")
#: The first line of the confirmation when edits are staged (round 2).
APPLY_AND_MERGE = ("「Apply 并合并」：Steps 里还有没 Apply 的修改（{what}）。选 Yes 会先 Apply 这张表、"
                   "再合并，一次写入；任何一步被拒绝都什么也不写，表里的修改原样留着。")


def merge_choices(data: "StepTableData", key: str) -> list[str]:
    """The instances ``key`` can be merged into: same class, not a draft, not itself."""
    rec = data.instances.get(key)
    if rec is None or is_provisional(key):
        return []
    return sorted(other for other, inst in data.instances.items()
                  if other != key and inst.cls == rec.cls and not is_provisional(other))


def choice_label(data: "StepTableData", key: str) -> str:
    """``cover.01 — Screw Cover（of=motherboard_screws；第 38 步 open）`` for the picker.

    The ``of`` is shown (round 3) because the one kept decides it -- and the
    rule edges it derives -- for the merged part.
    """
    rec = data.instances[key]
    names = "、".join(rec.raw_names) if rec.raw_names else "日志里没有"
    acts = [f"第 {a.step} 步 {a.verb}" for a in data.actions if a.target == key]
    of = (rec.attrs or {}).get("of")
    head = f"of={of}；" if of else ""
    return f"{key} — {names}（{head}{'，'.join(acts) if acts else '没有动作'}）"


def staged_edits(data: "StepTableData", db: Db) -> list[str]:
    """What the panel holds that the database does not, in the annotator's words."""
    from tda.ui.steps_model import stored_signature, table_signature

    from tda.core.model import machine_only

    what: list[str] = []
    if table_signature(data.steps, data.actions) != stored_signature(db, data.desktop):
        what.append("步骤/动作")
    # the panel's table is the machine's own (the 框外零件 sit in data.context,
    # task U18): compared with the whole stored table a desktop with one always
    # read as an unapplied instance edit (found in U34)
    stored = machine_only(db.instances(data.desktop))
    if data.extras:
        what.append("添加的零件")
    elif {k: v for k, v in data.instances.items()} != stored:
        what.append("实例表")
    if data.relations is not None and data.relations.dirty:
        what.append("约束边")
    if data.relabelled:
        # a 框外零件 renamed and not applied (U18): a write-through command reloads
        # the table, and the rename would go without a word (U34 round 2, N3)
        what.append("框外零件名称")
    return what


@dataclass
class AppliedMerge:
    """What 「Apply 并合并」 did: the save's state warnings and the merge."""

    report: MergeReport
    #: :func:`~tda.core.states.validate_events` after the save (as ``Apply`` says).
    messages: list[str] = field(default_factory=list)
    #: ``{view: segments}`` when the save re-cut the pose segments.
    recut: dict = field(default_factory=dict)
    #: What was staged and is now written.
    applied: list[str] = field(default_factory=list)


class _Rehearsal(Exception):
    """Raised to roll a planning transaction back once the plan is in hand."""

    def __init__(self, planned: MergePlan) -> None:
        super().__init__("rehearsal")
        self.planned = planned


def _copy(data: "StepTableData") -> "StepTableData":
    """The panel's table to save from, so a refusal leaves the original untouched.

    ``save`` clears the staged parts and marks the edges written on the object
    it runs on; the taxonomy is shared, not copied.
    """
    return copy.deepcopy(data, {id(data.tax): data.tax})


def _outermost(db: Db) -> None:
    """The save-and-merge transaction must be the outermost one.

    Inside somebody else's block a refusal would not roll the save back -- only
    the outermost block does -- and the rehearsal of :func:`plan` would leave
    its save in the caller's transaction.
    """
    if getattr(db, "_tx_depth", 0):
        raise EditError("「Apply 并合并」不能在另一个事务里进行 / not inside another transaction")


def _check_staged(data: "StepTableData", db: Db) -> None:
    what = staged_edits(data, db)
    if what:
        raise EditError(STAGED.format(what="、".join(what)))


def plan(data: "StepTableData", db: Db, gone: str, keep: str,
         with_staged: bool = False, overlap: str = OVERLAP_REFUSE) -> MergePlan:
    """What merging ``gone`` into ``keep`` would do; nothing is written.

    ``with_staged`` plans the merge on top of the staged edits: the save runs
    in a transaction that is rolled back once the plan is read. ``overlap``
    as :func:`tda.core.merge_instances.plan_merge`; a refusal for overlap
    alone comes back as :class:`MergeOverlap`.
    """
    if not with_staged or not staged_edits(data, db):
        _check_staged(data, db)
        try:
            return plan_merge(db, data.tax, data.desktop, gone, keep, overlap=overlap)
        except MergeError as error:
            raise _refusal(error) from None
    _outermost(db)
    trial = _copy(data)
    try:
        with db.transaction():
            trial.save(db)
            raise _Rehearsal(plan_merge(db, data.tax, data.desktop, gone, keep,
                                        overlap=overlap))
    except _Rehearsal as done:
        return done.planned
    except MergeError as error:
        raise _refusal(error) from None
    except EditError:
        raise
    except Exception as error:  # the save itself refused (a deadlock, ...)
        raise EditError(f"Apply 不了 / could not apply: {error}") from error


def merge(data: "StepTableData", db: Db, gone: str, keep: str,
          with_staged: bool = False,
          overlap: str = OVERLAP_REFUSE) -> MergeReport | AppliedMerge:
    """Merge ``gone`` into ``keep``, written through; the caller reloads the table.

    ``with_staged`` is 「Apply 并合并」: save, plan and write in one
    transaction, and an :class:`AppliedMerge` back. Without staged edits it is
    the plain merge either way. ``overlap="keep"`` is 「合并，重叠处保留 <keep>」.
    """
    what = staged_edits(data, db)
    if not with_staged or not what:
        planned = plan(data, db, gone, keep, overlap=overlap)
        try:
            return apply_merge(db, data.tax, planned, data.annotator)
        except MergeError as error:
            raise _refusal(error) from None
    _outermost(db)
    trial = _copy(data)
    try:
        with db.transaction():
            messages = trial.save(db)
            planned = plan_merge(db, data.tax, data.desktop, gone, keep, overlap=overlap)
            report = apply_merge(db, data.tax, planned, data.annotator)
    except MergeError as error:
        raise _refusal(error) from None
    except EditError:
        raise
    except Exception as error:  # the save refused: nothing was written
        raise EditError(f"Apply 不了 / could not apply: {error}") from error
    return AppliedMerge(report=report, messages=list(messages), recut=dict(trial.recut),
                        applied=what)
