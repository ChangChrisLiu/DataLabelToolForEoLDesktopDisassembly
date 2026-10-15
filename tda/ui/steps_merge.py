"""「合并到…」 in stage S1: the Qt-free half (task U12).

The importer sometimes makes two instances of one part (D2: ``cover.01``
"Screw Cover" and ``cover.02`` "Front Cover", one cover). Steps mode merges
them with :mod:`tda.core.merge_instances`, and like
:func:`tda.ui.steps_delete.delete_instance` the merge is **written through**:
it moves shapes, confirmed rows and events that ``Apply`` knows nothing about,
so it cannot wait for it. That is why it is refused while the panel holds
anything not applied yet (:func:`staged_edits`): the panel reloads from the
database afterwards, and an edit made a minute ago would be lost without a
word.

This module is what the panel calls: the choices offered, the refusal for
staged edits, the plan for the confirmation and the merge itself. Refusals come
back as :class:`~tda.ui.steps_values.EditError`, like every other S1 refusal.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from tda.core.db import Db
from tda.core.merge_instances import MergeError, MergePlan, MergeReport, apply_merge, plan_merge
from tda.core.model import is_provisional
from tda.ui.steps_values import EditError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tda.ui.steps_model import StepTableData

__all__ = ["MENU_TEXT", "STAGED", "choice_label", "merge", "merge_choices", "plan",
           "staged_edits"]

#: The context-menu entry on an instance row.
MENU_TEXT = "合并到… / Merge into…"
#: The refusal while the panel holds unapplied edits.
STAGED = ("先 Apply 或 Revert：Steps 里还有没写入的修改（{what}）。合并会立即写入并重新读入这张表 / "
          "apply or revert the staged edits first")


def merge_choices(data: "StepTableData", key: str) -> list[str]:
    """The instances ``key`` can be merged into: same class, not a draft, not itself."""
    rec = data.instances.get(key)
    if rec is None or is_provisional(key):
        return []
    return sorted(other for other, inst in data.instances.items()
                  if other != key and inst.cls == rec.cls and not is_provisional(other))


def choice_label(data: "StepTableData", key: str) -> str:
    """``cover.01 — Screw Cover（第 38 步 open）`` for the picker."""
    rec = data.instances[key]
    names = "、".join(rec.raw_names) if rec.raw_names else "日志里没有"
    acts = [f"第 {a.step} 步 {a.verb}" for a in data.actions if a.target == key]
    return f"{key} — {names}（{'，'.join(acts) if acts else '没有动作'}）"


def staged_edits(data: "StepTableData", db: Db) -> list[str]:
    """What the panel holds that the database does not, in the annotator's words."""
    from tda.ui.steps_model import stored_signature, table_signature

    what: list[str] = []
    if table_signature(data.steps, data.actions) != stored_signature(db, data.desktop):
        what.append("步骤/动作")
    stored = db.instances(data.desktop)
    if data.extras:
        what.append("添加的零件")
    elif {k: v for k, v in data.instances.items()} != stored:
        what.append("实例表")
    if data.relations is not None and data.relations.dirty:
        what.append("约束边")
    return what


def _check_staged(data: "StepTableData", db: Db) -> None:
    what = staged_edits(data, db)
    if what:
        raise EditError(STAGED.format(what="、".join(what)))


def plan(data: "StepTableData", db: Db, gone: str, keep: str) -> MergePlan:
    """What merging ``gone`` into ``keep`` would do; nothing is written."""
    _check_staged(data, db)
    try:
        return plan_merge(db, data.tax, data.desktop, gone, keep)
    except MergeError as error:
        raise EditError(str(error)) from None


def merge(data: "StepTableData", db: Db, gone: str, keep: str) -> MergeReport:
    """Merge ``gone`` into ``keep``, written through; the caller reloads the table."""
    planned = plan(data, db, gone, keep)
    try:
        return apply_merge(db, data.tax, planned, data.annotator)
    except MergeError as error:
        raise EditError(str(error)) from None
