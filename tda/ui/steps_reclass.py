"""「其实是硬盘架」 / 「其实是卡扣」 in stage S1: the Qt-free half (task U21).

Right-click a ``drive_latch`` or ``drive_cage`` row of the Instances table ->
「其实是硬盘架…」 / 「其实是卡扣…」 -> the confirmation (what moves, the cage's
latch, which confirmed frames re-check) -> written through, like 「合并到…」
(:mod:`tda.ui.steps_merge`): the reclass moves shapes and confirmed rows that
``Apply`` knows nothing about. So it is refused while the panel holds anything
not applied yet (:data:`STAGED`) -- the panel reloads afterwards, and the staged
edit would be lost without a word. The work is :mod:`tda.core.reclass`;
refusals come back as :class:`~tda.ui.steps_values.EditError`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from tda.core.db import Db
from tda.core.implied import is_implied
from tda.core.merge_instances import MergeError
from tda.core.model import is_context, is_provisional
from tda.core.reclass import ReclassPlan, ReclassReport, plan_reclass, reclass_instance
from tda.core.reclass_replay import CAPTION, reclass_target
from tda.ui.steps_merge import staged_edits
from tda.ui.steps_values import EditError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tda.ui.steps_model import StepTableData

__all__ = ["STAGED", "menu_text", "plan", "reclass", "reclass_choice"]

#: The refusal while the panel holds unapplied edits.
STAGED = ("先 Apply 或 Revert：Steps 里还有没写入的修改（{what}）。改类别会立即写入并重新读入这张表 / "
          "apply or revert the staged edits first")
#: What the Steps undo hint says instead of the session's Ctrl+Z.
NO_UNDO = "（Steps 里没有撤销）"


def reclass_choice(data: "StepTableData", key: str) -> Optional[str]:
    """The class ``key`` may be said to be instead, or ``None`` (no menu entry)."""
    rec = data.instances.get(key)
    if rec is None or is_provisional(key) or is_context(rec) or is_implied(rec):
        return None
    return reclass_target(rec.cls)


def menu_text(target: str) -> str:
    """``其实是硬盘架… / Reclass as drive_cage`` for the context menu."""
    return f"{CAPTION.get(target, target)}… / Reclass as {target}"


def _check_staged(data: "StepTableData", db: Db) -> None:
    what = staged_edits(data, db)
    if what:
        raise EditError(STAGED.format(what="、".join(what)))


def plan(data: "StepTableData", db: Db, key: str) -> ReclassPlan:
    """What reclassing ``key`` would do (the confirmation); nothing is written."""
    _check_staged(data, db)
    try:
        return plan_reclass(db, data.tax, data.desktop, key)
    except MergeError as error:            # ReclassError is one
        raise EditError(str(error)) from None


def reclass(data: "StepTableData", db: Db, key: str) -> ReclassReport:
    """Reclass ``key``, written through; the caller reloads the table."""
    _check_staged(data, db)
    try:
        return reclass_instance(db, data.tax, data.desktop, key, None, data.annotator)
    except MergeError as error:
        raise EditError(str(error)) from None
