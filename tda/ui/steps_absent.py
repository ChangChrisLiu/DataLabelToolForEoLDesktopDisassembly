"""「这台没有这个零件」 / 「其实有」 in stage S1: the Qt-free half (task U34).

Right-click a row of the Instances table -> 「这台没有这个零件」 (a part the
sheet logs and the machine does not have: D34's ``optical_drive.01``) -> the
confirmation -> written through, like 「其实是硬盘架…」
(:mod:`tda.ui.steps_reclass`): the write recompiles the stored events and
queues frozen frames, which ``Apply`` knows nothing about, and the panel
reloads afterwards -- so it is refused while the panel holds anything not
applied yet (:data:`STAGED`). The inverse, on a row said absent, is 「其实有」.
The work is :mod:`tda.core.absent`; refusals come back as
:class:`~tda.ui.steps_values.EditError`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from tda.core import absent as AB
from tda.core.db import Db
from tda.core.log_identity import CHASSIS_KEY
from tda.core.model import is_extra, is_provisional
from tda.core.states import REMOVED
from tda.ui.steps_merge import staged_edits
from tda.ui.steps_values import EditError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tda.ui.steps_model import StepTableData

__all__ = ["ABSENT", "MENU_ABSENT", "MENU_PRESENT", "PRESENT", "STAGED", "absent_choice", "plan",
           "is_absent_row", "mark", "unmark"]

ABSENT = "absent"
PRESENT = "present"
#: The two menu entries.
MENU_ABSENT = "这台没有这个零件… / Not in this machine…"
MENU_PRESENT = "其实有… / It is there after all…"
#: The refusal while the panel holds unapplied edits.
STAGED = ("先 Apply 或 Revert：Steps 里还有没写入的修改（{what}）。「这台没有这个零件」会立即写入"
          "并重新读入这张表 / apply or revert the staged edits first")


def is_absent_row(data: "StepTableData", key: str) -> bool:
    """Is ``key`` said to be absent from the start (the stored initial state)?"""
    return data.stored_initial.get(str(key)) == REMOVED


def absent_choice(data: "StepTableData", key: str) -> Optional[str]:
    """Which entry the row's menu offers: :data:`ABSENT`, :data:`PRESENT` or ``None``.

    ``None`` for a row the op is never about (a staged part, a 框外零件, a
    draft, the chassis, a class with no ``removed`` state, an added part);
    whatever else the core refuses (a part with shapes) is said when chosen.
    """
    rec = data.instances.get(str(key))
    if rec is None or data.staged_extra(key) or key in data.context:
        return None
    if is_absent_row(data, key):
        return PRESENT
    if is_provisional(key) or is_extra(rec) or rec.cls == CHASSIS_KEY:
        return None
    if REMOVED not in data.tax.states_of(rec.cls):
        return None
    return ABSENT


def _check_staged(data: "StepTableData", db: Db) -> None:
    what = staged_edits(data, db)
    if what:
        raise EditError(STAGED.format(what="、".join(what)))


def plan(data: "StepTableData", db: Db, key: str) -> AB.AbsentPatch:
    """What the menu entry would write (the confirmation); nothing is written."""
    _check_staged(data, db)
    try:
        if is_absent_row(data, key):
            return AB.plan_present(db, data.tax, data.desktop, key)
        return AB.plan_absent(db, data.tax, data.desktop, key)
    except AB.AbsentError as error:
        raise EditError(str(error)) from None


def mark(data: "StepTableData", db: Db, key: str) -> AB.AbsentPatch:
    """「这台没有这个零件」, written through; the caller reloads the table."""
    _check_staged(data, db)
    try:
        return AB.mark_absent(db, data.tax, data.desktop, key, data.annotator)[1]
    except AB.AbsentError as error:
        raise EditError(str(error)) from None


def unmark(data: "StepTableData", db: Db, key: str) -> AB.AbsentPatch:
    """「其实有」, written through; the caller reloads the table."""
    _check_staged(data, db)
    try:
        return AB.mark_present(db, data.tax, data.desktop, key, data.annotator)[1]
    except AB.AbsentError as error:
        raise EditError(str(error)) from None
