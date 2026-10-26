"""「这台没有这个零件…」 / 「其实有…」 on an Instances row of the S1 panel (task U34).

Mixed into :class:`tda.ui.panels.steptable.StepTablePanel` like
:class:`~tda.ui.panels.reclass_menu.ReclassMenuMixin`, which provides
``self.data``, ``self.db``, ``self.desktop``, ``self.status``,
``self.set_desktop``, ``self._show_error`` and ``sigSaved``. Right-click a part
-> the entry -> the confirmation -> written through; the panel reloads and
``sigSaved`` makes the window reopen the session on the same frame (the part
leaves -- or comes back to -- every task card). The logic is in
:mod:`tda.ui.steps_absent`.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QMenu, QMessageBox

from tda.core.absent import AbsentPatch
from tda.ui import steps_absent as SA
from tda.ui.steps_values import EditError

__all__ = ["ABSENT_TITLE", "AbsentMenuMixin", "confirm_lines", "done_line"]

ABSENT_TITLE = "这台没有这个零件 / not in this machine"


def confirm_lines(key: str, patch: AbsentPatch) -> list[str]:
    """The confirmation of either entry, in the annotator's words."""
    if patch.absent:
        return [f"{key}：这台机器根本没有它？",
                "从第 1 帧起它就不在机器里：每一帧的任务卡都不再要它，cli check 和导出里也没有它；"
                "日志里对它的动作留着，但不再算。",
                "它的螺丝、接头不跟着变，要单独看。",
                "立即写入，不经 Apply；右键「其实有」改回来。"]
    return [f"{key}：其实有这个零件？",
            "它回到机器里，从第 1 帧起照日志算，任务卡又要它。",
            "立即写入，不经 Apply。"]


def done_line(key: str, patch: AbsentPatch) -> str:
    """The status line after either entry."""
    if patch.absent:
        text = f"{key}：一开始就不在（这台没有它），每一帧都不再要它"
    else:
        text = f"{key}：其实有，回到机器里（照日志算）"
    if patch.frames:
        text += f"；{patch.frames} 个已确认帧会重新核对"
    return text + " / written"


class AbsentMenuMixin:
    """The 「这台没有这个零件」 command of the Instances table."""

    def add_absent_action(self, menu: QMenu, key: str):
        """Add 「这台没有这个零件…」 (or 「其实有…」 on a row said absent) for ``key``."""
        choice = SA.absent_choice(self.data, key)
        if choice is None:
            return None
        text = SA.MENU_PRESENT if choice == SA.PRESENT else SA.MENU_ABSENT
        return menu.addAction(text, lambda: self.prompt_absent(key))

    def ask_absent(self, key: str, lines: list[str]) -> bool:
        """The confirmation (a method, so a test answers it); ``True``: go ahead."""
        answer = QMessageBox.question(self, ABSENT_TITLE, "\n".join(lines),
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes

    def prompt_absent(self, key: str) -> Optional[AbsentPatch]:
        """Say what the entry does, then do it."""
        try:
            planned = SA.plan(self.data, self.db, key)
        except EditError as error:
            self._show_error(str(error))
            return None
        if not self.ask_absent(key, confirm_lines(key, planned)):
            self.status.setText(f"没有改 / {key} was not changed.")
            return None
        return self.absent_instance(key, present=not planned.absent)

    def absent_instance(self, key: str, present: bool = False) -> Optional[AbsentPatch]:
        """Write 「这台没有这个零件」 (or, ``present``, 「其实有」) now; reload, tell the window."""
        try:
            patch = (SA.unmark if present else SA.mark)(self.data, self.db, key)
        except EditError as error:
            self._show_error(str(error))
            return None
        except Exception as error:  # a failed write rolled back; the panel stays up
            self._show_error(f"{type(error).__name__}: {error}")
            return None
        self.set_desktop(self.desktop)     # written through: read it back
        self.status.setText(done_line(key, patch))
        self.sigSaved.emit(self.desktop)   # the window reopens the session on this frame
        return patch
