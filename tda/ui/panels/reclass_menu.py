"""「其实是硬盘架…」 / 「其实是卡扣…」 on an Instances row of the S1 panel (task U21).

Mixed into :class:`tda.ui.panels.steptable.StepTablePanel` like
:class:`~tda.ui.panels.merge_into.MergeIntoMixin`, which provides ``self.data``,
``self.db``, ``self.desktop``, ``self.status``, ``self.set_desktop``,
``self._show_error`` and ``sigSaved``. Right-click a drive latch or drive cage ->
the entry -> the confirmation -> written through; the panel reloads and
``sigSaved`` makes the window reopen the session on the same frame (the part
carries another key now). The logic is in :mod:`tda.ui.steps_reclass`.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QMenu, QMessageBox

from tda.core.reclass import ReclassReport
from tda.ui import steps_reclass as RC
from tda.ui.steps_values import EditError

__all__ = ["RECLASS_TITLE", "ReclassMenuMixin"]

RECLASS_TITLE = "改类别 / reclass"


class ReclassMenuMixin:
    """The reclass command of the Instances table."""

    def add_reclass_action(self, menu: QMenu, key: str):
        """Add 「其实是硬盘架…」 / 「其实是卡扣…」 for ``key`` when it is one of the two."""
        target = RC.reclass_choice(self.data, key)
        if target is None:
            return None
        return menu.addAction(RC.menu_text(target), lambda: self.prompt_reclass(key))

    def ask_reclass(self, key: str, lines: list[str]) -> bool:
        """The confirmation (a method, so a test answers it); ``True``: go ahead."""
        answer = QMessageBox.question(self, RECLASS_TITLE, "\n".join(lines),
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes

    def prompt_reclass(self, key: str) -> Optional[ReclassReport]:
        """Show what reclassing ``key`` moves, then do it."""
        try:
            planned = RC.plan(self.data, self.db, key)
        except EditError as error:
            self._show_error(str(error))
            return None
        if not self.ask_reclass(key, planned.summary_lines(undo_hint=RC.NO_UNDO)):
            self.status.setText(f"没有改类别 / {key} was not reclassed.")
            return None
        return self.reclass_instance(key)

    def reclass_instance(self, key: str) -> Optional[ReclassReport]:
        """Reclass ``key`` now; reload and tell the window."""
        try:
            report = RC.reclass(self.data, self.db, key)
        except EditError as error:
            self._show_error(str(error))
            return None
        except Exception as error:  # a failed reclass rolled back; the panel stays up
            self._show_error(f"{type(error).__name__}: {error}")
            return None
        self.set_desktop(self.desktop)     # written through: read it back
        self.status.setText(report.status_line())
        self.sigSaved.emit(self.desktop)   # the window reopens the session on this frame
        return report
