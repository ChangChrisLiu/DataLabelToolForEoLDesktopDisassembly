"""「合并到…」 on an Instances row of the S1 panel (task U12): the Qt half.

Mixed into :class:`tda.ui.panels.steptable.StepTablePanel`, which provides
``self.data``, ``self.db``, ``self.desktop``, ``self.status``,
``self.set_desktop``, ``self._show_error`` and ``sigSaved``. Right-click a row
-> 「合并到…」 -> pick the instance it really is (same class only) -> a
confirmation listing what moves and what is dropped -> written through, the
panel reloads, and ``sigSaved`` makes the window reopen the session on the same
frame (the shapes now carry another key). The logic is in
:mod:`tda.ui.steps_merge` and :mod:`tda.core.merge_instances`.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QInputDialog, QMenu, QMessageBox

from tda.core.merge_instances import MergeReport
from tda.ui import steps_merge as M
from tda.ui.steps_values import EditError

__all__ = ["MergeIntoMixin", "PICK_TITLE"]

#: The picker's and the confirmation's window title.
PICK_TITLE = "合并到… / Merge into"


class MergeIntoMixin:
    """The merge command of the Instances table."""

    def add_merge_action(self, menu: QMenu, key: str):
        """Add 「合并到…」 for ``key``; greyed out when nothing of its class is there."""
        action = menu.addAction(M.MENU_TEXT, lambda: self.prompt_merge(key))
        action.setEnabled(bool(M.merge_choices(self.data, key)))
        return action

    def prompt_merge(self, key: str) -> None:
        """Ask which instance ``key`` really is, show what moves, then merge."""
        choices = M.merge_choices(self.data, key)
        if not choices:
            self._show_error(f"没有能合并的同类实例 / nothing of {key}'s class to merge into")
            return
        labels = [M.choice_label(self.data, other) for other in choices]
        label, ok = QInputDialog.getItem(
            self, PICK_TITLE,
            f"{key} 其实就是哪一个零件？它的形状、确认过的行、标签和动作都并过去，"
            f"{key} 这一行删除。\n{key} is really which part?",
            labels, 0, False)
        if not ok or label not in labels:
            return
        keep = choices[labels.index(label)]
        try:
            planned = M.plan(self.data, self.db, key, keep)
        except EditError as error:
            self._show_error(str(error))
            return
        answer = QMessageBox.question(
            self, PICK_TITLE, "\n".join(planned.summary_lines()),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            self.status.setText(f"没有合并 / {key} was not merged.")
            return
        self.merge_instance(key, keep)

    def merge_instance(self, gone: str, keep: str) -> Optional[MergeReport]:
        """Merge ``gone`` into ``keep`` now; reload and tell the window."""
        try:
            report = M.merge(self.data, self.db, gone, keep)
        except EditError as error:
            self._show_error(str(error))
            return None
        except Exception as error:  # a failed merge rolled back; the panel stays up
            self._show_error(f"{type(error).__name__}: {error}")
            return None
        self.set_desktop(self.desktop)     # written through: read it back
        self.status.setText(report.status_line())
        self.sigSaved.emit(self.desktop)   # the window reopens the session on this frame
        return report
