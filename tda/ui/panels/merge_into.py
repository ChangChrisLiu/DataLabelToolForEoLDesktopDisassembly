"""「合并到…」 on an Instances row of the S1 panel (task U12): the Qt half.

Mixed into :class:`tda.ui.panels.steptable.StepTablePanel`, which provides
``self.data``, ``self.db``, ``self.desktop``, ``self.status``,
``self.set_desktop``, ``self._show_error`` and ``sigSaved``. Right-click a row
-> 「合并到…」 -> pick the instance it really is (same class only) -> a
confirmation listing what moves and what is dropped -> written through, the
panel reloads, and ``sigSaved`` makes the window reopen the session on the same
frame (the shapes now carry another key).

With edits staged in the table the confirmation offers **「Apply 并合并」**
(round 2): the table is applied and the merge written in one transaction, so
the session is reopened -- and the frozen frames re-checked -- only once,
against the merged state; a refusal anywhere writes nothing and leaves the
staged edits in the table. The logic is in :mod:`tda.ui.steps_merge` and
:mod:`tda.core.merge_instances`.

When the plain merge is refused only because both are drawn or confirmed on
the same frames (task U21 round 2), the refusal is shown as U12 words it with a
second button, 「合并，重叠处保留 <keep>」: ``keep``'s shapes and rows stay
there, ``gone``'s go (:meth:`MergeIntoMixin.ask_merge_overlap`).
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
        staged = M.staged_edits(self.data, self.db)
        try:
            planned = M.plan(self.data, self.db, key, keep, with_staged=bool(staged))
        except M.MergeOverlap as overlap:
            # both drawn on the same frames: the refusal stands as U12 says
            # it, and the merge keeping keep's there is offered (U21 round 2)
            self.prompt_merge_overlap(key, keep, str(overlap), bool(staged), staged)
            return
        except EditError as error:
            self._show_error(str(error))
            return
        lines = planned.summary_lines(with_apply=bool(staged))
        if staged:
            lines = [M.APPLY_AND_MERGE.format(what="、".join(staged)), "", *lines]
        answer = QMessageBox.question(
            self, PICK_TITLE, "\n".join(lines),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            self.status.setText(f"没有合并 / {key} was not merged.")
            return
        self.merge_instance(key, keep, with_staged=bool(staged))

    def prompt_merge_overlap(self, gone: str, keep: str, refusal: str, with_staged: bool,
                             staged: list) -> Optional[MergeReport]:
        """The plain merge is refused for overlap: offer 「合并，重叠处保留 <keep>」."""
        try:
            planned = M.plan(self.data, self.db, gone, keep, with_staged=with_staged,
                             overlap=M.OVERLAP_KEEP)
        except EditError as error:
            self._show_error(str(error))
            return None
        lines = planned.summary_lines(with_apply=with_staged)
        if staged:
            lines = [M.APPLY_AND_MERGE.format(what="、".join(staged)), "", *lines]
        if not self.ask_merge_overlap(gone, keep, refusal, lines):
            self.status.setText(f"没有合并 / {gone} was not merged.")
            return None
        return self.merge_instance(gone, keep, with_staged=with_staged, overlap=M.OVERLAP_KEEP)

    def ask_merge_overlap(self, gone: str, keep: str, refusal: str, lines: list[str]) -> bool:
        """The plain merge's refusal, then the overlap merge's confirmation; ``True``: merge.

        The second button, 「合并，重叠处保留 <keep>」, is the only way on; a
        method so a test answers it without a modal loop.
        """
        box = QMessageBox(self)
        box.setWindowTitle(PICK_TITLE)
        box.setIcon(QMessageBox.Question)
        box.setText("\n".join([refusal, "", *lines]))
        merge = box.addButton(M.KEEP_BUTTON.format(keep=keep), QMessageBox.AcceptRole)
        cancel = box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(cancel)
        box.exec()
        return box.clickedButton() is merge

    def merge_instance(self, gone: str, keep: str, with_staged: bool = False,
                       overlap: str = "refuse") -> Optional[MergeReport]:
        """Merge ``gone`` into ``keep`` now; reload and tell the window.

        ``with_staged`` is 「Apply 并合并」: the staged edits are written in
        the same transaction. ``overlap="keep"`` keeps ``keep``'s shapes and rows
        where both are drawn or confirmed. Refused, nothing is written and the
        table keeps its edits.
        """
        try:
            result = M.merge(self.data, self.db, gone, keep, with_staged=with_staged,
                             overlap=overlap)
        except EditError as error:
            self._show_error(str(error))
            return None
        except Exception as error:  # a failed merge rolled back; the panel stays up
            self._show_error(f"{type(error).__name__}: {error}")
            return None
        applied = result if isinstance(result, M.AppliedMerge) else None
        report = applied.report if applied else result
        self.set_desktop(self.desktop)     # written through: read it back
        text = report.status_line()
        if applied:
            warnings = (f"，状态警告 {len(applied.messages)} 条" if applied.messages else "")
            recut = (f"，位姿段重切 {len(applied.recut)} 个视角" if applied.recut else "")
            text = f"已 Apply（{'、'.join(applied.applied)}{warnings}{recut}）并合并：{text}"
        self.status.setText(text)
        self.sigSaved.emit(self.desktop)   # the window reopens the session on this frame
        return report
