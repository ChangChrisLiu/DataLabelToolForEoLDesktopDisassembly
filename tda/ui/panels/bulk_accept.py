"""The 「接受全部新值…」 dialog of Review mode (``Shift+N``, task U7).

A deliberate re-trace of a confirmed shape puts every confirmed frame it
reaches in conflict, and the right answer to all of them is the same:
``accept_new``.  On D13/scan that was 145 conflicts about five parts, and ``N``
answered one at a time.  This dialog lists the view's open conflicts **per
instance** -- ``ram_latch.01 ×25（帧 17–41）`` -- with a box ticked for each,
and the window hands the ticked instances to
:meth:`~tda.ui.session_review.ReviewMixin.accept_new_bulk`.

:func:`group_conflicts` is the dialog's model and is Qt-free; the dialog only
shows it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from tda.ui.guide import steps_text

__all__ = ["BulkAcceptDialog", "ConflictGroup", "INTRO", "TITLE", "group_conflicts"]

TITLE = "接受全部新值 / Accept every new value"
#: When to use it, and what it does not do.
INTRO = ("这些冲突是你刚重画的形状造成的：接受 = 已确认帧里对应的部分改成现在的形状"
         "（和逐条按 N 一样）。只勾你确定新画的才对的零件；要保留旧形状的，"
         "取消勾选，回队列里逐条按 K。")
#: The confirm button: how many conflicts the ticked rows hold.
CONFIRM = "接受勾选的 {n} 个 / Accept {n}"
INSTANCE_ROLE = int(Qt.ItemDataRole.UserRole)


@dataclass(frozen=True)
class ConflictGroup:
    """The open conflicts of one instance in the view."""

    instance: str
    steps: tuple[int, ...] = field(default_factory=tuple)

    @property
    def count(self) -> int:
        return len(self.steps)

    @property
    def text(self) -> str:
        """``ram_latch.01 ×25（帧 17–41）``."""
        return f"{self.instance} ×{self.count}（帧 {steps_text(self.steps)}）"


def group_conflicts(conflicts: Iterable[dict]) -> list[ConflictGroup]:
    """The dialog's rows: one per instance, its conflicts' frames ascending.

    ``conflicts`` are queue entries (``step``, ``instance``); a frame counts
    once per instance, so a database from before U7 that still holds stacked
    duplicates of one row is not counted twice here either.  Sorted by
    instance key.
    """
    found: dict[str, set[int]] = {}
    for entry in conflicts:
        instance = str(entry.get("instance") or "")
        if not instance:
            continue
        found.setdefault(instance, set()).add(int(entry["step"]))
    return [ConflictGroup(instance, tuple(sorted(steps)))
            for instance, steps in sorted(found.items())]


class BulkAcceptDialog(QDialog):
    """One checkbox per instance, all ticked; the confirm button counts them."""

    def __init__(self, groups: list[ConflictGroup], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(TITLE)
        self.groups = list(groups)
        intro = QLabel(INTRO, self)
        intro.setWordWrap(True)
        self.list = QListWidget(self)
        for group in self.groups:
            item = QListWidgetItem(group.text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(INSTANCE_ROLE, group.instance)
            self.list.addItem(item)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.list.itemChanged.connect(self._refresh)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.list)
        layout.addWidget(self.buttons)
        self.resize(460, 120 + 22 * min(len(self.groups), 12))
        self._refresh()

    def item(self, instance: str) -> Optional[QListWidgetItem]:
        """The row of one instance (for the tests)."""
        for row in range(self.list.count()):
            if self.list.item(row).data(INSTANCE_ROLE) == instance:
                return self.list.item(row)
        return None

    def selected_instances(self) -> list[str]:
        """The ticked instances, in the order shown."""
        return [str(self.list.item(row).data(INSTANCE_ROLE))
                for row in range(self.list.count())
                if self.list.item(row).checkState() == Qt.CheckState.Checked]

    def selected_count(self) -> int:
        """How many conflicts the ticked rows hold."""
        ticked = set(self.selected_instances())
        return sum(group.count for group in self.groups if group.instance in ticked)

    def _refresh(self, *_args) -> None:
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        n = self.selected_count()
        ok.setText(CONFIRM.format(n=n))
        ok.setEnabled(n > 0)
