"""The 「可见性审查」 tab's run and other-views widgets (task U36).

Two widgets the review panel (:mod:`tda.ui.panels.review`) puts around its
可见性审查 list; neither decides anything, both only show what the session
read and **report** what was clicked:

* :class:`RunSummary` -- at the top of the queue, one row per part in the
  queue's order (the longest run first): how many frames / stretches it is
  看不见 in every view, 「—」 where a view is not annotated. A part hidden for
  forty frames in one view and three in another stands out; clicking a row
  asks for that part's first line (:attr:`RunSummary.sigPartPicked`);
* :class:`RunDetail` -- under the list, about the selected line: the run's
  first, middle and last frames cut around the part (the frame strip); what
  every view says on the same steps (seen / not seen / why, the views that
  see it highlighted); the reason they point at, as a default the annotator
  still answers with the key; and two buttons, no keys --
  「从这帧起另算」 (:attr:`RunDetail.sigSplit`) and 「同步到其他视角」
  (:attr:`RunDetail.sigSync`, enabled only when the window offers a sync).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tda.core.model import VIEWS
from tda.ui import visreview_text as VT

__all__ = ["HIGHLIGHT", "RunDetail", "RunSummary", "rgb_pixmap"]

#: The background of another view that sees the part on the run's steps.
HIGHLIGHT = QColor(200, 235, 200)
#: The data role a summary row keeps its part under.
PART_ROLE = int(Qt.ItemDataRole.UserRole)


def rgb_pixmap(image: Optional[np.ndarray]) -> QPixmap:
    """An RGB ``uint8`` array as a pixmap (a null one for ``None``)."""
    if image is None:
        return QPixmap()
    rgb = np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


def _table(columns: int, headers) -> QTableWidget:
    table = QTableWidget(0, columns)
    table.setHorizontalHeaderLabels(list(headers))
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    table.setWordWrap(False)
    head = table.horizontalHeader()
    head.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    head.setStretchLastSection(True)
    table.verticalHeader().setDefaultSectionSize(18)
    return table


class RunSummary(QWidget):
    """Per part: 看不见 frames / stretches in every view (the top of the queue)."""

    #: A row was clicked: that part's first line, please.
    sigPartPicked = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.table = _table(1 + len(VIEWS), ("零件", *VIEWS))
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setMaximumHeight(84)
        self.table.setToolTip("看不见的帧数 / 段数（每个视角；— 没标注，未确认 = 只有草稿）。"
                              "点一行选中这个零件的第一段")
        self.table.cellClicked.connect(self._clicked)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.table)

    def set_rows(self, rows: list[dict]) -> None:
        """Show :meth:`~tda.ui.session_visruns.VisRunsMixin.visibility_summary`'s rows."""
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            part = QTableWidgetItem(str(row["instance"]))
            part.setData(PART_ROLE, str(row["instance"]))
            self.table.setItem(i, 0, part)
            values = row.get("views") or {}
            counted = [int(v[0]) for v in values.values() if v is not None]
            most = max(counted) if counted else 0
            differ = len(counted) > 1 and min(counted) != most
            for j, view in enumerate(VIEWS, start=1):
                value = values.get(view)
                cell = QTableWidgetItem(VT.summary_cell(value, (row.get("states") or {}).get(
                    view, "")))
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if differ and value is not None and int(value[0]) == most and most:
                    font = QFont(cell.font())
                    font.setBold(True)
                    cell.setFont(font)
                if value is not None and int(value[0]):
                    cell.setToolTip(f"{view}：看不见 {value[0]} 帧，{value[1]} 段，最长 {value[2]} 帧")
                self.table.setItem(i, j, cell)
        self.setVisible(bool(rows))

    def rows(self) -> list[tuple[str, ...]]:
        """The cells as shown, row by row (for the tests and the window)."""
        return [tuple(self.table.item(i, j).text() if self.table.item(i, j) else ""
                      for j in range(self.table.columnCount()))
                for i in range(self.table.rowCount())]

    def bold(self, row: int, view: str) -> bool:
        """Is one cell bold (the view it is hidden longest in, when the views differ)?"""
        item = self.table.item(row, 1 + list(VIEWS).index(view))
        return bool(item is not None and item.font().bold())

    def _clicked(self, row: int, _column: int) -> None:
        item = self.table.item(row, 0)
        if item is not None:
            self.sigPartPicked.emit(str(item.data(PART_ROLE)))


class RunDetail(QWidget):
    """The selected line: its frame strip, the other views, the suggestion, two buttons."""

    #: 「从这帧起另算」 was clicked.
    sigSplit = Signal()
    #: 「同步到其他视角」 was clicked.
    sigSync = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._strip_labels: list[QLabel] = []
        self._strip_steps: list[int] = []
        strip = QHBoxLayout()
        strip.setContentsMargins(0, 0, 0, 0)
        for _ in range(3):
            label = QLabel()
            label.setFixedSize(100, 100)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: gray; font-size: 8pt")
            self._strip_labels.append(label)
            strip.addWidget(label)
        strip.addStretch(1)
        self._strip_caption = QLabel("")
        self._strip_caption.setStyleSheet("color: gray; font-size: 8pt")
        self.views = _table(len(VT.VIEW_COLUMNS), VT.VIEW_COLUMNS)
        self.views.setRowCount(len(VIEWS))
        self.views.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._fit_views()
        self.views.setToolTip("同一个零件、同样的这几步，在每个视角里：看得见几帧、看不见几帧、为什么。"
                              "绿底 = 另一个视角看得见它")
        self._changes = QLabel("")
        self._changes.setWordWrap(True)
        self._changes.setStyleSheet("color: #a05000; font-size: 8pt")
        self._suggestion = QLabel("")
        self._suggestion.setWordWrap(True)
        self._suggestion.setStyleSheet("font-weight: bold")
        self.split_button = QToolButton()
        self.split_button.setText(VT.SPLIT_BUTTON)
        self.split_button.setToolTip(VT.SPLIT_TIP)
        self.sync_button = QToolButton()
        self.sync_button.setText(VT.SYNC_BUTTON)
        self.sync_button.setToolTip(VT.SYNC_IDLE)
        self.sync_button.setEnabled(False)
        for button in (self.split_button, self.sync_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.split_button.clicked.connect(lambda _c=False: self.sigSplit.emit())
        self.sync_button.clicked.connect(lambda _c=False: self.sigSync.emit())
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addWidget(self.split_button)
        buttons.addWidget(self.sync_button)
        buttons.addStretch(1)
        # the most needed first: a short dock shows the suggestion and the
        # buttons, then the views, and scrolls to the pictures
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        box.addWidget(self._suggestion)
        box.addLayout(buttons)
        box.addWidget(self.views)
        box.addWidget(self._changes)
        box.addLayout(strip)
        box.addWidget(self._strip_caption)
        box.addStretch(1)
        self.show_entry(None, {})

    def _fit_views(self) -> None:
        """The other-views table exactly as tall as its header and four rows."""
        rows = sum(self.views.rowHeight(i) for i in range(self.views.rowCount()))
        head = self.views.horizontalHeader().sizeHint().height()
        self.views.setFixedHeight(head + rows + 2 * self.views.frameWidth() + 2)

    # -- content ---------------------------------------------------------------
    def show_entry(self, entry: Optional[dict], detail: dict) -> None:
        """The line ``entry`` (``None``: nothing selected) and what the views say."""
        rows = list(detail.get("views") or ())
        for i in range(len(VIEWS)):
            row = rows[i] if i < len(rows) else None
            cells = VT.view_cells(row) if row is not None else (VIEWS[i], "", "", "")
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if j in (1, 2):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if row is not None and row.get("highlight"):
                    item.setBackground(QBrush(HIGHLIGHT))
                if row is not None and row.get("here"):
                    font = QFont(item.font())
                    font.setBold(True)
                    item.setFont(font)
                if j == 3 and text:
                    item.setToolTip(text)
                self.views.setItem(i, j, item)
        self._fit_views()
        changes = (entry or {}).get("changes") or []
        self._changes.setText(VT.CHANGES.format(steps=VT.steps_text(changes)) if changes else "")
        self._changes.setVisible(bool(changes))
        text = VT.suggestion_text(detail.get("suggestion"), entry or {})
        self._suggestion.setText(text)
        self._suggestion.setVisible(bool(text))
        self.split_button.setEnabled(entry is not None and len(entry.get("steps") or ()) > 0)
        if entry is None:
            self.set_strip([])

    def set_strip(self, frames: list[dict]) -> None:
        """The run's thumbnails: ``[{"step", "image"}]`` (at most three)."""
        self._strip_steps = [int(f["step"]) for f in frames]
        self._strip_caption.setText(
            "第 " + " · ".join(str(s) for s in self._strip_steps) + " 帧（首 · 中 · 尾，框 = 零件完整形状）"
            if self._strip_steps else "")
        self._strip_caption.setVisible(bool(self._strip_steps))
        for i, label in enumerate(self._strip_labels):
            if i < len(frames):
                pixmap = rgb_pixmap(frames[i].get("image"))
                label.setPixmap(pixmap)
                if pixmap.isNull():
                    label.setText(f"第 {frames[i]['step']} 帧\n（没有图）")
                label.setToolTip(f"第 {frames[i]['step']} 帧")
                label.setVisible(True)
            else:
                label.clear()
                label.setVisible(False)

    def set_sync_offer(self, text: Optional[str]) -> None:
        """Enable 「同步到其他视角」 with what it would do, or disable it (``None``)."""
        self.sync_button.setEnabled(bool(text))
        self.sync_button.setToolTip(f"{VT.SYNC_TIP}\n{text}" if text else VT.SYNC_IDLE)

    # -- for the window and the tests --------------------------------------------
    def rows(self) -> list[tuple[str, ...]]:
        """The other-views table as shown."""
        return [tuple(self.views.item(i, j).text() if self.views.item(i, j) else ""
                      for j in range(self.views.columnCount()))
                for i in range(self.views.rowCount())]

    def highlighted(self) -> list[str]:
        """The views shown on the highlight (they see the part on these steps)."""
        out = []
        for i in range(self.views.rowCount()):
            item = self.views.item(i, 0)
            if item is not None and item.background().color() == HIGHLIGHT:
                out.append(item.text())
        return out

    def suggestion(self) -> str:
        """The suggestion line as shown (``""`` for none)."""
        return self._suggestion.text()

    def changes(self) -> str:
        """The 「挡住它的零件变了」 line as shown (``""`` for none)."""
        return self._changes.text()

    def strip_steps(self) -> list[int]:
        """The frames the strip shows."""
        return list(self._strip_steps)

    def strip_pixmaps(self) -> list[QPixmap]:
        """The strip's pictures (null where a frame has no image)."""
        return [label.pixmap() for label in self._strip_labels[:len(self._strip_steps)]]
