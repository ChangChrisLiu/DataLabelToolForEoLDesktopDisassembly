"""The 「说明」 dock of Review mode: the frame's occlusion order and its note (task U44).

The user (2026-10-30): the note is what helps the reader -- and the VLM work
after it -- understand what happened on a frame, so writing it should be
easier and the box bigger; and what the notes written so far explain is the
occlusion order (「这个导风罩拉下来了之后变成了最上面的，遮挡住了别的所有的」).
So the box of task U35 (:class:`~tda.ui.panels.frame_note.FrameNoteBox`) moves
out of the bottom of the Review panel into this dock, where it takes the
spare height, and above it 「这一帧的遮挡顺序」 lists who is painted over whom on
the frame on screen (:mod:`tda.ui.occlusion_order`), so the annotator writes
only what is wrong.  The note stays free text.

Like every panel this decides nothing: clicking a line **reports** its part
(:attr:`OcclusionList.sigPicked`) and the window selects it in the instance
table.  The list takes no keys (no focus); the wheel scrolls it.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDockWidget,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from tda.ui.occlusion_order import FLOOR_NOTE, OrderLine, header_text, line_text
from tda.ui.panels.frame_note import FrameNoteBox

__all__ = ["DOCK_TITLE", "ORDER_TITLE", "ORDER_TIP", "INSTANCE_ROLE", "NotesDock",
           "NotesPanel", "OcclusionList"]

DOCK_TITLE = "说明"
ORDER_TITLE = "这一帧的遮挡顺序"
ORDER_TIP = ("这一帧编译出的零件，从上到下按画的顺序（最上面的先写）：每个零件后面是它在这一帧压着的零件和"
             "压住的像素，多的在前。「（手动）」：这一对的上下是「改层级」定的；「（这一帧单独画）」："
             "这个零件这一帧有单独画的形状；「被完全盖住」：它这一帧一点都露不出来。画框外的零件不列；"
             "机箱和主板垫在最下面，不列为被压的零件（「改层级」定的除外）。"
             "只看不改：点一行在实例表里选中那个零件；遮挡关系不对就在下面的说明里写出来。")
#: The part a line is about.
INSTANCE_ROLE = int(Qt.ItemDataRole.UserRole)


class OcclusionList(QWidget):
    """「这一帧的遮挡顺序」: a header line and one line per part."""

    #: A line was clicked: its instance key (the window selects it).
    sigPicked = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.title = QLabel(ORDER_TITLE)
        self.title.setStyleSheet("font-weight: bold")
        self.header = QLabel(header_text([]))
        self.header.setStyleSheet("color: gray")
        self.list = QListWidget()
        self.list.setWordWrap(True)
        self.list.setFocusPolicy(Qt.FocusPolicy.NoFocus)     # no keys: the canvas keeps them
        self.list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list.itemClicked.connect(self._on_clicked)
        # round 2: the floor layers are not listed as pressed on; said once, small
        self.footer = QLabel(FLOOR_NOTE)
        self.footer.setStyleSheet("color: gray; font-size: 8pt")
        for widget in (self, self.title, self.header, self.list, self.footer):
            widget.setToolTip(ORDER_TIP)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        head = QVBoxLayout()
        head.setSpacing(0)
        head.addWidget(self.title)
        head.addWidget(self.header)
        layout.addLayout(head)
        layout.addWidget(self.list, 1)
        layout.addWidget(self.footer)

    def set_lines(self, lines: list[OrderLine],
                  label: Optional[Callable[[str], str]] = None) -> None:
        """Show ``lines`` (top first); an empty list says so in the header."""
        self.header.setText(header_text(lines))
        self.list.clear()
        for line in lines:
            item = QListWidgetItem(line_text(line, label))
            item.setData(INSTANCE_ROLE, line.instance)
            item.setToolTip(ORDER_TIP)
            self.list.addItem(item)

    def texts(self) -> list[str]:
        """The lines as shown (tests, the report)."""
        return [self.list.item(i).text() for i in range(self.list.count())]

    def _on_clicked(self, item: QListWidgetItem) -> None:
        instance = item.data(INSTANCE_ROLE)
        if instance:
            self.sigPicked.emit(str(instance))


class NotesPanel(QWidget):
    """The dock's content: the order list above, the note box below taking the rest."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.order = OcclusionList()
        self.note_box: Optional[FrameNoteBox] = None
        self.split = QSplitter(Qt.Orientation.Vertical)
        self.split.setChildrenCollapsible(False)
        self.split.addWidget(self.order)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.split)

    def set_note_box(self, box: FrameNoteBox) -> None:
        """Take the Review panel's note box (U35's, unchanged) and give it the spare height."""
        self.note_box = box
        box.set_tall(True)
        self.split.addWidget(box)
        # the box is what the user asked to be bigger: it gets the larger share
        self.split.setStretchFactor(0, 2)
        self.split.setStretchFactor(1, 3)
        box.show()


class NotesDock(QDockWidget):
    """The dock, telling a close by the annotator apart from the window hiding it."""

    #: The title bar's close button: the annotator does not want the dock.
    sigClosedByUser = Signal()

    def closeEvent(self, event) -> None:  # noqa: D102 - Qt override
        super().closeEvent(event)
        if event.isAccepted():
            self.sigClosedByUser.emit()
