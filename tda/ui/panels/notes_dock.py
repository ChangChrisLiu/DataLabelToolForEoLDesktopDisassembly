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

Task W3c puts 「谁压着谁 · 这台还没裁定的（n）」 above the list
(:class:`RulingsSection`): the overlapping pairs of the frame that still want
the user's ruling (:mod:`tda.ui.cover_rulings`), each with 「挡着」 / 「不挡」
(buttons only -- **no keys**), a batch line per class pair shared by two or
more rows, and the folded 「已裁定（k）」 with 「撤销」.  It reports
(:attr:`RulingsSection.sigRule`, :attr:`RulingsSection.sigClear`); the window
writes.  It is collapsible, and whether it is open is remembered
(``QSettings notes/rulings_open``, the window's).
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Mapping, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tda.ui.cover_rulings import (
    BLOCKS,
    NOT_BLOCKS,
    RULING_TEXT,
    Candidate,
    candidate_text,
    class_pairs,
    ruled_text,
)
from tda.ui.occlusion_order import FLOOR_NOTE, OrderLine, header_text, line_text
from tda.ui.panels.frame_note import FrameNoteBox

__all__ = ["BATCH_TEXT", "DOCK_TITLE", "NONE_PENDING", "ORDER_TITLE", "ORDER_TIP",
           "INSTANCE_ROLE", "NotesDock", "NotesPanel", "OcclusionList", "RULED_TITLE",
           "RULINGS_TIP", "RULINGS_TITLE", "RulingsSection", "UNDO_RULING"]

DOCK_TITLE = "说明"
ORDER_TITLE = "这一帧的遮挡顺序"
ORDER_TIP = ("这一帧编译出的零件，从上到下按画的顺序（最上面的先写）：每个零件后面是它在这一帧压着的零件和"
             "压住的像素，多的在前。「（手动）」：这一对的上下是「改层级」定的；「（这一帧单独画）」："
             "这个零件这一帧有单独画的形状；「被完全盖住」：它这一帧一点都露不出来。画框外的零件不列；"
             "机箱和主板垫在最下面，不列为被压的零件（「改层级」定的除外）。"
             "只看不改：点一行在实例表里选中那个零件；遮挡关系不对就在下面的说明里写出来。")
#: The part a line is about.
INSTANCE_ROLE = int(Qt.ItemDataRole.UserRole)

#: 「谁压着谁」 (task W3c).
RULINGS_TITLE = "谁压着谁 · 这台还没裁定的（{n}）"
RULED_TITLE = "已裁定（{k}）"
BATCH_TEXT = "同类对本帧全部：{above} → {below}（{n} 对）"
NONE_PENDING = "这一帧没有要裁定的对"
UNDO_RULING = "撤销"
RULINGS_TIP = ("这一帧上重叠的对（像素 ≥ 1000，机箱里），还没有任何边、没有结构关系（随父件、装在 / 插在 / "
               "拧住、主板上的 CPU / 内存 / 散热器 / 扩展卡）、这台也还没裁定的。看图判断："
               "「挡着」= 拆下面那件之前必须先拿走上面那件；「不挡」= 只是画面上重叠。每对这台只问一次。"
               "「建议收」/「存疑」是裁定单按类别给的提示（悬停一行看理由），不是裁定。没有快捷键；"
               "Ctrl+Z 撤回最近一次裁定，「已裁定」里的「撤销」去掉任一条。")
#: The two areas' tallest heights: the dock keeps most of its room for the note.
PENDING_MAX_HEIGHT = 220
RULED_MAX_HEIGHT = 140


def _button(text: str, tip: str = "") -> QPushButton:
    """A small button that never takes the keyboard (no keys: the canvas keeps them)."""
    button = QPushButton(text)
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    button.setAutoDefault(False)
    button.setDefault(False)
    # compact: the dock is narrow, and the words beside it should get the room
    button.setStyleSheet("QPushButton { padding: 1px 6px; }")
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    if tip:
        button.setToolTip(tip)
    return button


def _fold(text: str, opened: bool) -> QToolButton:
    fold = QToolButton()
    fold.setCheckable(True)
    fold.setChecked(opened)
    fold.setAutoRaise(True)
    fold.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    fold.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    fold.setArrowType(Qt.ArrowType.DownArrow if opened else Qt.ArrowType.RightArrow)
    fold.setText(text)
    return fold


def _scroll(inner: QWidget, tallest: int) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.setMaximumHeight(tallest)
    area.setWidget(inner)
    return area


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.hide()
            widget.deleteLater()


class RulingsSection(QWidget):
    """「谁压着谁 · 这台还没裁定的（n）」: the pending pairs, their buttons, and what was ruled."""

    #: ``(pairs, ruling, classes)``: rule these ``(above, below)`` pairs (one op);
    #: ``classes`` is the class pair of a batch line, ``None`` for a single row.
    sigRule = Signal(list, str, object)
    #: ``(above, below)``: take that ruling away (「撤销」).
    sigClear = Signal(str, str)
    #: The section was folded (``False``) or unfolded (``True``) by the annotator.
    sigToggled = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None, opened: bool = True) -> None:
        super().__init__(parent)
        self.toggle = _fold(RULINGS_TITLE.format(n=0), opened)
        self.toggle.setStyleSheet("font-weight: bold")
        self.toggle.toggled.connect(self._on_toggled)
        self.body = QWidget()
        self.empty = QLabel(NONE_PENDING)
        self.empty.setStyleSheet("color: gray")
        self.pending = QWidget()
        self.pending_layout = QVBoxLayout(self.pending)
        self.pending_layout.setContentsMargins(0, 0, 0, 0)
        self.pending_layout.setSpacing(1)
        self.pending_area = _scroll(self.pending, PENDING_MAX_HEIGHT)
        self.ruled_toggle = _fold(RULED_TITLE.format(k=0), False)
        self.ruled_toggle.toggled.connect(self._on_ruled_toggled)
        self.ruled = QWidget()
        self.ruled_layout = QVBoxLayout(self.ruled)
        self.ruled_layout.setContentsMargins(0, 0, 0, 0)
        self.ruled_layout.setSpacing(1)
        self.ruled_area = _scroll(self.ruled, RULED_MAX_HEIGHT)
        self.ruled_area.setVisible(False)
        for widget in (self, self.toggle, self.empty, self.ruled_toggle):
            widget.setToolTip(RULINGS_TIP)
        inner = QVBoxLayout(self.body)
        inner.setContentsMargins(6, 0, 0, 0)
        inner.setSpacing(2)
        inner.addWidget(self.empty)
        inner.addWidget(self.pending_area)
        inner.addWidget(self.ruled_toggle)
        inner.addWidget(self.ruled_area)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toggle)
        layout.addWidget(self.body)
        self.body.setVisible(opened)
        #: What is on show (tests, the report): the candidates, the batch lines'
        #: class pairs, and the ruled pairs, each with its buttons.
        self.candidates: list[Candidate] = []
        self.row_buttons: list[tuple[Candidate, QPushButton, QPushButton]] = []
        self.batch_buttons: list[tuple[tuple[str, str], list, QPushButton, QPushButton]] = []
        self.ruled_buttons: list[tuple[tuple[str, str], QPushButton]] = []
        self._row_labels: list[QLabel] = []
        self._ruled_labels: list[QLabel] = []

    # ------------------------------------------------------------ folding
    def is_open(self) -> bool:
        """Is the section unfolded?"""
        return self.toggle.isChecked()

    def set_open(self, opened: bool) -> None:
        """Fold or unfold it without telling (the window restoring the setting)."""
        blocked = self.toggle.blockSignals(True)
        self.toggle.setChecked(bool(opened))
        self.toggle.blockSignals(blocked)
        self._show_open(bool(opened))

    def _show_open(self, opened: bool) -> None:
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if opened else Qt.ArrowType.RightArrow)
        self.body.setVisible(opened)

    def _on_toggled(self, opened: bool) -> None:
        self._show_open(bool(opened))
        self.sigToggled.emit(bool(opened))

    def _on_ruled_toggled(self, opened: bool) -> None:
        self.ruled_toggle.setArrowType(Qt.ArrowType.DownArrow if opened
                                       else Qt.ArrowType.RightArrow)
        self.ruled_area.setVisible(bool(opened))

    # ------------------------------------------------------------- content
    def set_rulings(self, candidates: list[Candidate],
                    ruled: Mapping[tuple[str, str], Mapping]) -> None:
        """Show the frame's ``candidates`` (in order) and the desktop's ``ruled`` rows."""
        self.candidates = list(candidates)
        self.toggle.setText(RULINGS_TITLE.format(n=len(self.candidates)))
        self.ruled_toggle.setText(RULED_TITLE.format(k=len(ruled)))
        _clear(self.pending_layout)
        self.row_buttons, self.batch_buttons, self._row_labels = [], [], []
        for cand in self.candidates:
            self._add_row(cand)
        for classes, group in class_pairs(self.candidates):
            self._add_batch(classes, group)
        self.pending_layout.addStretch(1)
        self.empty.setVisible(not self.candidates)
        self.pending_area.setVisible(bool(self.candidates))
        _clear(self.ruled_layout)
        self.ruled_buttons, self._ruled_labels = [], []
        for (above, below), row in ruled.items():
            self._add_ruled(above, below, row)
        self.ruled_layout.addStretch(1)

    @staticmethod
    def _line(layout, text: str, tip: str, buttons: list[QPushButton],
              bold: bool = False) -> QLabel:
        """One line into ``layout``: the words (tooltip ``tip``), then the buttons."""
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setMinimumWidth(40)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        label.setToolTip(tip)
        if bold:
            label.setStyleSheet("color: #2a5d9f")
        row.setToolTip(tip)
        box.addWidget(label, 1)
        for button in buttons:
            box.addWidget(button)
        layout.addWidget(row)
        return label

    def _add_row(self, cand: Candidate) -> None:
        tip = f"{cand.above_class} → {cand.below_class}：{cand.reason}"
        blocks = _button(RULING_TEXT[BLOCKS], "挡着：拆下面那件之前必须先拿走上面那件")
        free = _button(RULING_TEXT[NOT_BLOCKS], "不挡：只是画面上重叠，下面那件拿得出来")
        pairs = [(cand.above, cand.below)]
        blocks.clicked.connect(partial(self._emit_rule, pairs, BLOCKS, None))
        free.clicked.connect(partial(self._emit_rule, pairs, NOT_BLOCKS, None))
        self._row_labels.append(self._line(self.pending_layout, candidate_text(cand), tip,
                                           [blocks, free]))
        self.row_buttons.append((cand, blocks, free))

    def _add_batch(self, classes: tuple[str, str], group: list[Candidate]) -> None:
        text = BATCH_TEXT.format(above=classes[0], below=classes[1], n=len(group))
        tip = "这一帧上这一类对的每一对都这样裁定（一次 Ctrl+Z 全部撤回）：" + "、".join(
            f"{c.above} → {c.below}" for c in group)
        blocks = _button(RULING_TEXT[BLOCKS], tip)
        free = _button(RULING_TEXT[NOT_BLOCKS], tip)
        pairs = [(c.above, c.below) for c in group]
        blocks.clicked.connect(partial(self._emit_rule, pairs, BLOCKS, classes))
        free.clicked.connect(partial(self._emit_rule, pairs, NOT_BLOCKS, classes))
        self._line(self.pending_layout, text, tip, [blocks, free], bold=True)
        self.batch_buttons.append((classes, pairs, blocks, free))

    def _add_ruled(self, above: str, below: str, row: Mapping) -> None:
        undo = _button(UNDO_RULING, "去掉这条裁定（这一对会回到上面，能重新裁定；Ctrl+Z 可撤回）")
        undo.clicked.connect(partial(self._emit_clear, above, below))
        tip = f"{row.get('by') or '-'} · {row.get('ts') or '-'}"
        self._ruled_labels.append(self._line(self.ruled_layout, ruled_text(above, below, row),
                                             tip, [undo]))
        self.ruled_buttons.append(((above, below), undo))

    def _emit_rule(self, pairs: list, ruling: str, classes, *_args) -> None:
        self.sigRule.emit(list(pairs), ruling, classes)

    def _emit_clear(self, above: str, below: str, *_args) -> None:
        self.sigClear.emit(above, below)

    def texts(self) -> list[str]:
        """The pending lines as shown, rows then batch lines (tests, the report)."""
        return [label.text() for label in self._row_labels] + [
            BATCH_TEXT.format(above=c[0], below=c[1], n=len(p))
            for c, p, _b, _f in self.batch_buttons]

    def ruled_texts(self) -> list[str]:
        """The 「已裁定」 lines as shown."""
        return [label.text() for label in self._ruled_labels]


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
    """The dock's content: 「谁压着谁」 on top (W3c), the order list, the note box taking the rest."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.rulings = RulingsSection()
        self.order = OcclusionList()
        self.note_box: Optional[FrameNoteBox] = None
        self.split = QSplitter(Qt.Orientation.Vertical)
        self.split.setChildrenCollapsible(False)
        self.split.addWidget(self.order)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.rulings)
        layout.addWidget(self.split, 1)

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
