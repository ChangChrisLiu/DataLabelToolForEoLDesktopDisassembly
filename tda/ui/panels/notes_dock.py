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

Task W3f makes the rows readable: each says both parts' states at the frame
(「cover.01（打开）→ motherboard.01（装着）」) and its tooltip what the log did
to each (「日志：…」); **resting the pointer on any row** -- open, folded or
ruled -- reports the pair (:attr:`RulingsSection.sigHover`; ``("", "")`` when
it leaves) and the window outlines the two parts on the canvas.  Two folded
lists take the pairs nobody needs to rule: 「规则已间接覆盖（k）」 (the graph
orders them through another part; the chain in words, no button) and
「日志已证明不挡（k）」 (the log acted on the lower part with the upper one
still in the way; one 「不挡」 per row for the record, never 「挡着」).

Task W3g adds a third fold of the same kind, 「用户已定不挡（类级）（k）」: the
pairs whose class pair the user ruled 不挡 for the whole class (the
whitelist's ``not_blocks:``), the user's words on each row, no button.  An
open row whose class pair the user ruled 挡 is tagged 「用户已定：挡」 and keeps
both buttons.

Task W3h turns W3f's 「日志已证明不挡」 into 「日志里下面那件先动了（k）」 (subtitle
「不挡，或者层序画反了 —— 请看图定」): the same pairs (the log moved the lower part
while the upper one was still in the way), no longer a proof -- the user
decides with the picture whether the pair is 不挡 or the layer order is painted
the wrong way round.  Each row says which step moved the lower part, carries
the tag 「用户已定：挡 → 层序可能画反」 when the class pair is one the user ruled
挡 (either way round), and has **two** buttons: 「不挡」 (the ordinary ruling, as
before) and 「上下翻转」 (:attr:`RulingsSection.sigFlip`; the window's session
writes the pair override and settles the frozen frames).  Buttons only, no key.
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
    FLIP_TEXT,
    NOT_BLOCKS,
    RULING_TEXT,
    Candidate,
    candidate_text,
    chain_tip,
    class_not_blocks_text,
    class_pairs,
    indirect_text,
    proven_text,
    proven_tip,
    ruled_text,
)
from tda.ui.occlusion_order import FLOOR_NOTE, OrderLine, header_text, line_text
from tda.ui.panels.frame_note import FrameNoteBox

__all__ = ["BATCH_TEXT", "CLASS_NO_TIP", "CLASS_NO_TITLE", "DOCK_TITLE", "FLIP_TIP",
           "INDIRECT_TIP", "INDIRECT_TITLE", "NONE_PENDING",
           "ORDER_TITLE", "ORDER_TIP", "INSTANCE_ROLE", "NotesDock", "NotesPanel",
           "OcclusionList", "PROVEN_NOTE", "PROVEN_TIP", "PROVEN_TITLE", "RULED_TITLE",
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
               "「用户已定：挡」= 你已经按类别裁定这一类对挡着（悬停一行看你的原话），这一对仍可以单独改判；"
               "按类别裁定不挡的对折在「用户已定不挡（类级）」里，不再问。"
               "日志里下面那件先动了的对折在「日志里下面那件先动了」里：要么不挡，要么层序画反了，"
               "看图定——点「不挡」记一条裁定，或点「上下翻转」改层级。没有快捷键；"
               "Ctrl+Z 撤回最近一次裁定，「已裁定」里的「撤销」去掉任一条。"
               "鼠标停在任一行上，画布上框出这两件：橙框 = 压在上面的，蓝框 = 被压的（只是显示）。"
               "括号里是这一帧的状态；悬停看日志对两件做了什么。")
#: The two folded lists of pairs nobody needs to rule (task W3f).
INDIRECT_TITLE = "规则已间接覆盖（{k}）"
INDIRECT_TIP = ("图里已经有边通过别的零件定了这两件的先后（例：盖板挡着主板螺丝、螺丝拧住主板），"
                "不用再裁定。悬停看是哪几条边。")
#: Task W3h: W3f's 日志已证明不挡 became a question the user answers with the picture.
PROVEN_TITLE = "日志里下面那件先动了（{k}）"
PROVEN_NOTE = "不挡，或者层序画反了 —— 请看图定"
PROVEN_TIP = ("日志里下面那件被拆下 / 打开 / 挪开（或拧下 / 拔掉 / 松开）的时候，上面那件还在原处。"
              "这不是「不挡」的证明：拆解顺序是主要参考，也可能是层序画反了（例：翻下来压住硬盘的电源模块"
              "被画在硬盘之下）。看图定——「不挡」记一条普通裁定；「上下翻转」把下面那件翻到上面那件之上"
              "（这一视角这一段的改层级，一个 op），重查这一段已确认的帧，只接受这两件的新值并重签"
              "（容差以内得到像素的一方也重签）；Ctrl+Z 撤回覆盖并同样接受 / 重签这两件的行；"
              "所到的帧上有未裁决的冲突时不翻。「用户已定：挡 → 层序可能画反」= 这一类对"
              "你已按类别裁定挡着（正着或反着），多半是画反了。悬停看日志和你的原话。")
FLIP_TIP = ("把下面那件翻到上面那件之上（这一视角这一段）：写一条改层级，重查已确认帧，"
            "接受这两件的新值并重签；Ctrl+Z 撤回改层级并同样接受 / 重签")
#: Task W3g: the pairs whose class pair the user ruled 不挡 for the whole class.
CLASS_NO_TITLE = "用户已定不挡（类级）（{k}）"
CLASS_NO_TIP = ("这一类对你已经按类别裁定为不挡（只是拆的习惯顺序，或两件彼此独立；"
                "configs/covers_whitelist.yaml 的 not_blocks），不再问，也不会写成边。悬停看你的原话。")
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


class _HoverRow(QWidget):
    """One line of the section that says when the pointer comes onto it and leaves it (W3f)."""

    def __init__(self, on_hover: Optional[Callable[[bool], None]] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._on_hover = on_hover

    def enterEvent(self, event) -> None:  # noqa: D102 - Qt override
        super().enterEvent(event)
        if self._on_hover is not None:
            self._on_hover(True)

    def leaveEvent(self, event) -> None:  # noqa: D102 - Qt override
        super().leaveEvent(event)
        if self._on_hover is not None:
            self._on_hover(False)


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
    #: ``(above, below)``: the pointer rests on that pair's row; ``("", "")``:
    #: it left (or the rows were rebuilt under it).  Display only (task W3f).
    sigHover = Signal(str, str)
    #: ``(above, below)``: 「上下翻转」 on a 日志里下面那件先动了 row -- put ``below``
    #: above ``above`` for this segment (task W3h; the window's session writes).
    sigFlip = Signal(str, str)

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
        # task W3f: the two folded lists of pairs nobody needs to rule
        self.indirect_toggle, self.indirect, self.indirect_layout, self.indirect_area = \
            self._folded(INDIRECT_TITLE.format(k=0), INDIRECT_TIP)
        self.proven_toggle, self.proven, self.proven_layout, self.proven_area = \
            self._folded(PROVEN_TITLE.format(k=0), PROVEN_TIP)
        # task W3g: the class pairs the user ruled 不挡, the same kind of fold
        self.class_no_toggle, self.class_no, self.class_no_layout, self.class_no_area = \
            self._folded(CLASS_NO_TITLE.format(k=0), CLASS_NO_TIP)
        for widget in (self, self.toggle, self.empty, self.ruled_toggle):
            widget.setToolTip(RULINGS_TIP)
        inner = QVBoxLayout(self.body)
        inner.setContentsMargins(6, 0, 0, 0)
        inner.setSpacing(2)
        inner.addWidget(self.empty)
        inner.addWidget(self.pending_area)
        inner.addWidget(self.indirect_toggle)
        inner.addWidget(self.indirect_area)
        inner.addWidget(self.proven_toggle)
        inner.addWidget(self.proven_area)
        inner.addWidget(self.class_no_toggle)
        inner.addWidget(self.class_no_area)
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
        #: The folded lists (W3f): what is in them, and per 日志里下面那件先动了 row
        #: 「不挡」 and 「上下翻转」 (W3h).
        self.indirect_candidates: list[Candidate] = []
        self.proven_candidates: list[Candidate] = []
        self.proven_buttons: list[tuple[Candidate, QPushButton, QPushButton]] = []
        self._indirect_labels: list[QLabel] = []
        self._proven_labels: list[QLabel] = []
        #: The 用户已定不挡（类级） fold (W3g): what is in it; no buttons.
        self.class_no_candidates: list[Candidate] = []
        self._class_no_labels: list[QLabel] = []
        #: Every hoverable line: ``(pair, row widget)`` (tests drive enter / leave).
        self.hover_rows: list[tuple[tuple[str, str], QWidget]] = []
        #: The pair the pointer rests on, or ``None``.
        self.hovered: Optional[tuple[str, str]] = None

    def _folded(self, title: str, tip: str):
        """A folded list under its own toggle (closed until clicked)."""
        toggle = _fold(title, False)
        toggle.setToolTip(tip)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        area = _scroll(inner, RULED_MAX_HEIGHT)
        area.setVisible(False)

        def opened(on: bool) -> None:
            toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
            area.setVisible(bool(on))

        toggle.toggled.connect(opened)
        return toggle, inner, layout, area

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
                    ruled: Mapping[tuple[str, str], Mapping],
                    proven: Optional[list[Candidate]] = None,
                    indirect: Optional[list[Candidate]] = None,
                    class_not_blocks: Optional[list[Candidate]] = None) -> None:
        """Show the frame's ``candidates`` (in order) and the desktop's ``ruled`` rows.

        ``proven`` / ``indirect`` (task W3f; ``proven`` is the 日志里下面那件先动了
        bucket since task W3h) and ``class_not_blocks`` (task W3g) fill the
        folded lists; a fold with nothing in it is not shown.  Rebuilding under
        a resting pointer reports the hover gone (``sigHover("", "")``): its row
        is no more.
        """
        if self.hovered is not None:
            self.hovered = None
            self.sigHover.emit("", "")
        self.hover_rows = []
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
        self.indirect_candidates = list(indirect or [])
        _clear(self.indirect_layout)
        self._indirect_labels = []
        for cand in self.indirect_candidates:
            self._indirect_labels.append(self._line(
                self.indirect_layout, indirect_text(cand),
                f"{chain_tip(cand)}\n{cand.log}".strip(), [], hover=(cand.above, cand.below)))
        self.indirect_layout.addStretch(1)
        self.indirect_toggle.setText(INDIRECT_TITLE.format(k=len(self.indirect_candidates)))
        self._show_fold(self.indirect_toggle, self.indirect_area, bool(self.indirect_candidates))
        self.proven_candidates = list(proven or [])
        _clear(self.proven_layout)
        self.proven_buttons, self._proven_labels = [], []
        if self.proven_candidates:                  # the subtitle (W3h): what to do here
            self.proven_note = QLabel(PROVEN_NOTE)
            self.proven_note.setStyleSheet("color: gray")
            self.proven_note.setToolTip(PROVEN_TIP)
            self.proven_layout.addWidget(self.proven_note)
        for cand in self.proven_candidates:
            self._add_proven(cand)
        self.proven_layout.addStretch(1)
        self.proven_toggle.setText(PROVEN_TITLE.format(k=len(self.proven_candidates)))
        self._show_fold(self.proven_toggle, self.proven_area, bool(self.proven_candidates))
        self.class_no_candidates = list(class_not_blocks or [])
        _clear(self.class_no_layout)
        self._class_no_labels = []
        for cand in self.class_no_candidates:
            self._class_no_labels.append(self._line(
                self.class_no_layout, class_not_blocks_text(cand),
                f"{cand.above_class} → {cand.below_class}：{cand.reason}\n{cand.log}".strip(),
                [], hover=(cand.above, cand.below)))
        self.class_no_layout.addStretch(1)
        self.class_no_toggle.setText(CLASS_NO_TITLE.format(k=len(self.class_no_candidates)))
        self._show_fold(self.class_no_toggle, self.class_no_area,
                        bool(self.class_no_candidates))
        _clear(self.ruled_layout)
        self.ruled_buttons, self._ruled_labels = [], []
        for (above, below), row in ruled.items():
            self._add_ruled(above, below, row)
        self.ruled_layout.addStretch(1)

    @staticmethod
    def _show_fold(toggle: QToolButton, area: QScrollArea, any_rows: bool) -> None:
        toggle.setVisible(any_rows)
        area.setVisible(any_rows and toggle.isChecked())

    def _hover(self, pair: tuple[str, str], entered: bool) -> None:
        if entered:
            self.hovered = pair
            self.sigHover.emit(pair[0], pair[1])
        elif self.hovered == pair:
            self.hovered = None
            self.sigHover.emit("", "")

    def _line(self, layout, text: str, tip: str, buttons: list[QPushButton],
              bold: bool = False, hover: Optional[tuple[str, str]] = None) -> QLabel:
        """One line into ``layout``: the words (tooltip ``tip``), then the buttons.

        ``hover`` is the pair the line is about: resting the pointer on it
        reports the pair (:attr:`sigHover`, task W3f).
        """
        row = _HoverRow(None if hover is None else partial(self._hover, tuple(hover)))
        if hover is not None:
            self.hover_rows.append((tuple(hover), row))
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

    @staticmethod
    def _tip(cand: Candidate) -> str:
        """The sheet's hint, then the 日志 line (task W3f) when the log is known."""
        tip = f"{cand.above_class} → {cand.below_class}：{cand.reason}"
        return f"{tip}\n{cand.log}" if cand.log else tip

    def _add_row(self, cand: Candidate) -> None:
        blocks = _button(RULING_TEXT[BLOCKS], "挡着：拆下面那件之前必须先拿走上面那件")
        free = _button(RULING_TEXT[NOT_BLOCKS], "不挡：只是画面上重叠，下面那件拿得出来")
        pairs = [(cand.above, cand.below)]
        blocks.clicked.connect(partial(self._emit_rule, pairs, BLOCKS, None))
        free.clicked.connect(partial(self._emit_rule, pairs, NOT_BLOCKS, None))
        self._row_labels.append(self._line(self.pending_layout, candidate_text(cand),
                                           self._tip(cand), [blocks, free],
                                           hover=(cand.above, cand.below)))
        self.row_buttons.append((cand, blocks, free))

    def _add_proven(self, cand: Candidate) -> None:
        """A 日志里下面那件先动了 row: 「不挡」 (the ruling) and 「上下翻转」 (W3h), never 「挡着」."""
        free = _button(RULING_TEXT[NOT_BLOCKS], "记一条「不挡」：上面那件没挡着，只是画面上重叠")
        free.clicked.connect(partial(self._emit_rule, [(cand.above, cand.below)], NOT_BLOCKS,
                                     None))
        flip = _button(FLIP_TEXT, FLIP_TIP)
        flip.clicked.connect(partial(self._emit_flip, cand.above, cand.below))
        # the class hint, then the log's story (which ends with the 日志 line)
        tip = f"{cand.above_class} → {cand.below_class}：{cand.reason}\n{proven_tip(cand)}"
        self._proven_labels.append(self._line(self.proven_layout, proven_text(cand), tip,
                                              [free, flip], hover=(cand.above, cand.below)))
        self.proven_buttons.append((cand, free, flip))

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
                                             tip, [undo], hover=(above, below)))
        self.ruled_buttons.append(((above, below), undo))

    def _emit_rule(self, pairs: list, ruling: str, classes, *_args) -> None:
        self.sigRule.emit(list(pairs), ruling, classes)

    def _emit_clear(self, above: str, below: str, *_args) -> None:
        self.sigClear.emit(above, below)

    def _emit_flip(self, above: str, below: str, *_args) -> None:
        self.sigFlip.emit(above, below)

    def texts(self) -> list[str]:
        """The pending lines as shown, rows then batch lines (tests, the report)."""
        return [label.text() for label in self._row_labels] + [
            BATCH_TEXT.format(above=c[0], below=c[1], n=len(p))
            for c, p, _b, _f in self.batch_buttons]

    def ruled_texts(self) -> list[str]:
        """The 「已裁定」 lines as shown."""
        return [label.text() for label in self._ruled_labels]

    def proven_texts(self) -> list[str]:
        """The 「日志里下面那件先动了」 lines as shown (W3f, W3h)."""
        return [label.text() for label in self._proven_labels]

    def indirect_texts(self) -> list[str]:
        """The 「规则已间接覆盖」 lines as shown (W3f)."""
        return [label.text() for label in self._indirect_labels]

    def class_no_texts(self) -> list[str]:
        """The 「用户已定不挡（类级）」 lines as shown (W3g)."""
        return [label.text() for label in self._class_no_labels]

    def row_tips(self) -> list[str]:
        """The open rows' tooltips, in order (W3f: the sheet's hint and the 日志 line)."""
        return [label.toolTip() for label in self._row_labels]


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
