"""「这一帧的说明」: the note box of Review mode (task U35).

Since task U44 it lives in the 「说明」 dock (:mod:`tda.ui.panels.notes_dock`),
under the frame's occlusion order, and takes the dock's spare height
(:meth:`FrameNoteBox.set_tall`); what it does is unchanged.

The annotator types what they want the next reader to know about the frame on
screen -- what is odd on it, what they decided, what the picture shows; part
keys are fine, nothing more is structured.  「保存」 writes it, 「清除」 takes
it off; leaving the frame with the box changed saves it too (the window does
that and says so on the status line).

The box decides nothing, like every panel: the buttons **report**
(:attr:`FrameNoteBox.sigSave`, :attr:`FrameNoteBox.sigClear`, with the frame
the box is showing) and the window writes through the session -- one
undoable op.  What it shows comes in through :meth:`FrameNoteBox.show_frame`.

Typing here is the point, so the window's keys do not fire while the box has
the focus (:func:`tda.ui.app_actions.blocks_shortcuts` holds for any text
box), and a letter, a digit, ``Space`` or ``Enter`` typed here is not called
a missed shortcut (:data:`tda.ui.app_actions.TYPING_PROPERTY`); a window key
that types nothing -- ``PgDn``, ``Ctrl+S``, an F key -- still says, once,
that the focus is in the note box and ``Tab`` gets back to the canvas
(review round 2, N4).  ``Tab`` / ``Shift+Tab`` in the box report
(:attr:`FrameNoteBox.sigTabOut`) and the window puts the keyboard on the
canvas; while the box has the focus its frame is highlighted, so stray
typing is seen.  No key of the window's map is added.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tda.core.frame_notes import NOTE_PREFIX, clean_text, first_line, when_text
from tda.core.model import FrameKey
from tda.ui.app_actions import TYPING_PROPERTY

__all__ = ["CLEAR_TEXT", "FrameNoteBox", "NoteLine", "PLACEHOLDER", "SAVE_TEXT", "TITLE",
           "UNSAVED", "note_line_text", "note_tooltip"]

TITLE = "这一帧的说明"
SAVE_TEXT, CLEAR_TEXT = "保存", "清除"
PLACEHOLDER = ("这一帧有什么要说明的：奇怪的地方、你怎么定的、画面里是什么（可以写零件键，"
               "如 screw.psu.03）。写完点「保存」；换帧时没保存的也会自动保存。")
TIP = ("写给之后读数据的人：这一帧的怪处、你的决定、画面内容。「保存」写入（Ctrl+Z 可撤），"
       "「清除」删掉；换帧时改过没保存的自动保存。在框里打字时快捷键不生效：按 Tab 回到画布。")
UNSAVED = "未保存"
#: The box's frame while it has the keyboard (round 2, N4): stray typing goes
#: here, so it has to be visible where the keys are going.
EDIT_STYLE = ("QPlainTextEdit { border: 1px solid #b8b8b8; } "
              "QPlainTextEdit:focus { border: 2px solid #2f7de1; background: #f5f9ff; }")


class FrameNoteBox(QWidget):
    """A text box, 「保存」 / 「清除」, and a line saying whose note it is."""

    #: 「保存」: ``(frame key, text)`` -- the window writes it.
    sigSave = Signal(object, str)
    #: 「清除」: the frame key -- the window takes the note off.
    sigClear = Signal(object)
    #: ``Tab`` / ``Shift+Tab`` in the box: the window puts the keyboard on the canvas.
    sigTabOut = Signal()

    #: Height of the text box: three or four lines, the queue keeps the rest.
    EDIT_HEIGHT = 74

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._key: Optional[FrameKey] = None
        self._stored = ""
        self._note: Optional[dict] = None

        self.title = QLabel(TITLE)
        self.title.setStyleSheet("font-weight: bold")
        self.status = QLabel("")
        self.status.setStyleSheet("color: gray; font-size: 8pt")
        self.edit = QPlainTextEdit()
        self.edit.setPlaceholderText(PLACEHOLDER)
        self.edit.setToolTip(TIP)
        self.edit.setTabChangesFocus(True)
        self.edit.setFixedHeight(self.EDIT_HEIGHT)
        self.edit.setProperty(TYPING_PROPERTY, True)
        self.edit.setStyleSheet(EDIT_STYLE)
        self.edit.setEnabled(False)          # no frame yet: nothing to write about
        self.edit.textChanged.connect(self._sync)
        self.edit.installEventFilter(self)
        self.save_button = QToolButton()
        self.clear_button = QToolButton()
        for button, text, slot in ((self.save_button, SAVE_TEXT, self._on_save),
                                   (self.clear_button, CLEAR_TEXT, self._on_clear)):
            button.setText(text)
            button.setToolTip(TIP)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(slot)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self.title)
        head.addWidget(self.status, 1)
        head.addWidget(self.save_button)
        head.addWidget(self.clear_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(head)
        layout.addWidget(self.edit)
        self._sync()

    def set_tall(self, tall: bool) -> None:
        """Let the text box take the height it is given (task U44: the 「说明」 dock).

        Under the Review panel's queues it was three or four lines; in the dock
        it is the thing the user asked to be bigger.  At least as tall as before.
        """
        if tall:
            self.edit.setMinimumHeight(self.EDIT_HEIGHT)
            self.edit.setMaximumHeight(16777215)          # QWIDGETSIZE_MAX
            self.edit.setSizePolicy(QSizePolicy.Policy.Expanding,
                                    QSizePolicy.Policy.Expanding)
        else:
            self.edit.setFixedHeight(self.EDIT_HEIGHT)

    # -- content ------------------------------------------------------------
    def show_frame(self, key: Optional[FrameKey], note: Optional[dict]) -> None:
        """Show the note of ``key`` (``None``: no frame), dropping what was typed."""
        self._key = key
        self._set_stored(note)
        blocked = self.edit.blockSignals(True)
        self.edit.setPlainText(self._stored)
        self.edit.blockSignals(blocked)
        self.edit.setEnabled(key is not None)
        self._sync()

    def note_changed(self, note: Optional[dict]) -> None:
        """The stored note of the frame shown changed (a save, an undo).

        Unchanged typing is replaced by the stored text; typing that differs
        from what was stored before stays, and is compared with the new one.
        """
        if self.is_dirty():
            self._set_stored(note)
            self._sync()
            return
        self.show_frame(self._key, note)

    def _set_stored(self, note: Optional[dict]) -> None:
        self._note = dict(note) if note else None
        self._stored = clean_text(note["text"]) if note else ""

    def key(self) -> Optional[FrameKey]:
        """The frame the box is about."""
        return self._key

    def text(self) -> str:
        """What the box holds, as it would be stored."""
        return clean_text(self.edit.toPlainText())

    def stored_text(self) -> str:
        """The note the frame has now (``""``: none)."""
        return self._stored

    def is_dirty(self) -> bool:
        """Has the annotator typed something the frame does not say yet?"""
        return self._key is not None and self.text() != self._stored

    # -- internals ----------------------------------------------------------
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API
        """``Tab`` / ``Shift+Tab`` in the box: back to the canvas, not to the next widget."""
        if (obj is self.edit and event.type() == QEvent.Type.KeyPress
                and event.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab)):
            self.sigTabOut.emit()
            return True
        return super().eventFilter(obj, event)

    def _sync(self) -> None:
        dirty = self.is_dirty()
        self.save_button.setEnabled(dirty)
        self.clear_button.setEnabled(self._key is not None
                                     and bool(self._stored or self.edit.toPlainText()))
        if dirty:
            self.status.setText(UNSAVED)
        elif self._note:
            self.status.setText(f"{self._note.get('author') or '-'} · "
                                f"{when_text(self._note.get('updated_at'))}")
        else:
            self.status.setText("")

    def _on_save(self) -> None:
        if self._key is not None and self.is_dirty():
            self.sigSave.emit(self._key, self.text())

    def _on_clear(self) -> None:
        if self._key is None:
            return
        self.edit.clear()
        if self._stored:          # nothing stored: 清除 only empties what was typed
            self.sigClear.emit(self._key)


# --------------------------------------------------------------------------- #
# the read-only line: the task card's header, outside Review
# --------------------------------------------------------------------------- #
def note_line_text(note: Optional[dict]) -> str:
    """「说明：<first line>…」, or ``""`` for a frame with no note."""
    if not note:
        return ""
    return NOTE_PREFIX + first_line(note.get("text") or "")


def note_tooltip(note: Optional[dict]) -> str:
    """The whole note, who wrote it and when, and where it is changed."""
    if not note:
        return ""
    return (f"{TITLE}（{note.get('author') or '-'}，{when_text(note.get('updated_at'))}）：\n"
            f"{note.get('text') or ''}\n\n在复查模式（Review）右侧的「说明」窗里改。")


class NoteLine(QLabel):
    """One line, 「说明：…」, elided to the width it is given; the tooltip has it all.

    It never decides how wide its dock is: the text is cut to the label, not
    the label stretched to the text.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._full = ""
        self.setWordWrap(False)
        self.setMinimumWidth(1)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setStyleSheet("QLabel { padding: 2px 4px; background: #fff6d8; "
                           "border-radius: 3px; }")
        self.setTextFormat(Qt.TextFormat.PlainText)

    def set_note(self, note: Optional[dict]) -> None:
        """Show ``note`` (``None``: nothing; the caller hides the line)."""
        self._full = note_line_text(note)
        self.setToolTip(note_tooltip(note))
        self._elide()

    def full_text(self) -> str:
        """The line before it is cut to the width."""
        return self._full

    def resizeEvent(self, event) -> None:  # noqa: D102 - Qt override
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        room = max(1, self.width() - 10)
        shown = self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, room)
        if shown != self.text():
            super().setText(shown)
