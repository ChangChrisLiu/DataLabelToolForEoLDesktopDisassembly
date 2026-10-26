"""「这一帧的说明」 in the main window (task U35).

Mixed into :class:`tda.ui.app.MainWindow`.  The Review panel's note box
(:mod:`tda.ui.panels.frame_note`) reports; this writes through the session --
one ``set_frame_note`` op, ``Ctrl+Z`` takes it back -- and says what happened
on the status line:

* 「保存」 / 「清除」 (:attr:`~tda.ui.panels.review.ReviewPanel.sigNoteSave`,
  :attr:`~tda.ui.panels.review.ReviewPanel.sigNoteClear`); the keyboard goes
  back to the canvas afterwards, where the Review keys work;
* **leaving the frame** with the box changed and not saved: every move that
  goes through :meth:`~tda.ui.app_edit.EditMixin.leave_frame`, a confirmation
  (which steps the frame on), leaving Review and closing the window save it
  first (:meth:`NotesWindowMixin.flush_frame_note`) and the status line says
  so; any other way the frame changes, the panel hands the text over with the
  frame it was about (:attr:`~tda.ui.panels.review.ReviewPanel.sigNoteLeft`)
  and it is saved there.

The task card's 「说明：…」 line and the timeline's ✎ follow the session's
``sigNoteChanged`` themselves.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QTimer

from tda.core.frame_notes import OP_KIND
from tda.core.model import FrameKey, is_component_view
from tda.ui import app_compat as compat
from tda.ui import app_support as S

__all__ = ["NOTE_CLEARED", "NOTE_HOLD_MS", "NOTE_LEFT", "NOTE_SAVED", "NotesWindowMixin"]

NOTE_SAVED = "{frame}的说明已保存（Ctrl+Z 撤销）"
NOTE_CLEARED = "{frame}的说明已清除（Ctrl+Z 撤销）"
#: Leaving a frame with the box changed: saved, and said.
NOTE_LEFT = "{frame}的说明改过没保存：离开时已自动保存（Ctrl+Z 撤销）"
#: How long that line holds against the arrival's own lines.
NOTE_HOLD_MS = 3000


def _frame(key: FrameKey) -> str:
    """「第 12 帧」, or 「cmp_oak1 第 1013 张」 for a components-gallery shot."""
    if is_component_view(key.view):
        return f"{key.view} 第 {key.step} 张"
    return f"第 {key.step} 帧"


class NotesWindowMixin:
    """The note box's buttons, and the note saved on the way out of a frame."""

    def _init_notes(self) -> None:
        self.review.sigNoteSave.connect(self.on_note_save)
        self.review.sigNoteClear.connect(self.on_note_clear)
        self.review.sigNoteLeft.connect(self.on_note_left)

    def _write_note(self, key: Optional[FrameKey], text: str) -> Optional[FrameKey]:
        """One write through the session; the frame, or ``None`` when nothing changed."""
        writer = getattr(self.session, "set_frame_note", None)
        if key is None or not callable(writer) or not compat.is_open(self.session):
            return None
        written = writer(text, key=key)
        if written is not None:
            self.logger.info("frame note %s D%s/%s step %s (%d chars)",
                             "set" if text else "cleared", key.desktop, key.view, key.step,
                             len(text or ""))
        return written

    @S.guard
    def on_note_save(self, key: FrameKey, text: str) -> None:
        """「保存」: write what the box holds."""
        if self._write_note(key, text) is not None:
            self.report(NOTE_SAVED.format(frame=_frame(key)))
        self.focus_canvas()

    @S.guard
    def on_note_clear(self, key: FrameKey) -> None:
        """「清除」: the frame has no note any more."""
        if self._write_note(key, "") is not None:
            self.report(NOTE_CLEARED.format(frame=_frame(key)))
        self.focus_canvas()

    @S.guard
    def on_note_left(self, key: FrameKey, text: str) -> None:
        """The frame changed under a box typed in and not saved: save it, say so.

        Said from the event loop, after the arrival has cleared the status
        line, and held a moment against what the arrival says next.
        """
        if self._write_note(key, text) is None:
            return
        line = NOTE_LEFT.format(frame=_frame(key))
        # the window as context: a window deleted meanwhile drops the call
        QTimer.singleShot(0, self, lambda: None if self.closed else self.report(
            line, hold_ms=NOTE_HOLD_MS))

    def flush_frame_note(self) -> Optional[FrameKey]:
        """Save what the note box holds for its frame if it was changed; that frame.

        ``None`` when there was nothing to save.  Called before every way out
        of a frame the window controls, so the op lands on the undo stack of
        the view it is about; the caller says it (:meth:`note_saved_text`)
        once it knows whether the frame was left after all.
        """
        review = getattr(self, "review", None)
        box = getattr(review, "note_box", None)
        if box is None or not box.is_dirty():
            return None
        return self._write_note(box.key(), box.text())

    def note_saved_text(self, key: Optional[FrameKey], left: bool = True) -> str:
        """The status line for a note :meth:`flush_frame_note` saved (``""``: none).

        ``left``: the frame (or Review) was left, so it was saved on the way
        out; otherwise the plain 「已保存」.
        """
        if key is None:
            return ""
        return (NOTE_LEFT if left else NOTE_SAVED).format(frame=_frame(key))

    def note_op_text(self, undo: bool) -> str:
        """「撤销：第 12 帧的说明」 when the op ``Ctrl+Z`` / ``Ctrl+Y`` is about to replay is a note's."""
        stack = getattr(self.session, "undo_stack", None)
        if stack is None:
            return ""
        ops = stack.ops if undo else getattr(stack, "undone_ops", [])
        if not ops or ops[-1].kind != OP_KIND:
            return ""
        p = ops[-1].payload
        frame = _frame(FrameKey(int(p["desktop"]), int(p["step"]), str(p["view"])))
        return f"{'撤销' if undo else '重做'}：{frame}的说明 / {'undo' if undo else 'redo'}"
