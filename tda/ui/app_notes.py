"""「这一帧的说明」 in the main window (task U35).

Mixed into :class:`tda.ui.app.MainWindow`.  The Review panel's note box
(:mod:`tda.ui.panels.frame_note`) reports; this writes through the session --
one ``set_frame_note`` op, ``Ctrl+Z`` takes it back -- and says what happened
on the status line:

* 「保存」 / 「清除」 (:attr:`~tda.ui.panels.review.ReviewPanel.sigNoteSave`,
  :attr:`~tda.ui.panels.review.ReviewPanel.sigNoteClear`); the keyboard goes
  back to the canvas afterwards, where the Review keys work -- and so does
  ``Tab`` in the box (:attr:`~tda.ui.panels.review.ReviewPanel.sigNoteTabOut`);
* **leaving the frame** with the box changed and not saved: every move that
  goes through :meth:`~tda.ui.app_edit.EditMixin.leave_frame`, a confirmation
  (which steps the frame on), leaving Review and closing the window save it
  first (:meth:`NotesWindowMixin.flush_frame_note`) and the status line says
  so; any other way the frame changes, the panel hands the text over with the
  frame it was about (:attr:`~tda.ui.panels.review.ReviewPanel.sigNoteLeft`)
  and it is saved there.

The line promises ``Ctrl+Z`` only while the op is still on the undo stack:
opening another view or machine (and the other moves that re-open the
session) clears the stack after the note was saved, and the line says that
instead (review round 2, N1).  The task card's 「说明：…」 line and the
timeline's ✎ follow the session's ``sigNoteChanged`` themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QTimer

from tda.core.frame_notes import OP_KIND
from tda.core.model import FrameKey, is_component_view
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui.commands import Op

__all__ = ["NOTE_CLEARED", "NOTE_HOLD_MS", "NOTE_LEFT", "NOTE_LEFT_NO_UNDO", "NOTE_SAVED",
           "NOTE_SAVED_NO_UNDO", "NotesWindowMixin", "SavedNote"]

NOTE_SAVED = "{frame}的说明已保存（Ctrl+Z 撤销）"
NOTE_CLEARED = "{frame}的说明已清除（Ctrl+Z 撤销）"
#: Leaving a frame with the box changed: saved, and said.
NOTE_LEFT = "{frame}的说明改过没保存：离开时已自动保存（Ctrl+Z 撤销）"
#: ... when the move opened another view (or machine) and the undo stack went
#: with it: ``Ctrl+Z`` is not promised (review round 2, N1).
NOTE_LEFT_NO_UNDO = ("{frame}的说明改过没保存：离开时已自动保存（换了视角，撤销记录已清空；"
                     "要改回在复查里改）")
NOTE_SAVED_NO_UNDO = "{frame}的说明已保存（撤销记录已清空；要改回在复查里改）"
#: How long that line holds against the arrival's own lines.
NOTE_HOLD_MS = 3000


def _frame(key: FrameKey) -> str:
    """「第 12 帧」, or 「cmp_oak1 第 1013 张」 for a components-gallery shot."""
    if is_component_view(key.view):
        return f"{key.view} 第 {key.step} 张"
    return f"第 {key.step} 帧"


@dataclass(frozen=True)
class SavedNote:
    """A note the window wrote: its frame, and the op it pushed (``None``: none)."""

    key: FrameKey
    op: Optional[Op] = None


class NotesWindowMixin:
    """The note box's buttons, and the note saved on the way out of a frame."""

    def _init_notes(self) -> None:
        self.review.sigNoteSave.connect(self.on_note_save)
        self.review.sigNoteClear.connect(self.on_note_clear)
        self.review.sigNoteLeft.connect(self.on_note_left)
        self.review.sigNoteTabOut.connect(self.focus_canvas)

    def _undo_ops(self) -> list:
        stack = getattr(self.session, "undo_stack", None)
        return list(getattr(stack, "ops", None) or [])

    def _write_note(self, key: Optional[FrameKey], text: str) -> Optional[SavedNote]:
        """One write through the session; what was saved, or ``None`` when nothing changed.

        Raises what the session raises: the caller decides whether a failed
        write stops it (a slot's guard reports it; closing the window does not stop).
        """
        writer = getattr(self.session, "set_frame_note", None)
        if key is None or not callable(writer) or not compat.is_open(self.session):
            return None
        before = self._undo_ops()
        written = writer(text, key=key)
        if written is None:
            return None
        self.logger.info("frame note %s D%s/%s step %s (%d chars)",
                         "set" if text else "cleared", key.desktop, key.view, key.step,
                         len(text or ""))
        after = self._undo_ops()
        top = after[-1] if after else None
        pushed = (top if top is not None and top.kind == OP_KIND
                  and not (before and before[-1] is top) else None)
        return SavedNote(key, pushed)

    def note_undoable(self, saved: Optional[SavedNote]) -> bool:
        """Is the op that wrote ``saved`` still on the undo stack (``Ctrl+Z`` reaches it)?"""
        return bool(saved is not None and saved.op is not None
                    and any(op is saved.op for op in self._undo_ops()))

    @S.guard
    def on_note_save(self, key: FrameKey, text: str) -> None:
        """「保存」: write what the box holds."""
        saved = self._write_note(key, text)
        if saved is not None:
            self.report(self.note_saved_text(saved, left=False))
        self.focus_canvas()

    @S.guard
    def on_note_clear(self, key: FrameKey) -> None:
        """「清除」: the frame has no note any more."""
        if self._write_note(key, "") is not None:
            self.report(NOTE_CLEARED.format(frame=_frame(key)))
        self.focus_canvas()

    @S.guard
    def on_note_left(self, key: FrameKey, text: str) -> None:
        """The frame changed under a box typed in and not saved: save it, say so."""
        saved = self._write_note(key, text)
        if saved is not None:
            self.say_note_later(self.note_saved_text(saved, left=True))

    def say_note_later(self, line: str) -> None:
        """Say ``line`` from the event loop, after the line the move itself ends with.

        An arrival clears the status line (and a confirmation that stepped on
        writes its own): said then, the note's line comes after it -- joined to
        what is there -- and holds 3 s against what comes next.  While a
        confirmation is running (:meth:`hold_note_line`) the line waits for it:
        the confirmation steps the frame again, which would clear it, and
        takes it into its own line instead (:meth:`take_note_line`).
        """
        if not line:
            return
        self._note_line_pending = line
        # the window as context: a window deleted meanwhile drops the call
        QTimer.singleShot(0, self, self._say_pending_note)

    def _say_pending_note(self) -> None:
        if self.closed or getattr(self, "_note_line_held", False):
            return
        line = self.take_note_line()
        if not line:
            return
        now = self.status_message()
        self.report(line if not now or line in now else f"{now}；{line}",
                    hold_ms=NOTE_HOLD_MS)

    def take_note_line(self) -> str:
        """The note line waiting to be said (``""``: none); it is not said again."""
        line = getattr(self, "_note_line_pending", "") or ""
        self._note_line_pending = ""
        return line

    def hold_note_line(self, held: bool) -> None:
        """While ``held``, a waiting note line is not said (a confirmation is running).

        Released, a line nobody took is said from the event loop as usual.
        """
        self._note_line_held = bool(held)
        if not held and getattr(self, "_note_line_pending", ""):
            QTimer.singleShot(0, self, self._say_pending_note)

    def flush_frame_note(self) -> Optional[SavedNote]:
        """Save what the note box holds for its frame if it was changed.

        ``None`` when there was nothing to save.  Called before every way out
        of a frame the window controls, so the op lands on the undo stack of
        the view it is about; the caller says it (:meth:`note_saved_text`)
        once it knows whether the frame was left -- and whether the stack still
        holds the op.
        """
        review = getattr(self, "review", None)
        box = getattr(review, "note_box", None)
        if box is None or not box.is_dirty():
            return None
        return self._write_note(box.key(), box.text())

    def note_text_typed(self) -> str:
        """What the note box holds right now (``""``: no box) -- for an error that loses it."""
        box = getattr(getattr(self, "review", None), "note_box", None)
        return box.text() if box is not None else ""

    def note_saved_text(self, saved: Optional[SavedNote], left: bool = True) -> str:
        """The status line for a note the window saved (``""``: none).

        ``left``: the frame (or Review) was left, so it was saved on the way
        out; otherwise the plain 「已保存」.  ``Ctrl+Z`` is promised only while
        the op is on the undo stack (:meth:`note_undoable`).
        """
        if saved is None:
            return ""
        undoable = self.note_undoable(saved)
        if left:
            line = NOTE_LEFT if undoable else NOTE_LEFT_NO_UNDO
        else:
            line = NOTE_SAVED if undoable else NOTE_SAVED_NO_UNDO
        return line.format(frame=_frame(saved.key))

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
