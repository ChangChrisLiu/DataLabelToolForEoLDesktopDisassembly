"""「这一帧的说明」 in the session (task U35): read, write and undo a frame's note.

The Review panel's box (:mod:`tda.ui.panels.frame_note`) shows the note of the
frame on screen; 「保存」 / 「清除」 -- and leaving the frame with the box
changed -- end in :meth:`NotesMixin.set_frame_note`, which writes through
:func:`tda.core.frame_notes.set_note` (the row and its ``op_log`` row) and
pushes one ``set_frame_note`` op onto the window's undo stack: ``Ctrl+Z``
takes it back like any other edit.

A note is no input of the compiler, so nothing here recompiles, signs or
queues anything, and the frame's status does not move.  What changes is told
on :attr:`~tda.ui.session.AnnotationSession.sigNoteChanged` (the frame key):
the box, the task card's 「说明：…」 line and the timeline's ✎ follow it.
"""
from __future__ import annotations

from typing import Optional

from tda.core.frame_notes import OP_KIND, apply_note, set_note
from tda.core.model import FrameKey
from tda.ui.commands import Op

__all__ = ["NotesMixin"]


class NotesMixin:
    """Frame notes for :class:`tda.ui.session.AnnotationSession`."""

    #: ``{(desktop, view): steps with a note}``, read once per view.
    _noted: Optional[tuple[tuple[int, str], set[int]]] = None

    def _note_key(self, step: Optional[int]) -> Optional[FrameKey]:
        if not self.is_open:
            return None
        return FrameKey(self.desktop, int(self._step if step is None else step), self.view)

    def frame_note(self, step: Optional[int] = None,
                   key: Optional[FrameKey] = None) -> Optional[dict]:
        """The note of ``step`` of the open view (default: the frame on screen), or ``None``.

        ``key`` names a frame of any view instead.
        """
        key = key if key is not None else self._note_key(step)
        return None if key is None else self.db.frame_note(key)

    def noted_steps(self) -> set[int]:
        """The steps of the open view that carry a note (the timeline's ✎)."""
        if not self.is_open:
            return set()
        here = (int(self.desktop), str(self.view))
        if self._noted is None or self._noted[0] != here:
            self._noted = (here, self.db.noted_steps(*here))
        return set(self._noted[1])

    def set_frame_note(self, text: Optional[str], step: Optional[int] = None,
                       key: Optional[FrameKey] = None) -> Optional[FrameKey]:
        """Write the note of a frame (``""`` clears it); the frame, or ``None`` if unchanged.

        ``step`` is a step of the open view (default: the frame on screen);
        ``key`` names a frame of any view instead -- a note left behind by a
        move the window did not see coming.  The op goes onto the undo stack
        only when it is about the open view: the stack is that view's.
        """
        key = key if key is not None else self._note_key(step)
        if key is None:
            return None
        written = set_note(self.db, key, text, self.annotator)
        if written is None:
            return None
        if self.is_open and (key.desktop, key.view) == (self.desktop, self.view):
            self.undo_stack.push(Op(OP_KIND, *written), apply=False)
            self._refresh_dirty()
        self._note_changed(key)
        return key

    def _apply_note_op(self, payload: dict) -> None:
        """Both directions of a ``set_frame_note`` op on the undo stack."""
        self._note_changed(apply_note(self.db, payload))

    def _note_changed(self, key: FrameKey) -> None:
        self._noted = None
        self.sigNoteChanged.emit(key)
