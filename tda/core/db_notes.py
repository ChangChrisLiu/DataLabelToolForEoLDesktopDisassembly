"""「这一帧的说明」: the annotator's note on one frame, as rows (task U35).

``frame_note`` arrived with schema version 6: one row per ``(desktop, view,
step)`` -- multi-line text the annotator typed in Review about that frame (what
is odd on it, what they decided, what the picture shows), who wrote it and
when.  No row means no note; an empty text is never stored.

The note is no input of the compiler: writing one recompiles nothing, signs
nothing and moves no frame status.  It is keyed by the step like
``frame_override``, so ``load-index`` moves it with its photo
(:mod:`tda.core.db_moves`), and ``import-logs --force`` -- which rewrites the
step table, actions and instances -- never reaches it.  The undoable write
(one ``set_frame_note`` op with its inverse) is :mod:`tda.core.frame_notes`;
this module is the rows only.
"""
from __future__ import annotations

from typing import Optional

from tda.core.model import FrameKey

__all__ = ["FrameNotesMixin", "NOTE_TABLE"]

#: The table (schema version 6).
NOTE_TABLE = "frame_note"

_SELECT = "SELECT desktop, view, step, text, author, updated_at FROM frame_note"


def _row(r) -> dict:
    return {"desktop": int(r["desktop"]), "view": str(r["view"]), "step": int(r["step"]),
            "text": str(r["text"]), "author": r["author"], "updated_at": r["updated_at"]}


class FrameNotesMixin:
    """``frame_note`` for :class:`tda.core.db.Db`."""

    def frame_note(self, key: FrameKey) -> Optional[dict]:
        """The note of one frame (``desktop, view, step, text, author, updated_at``), or ``None``."""
        r = self.conn.execute(f"{_SELECT} WHERE desktop=? AND view=? AND step=?",
                              (key.desktop, key.view, key.step)).fetchone()
        return None if r is None else _row(r)

    def put_frame_note(self, key: FrameKey, text: str, author: Optional[str],
                       updated_at: Optional[str]) -> None:
        """Store the note of one frame as given (the whole row).

        An empty ``text`` is refused: "no note" is no row
        (:meth:`delete_frame_note`).
        """
        if not str(text or "").strip():
            raise ValueError("an empty note is no note: delete the row instead")
        with self._tx():
            self._ensure_desktop(key.desktop)
            self.conn.execute(
                "INSERT INTO frame_note(desktop, view, step, text, author, updated_at) "
                "VALUES(?, ?, ?, ?, ?, ?) ON CONFLICT(desktop, view, step) DO UPDATE SET "
                "text=excluded.text, author=excluded.author, updated_at=excluded.updated_at",
                (key.desktop, key.view, key.step, str(text), author, updated_at))

    def delete_frame_note(self, key: FrameKey) -> None:
        """Drop the note of one frame; nothing when it has none."""
        with self._tx():
            self.conn.execute("DELETE FROM frame_note WHERE desktop=? AND view=? AND step=?",
                              (key.desktop, key.view, key.step))

    def frame_notes(self, desktop: int, view: Optional[str] = None) -> list[dict]:
        """Every note of one desktop (of one view), by view, then step."""
        if view is None:
            rows = self.conn.execute(f"{_SELECT} WHERE desktop=? ORDER BY view, step",
                                     (desktop,)).fetchall()
        else:
            rows = self.conn.execute(f"{_SELECT} WHERE desktop=? AND view=? ORDER BY step",
                                     (desktop, view)).fetchall()
        return [_row(r) for r in rows]

    def noted_steps(self, desktop: int, view: str) -> set[int]:
        """The steps of one view that carry a note: what the timeline marks."""
        return {int(r[0]) for r in self.conn.execute(
            "SELECT step FROM frame_note WHERE desktop=? AND view=?", (desktop, view))}

    def note_counts(self) -> dict[int, dict[str, int]]:
        """``{desktop: {view: n}}`` of the notes the database holds (``cli status``)."""
        out: dict[int, dict[str, int]] = {}
        for r in self.conn.execute(
                "SELECT desktop, view, COUNT(*) AS n FROM frame_note GROUP BY desktop, view"):
            out.setdefault(int(r["desktop"]), {})[str(r["view"])] = int(r["n"])
        return out
