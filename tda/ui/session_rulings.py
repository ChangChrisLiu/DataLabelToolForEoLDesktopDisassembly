"""「谁压着谁」 in the session (task W3c): read, write and undo the rulings of the open desktop.

The 「说明」 dock's section (:mod:`tda.ui.panels.notes_dock`) lists the
overlapping pairs of the frame on screen that still want a ruling
(:mod:`tda.ui.cover_rulings`); 「挡着」 / 「不挡」 -- one pair, or every pair of
one class pair on the frame -- end in :meth:`RulingsMixin.set_cover_rulings`,
which writes through :meth:`tda.core.db_rulings.CoverRulingsMixin.set_cover_rulings`
(the rows and their ``op_log`` row) and pushes one ``cover_ruling`` op onto
the window's undo stack: ``Ctrl+Z`` takes it back like any other edit, and
「撤销」 on a ruled pair is :meth:`RulingsMixin.clear_cover_ruling`, one op too.

A ruling is no input of the compiler: nothing here recompiles, signs or
queues anything, and the frame's status does not move.  What changed is told
on :attr:`~tda.ui.session.AnnotationSession.sigRulingsChanged` (the desktop).
"""
from __future__ import annotations

from typing import Iterable, Optional

from tda.core.db_rulings import OP_KIND
from tda.ui.commands import Op

__all__ = ["OP_KIND", "RulingsMixin"]


class RulingsMixin:
    """Cover rulings for :class:`tda.ui.session.AnnotationSession`."""

    def cover_rulings(self) -> dict:
        """Every ruling of the open desktop, ``{(above, below): row}`` (``{}``: none open)."""
        if not self.is_open:
            return {}
        return self.db.cover_rulings(int(self.desktop))

    def _push_ruling(self, written) -> bool:
        if written is None:
            return False
        self.undo_stack.push(Op(OP_KIND, *written), apply=False)
        self._refresh_dirty()
        self.sigRulingsChanged.emit(int(written[0]["desktop"]))
        return True

    def set_cover_rulings(self, pairs: Iterable[tuple[str, str]], ruling: str,
                          step: Optional[int] = None) -> bool:
        """Rule ``pairs`` of the open desktop the same way, as one op; ``False``: nothing changed.

        ``view`` is the open view and ``step`` the frame on screen (default),
        ``by`` the annotator.
        """
        if not self.is_open:
            return False
        written = self.db.set_cover_rulings(
            int(self.desktop), list(pairs), ruling, view=str(self.view),
            step=int(self._step if step is None else step), by=self.annotator)
        return self._push_ruling(written)

    def clear_cover_ruling(self, above: str, below: str) -> bool:
        """Take the ruling on ``(above, below)`` away (one op); ``False``: there was none."""
        if not self.is_open:
            return False
        written = self.db.clear_cover_ruling(int(self.desktop), above, below,
                                             view=str(self.view), by=self.annotator)
        return self._push_ruling(written)

    def _apply_ruling_op(self, payload: dict) -> None:
        """Both directions of a ``cover_ruling`` op on the undo stack."""
        self.sigRulingsChanged.emit(self.db.apply_cover_ruling_op(payload))
