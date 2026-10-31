"""「谁压着谁」: the user's rulings on overlapping pairs, as rows and as one undoable op (task W3c).

``cover_ruling`` arrived with schema version 7: one row per ``(desktop, above,
below)`` -- on this desktop, does the part ``above`` (which paints over
``below`` on a confirmed frame) **block** ``below`` (``blocks``) or not
(``not_blocks``)?  Who said it and when (``by``, ``ts``), on which frame of
which view (``view``, ``step``) and an optional ``note``.  No row means no
ruling yet.  The table's DDL is fixed: W3a's ``cli pixel-edges`` reads it
by these very column names and turns ``blocks`` into imported ``covered_by``
edges.

The rulings are no input of the compiler: writing one recompiles nothing,
signs nothing, moves no frame status and touches no ``relation`` row.

**The op.**  Every change is one ``op_log`` row of kind ``cover_ruling``
(:data:`OP_KIND`) whose payload and inverse are two *states* of the same rows
-- ``{"desktop", "rows": [{"above", "below", "exists", "ruling", "by", "ts",
"view", "step", "note"}]}`` -- the way the window's other ops are written: the
inverse puts back the previous row byte for byte, or deletes the row a first
ruling made.  A batch (「同类对本帧全部」) is one op over several rows.
:meth:`CoverRulingsMixin.apply_cover_ruling_op` is both directions of the undo
stack.  The DDL is in ``schema.sql``.
"""
from __future__ import annotations

from typing import Iterable, Optional

from tda.core import dbrows as R

__all__ = ["CoverRulingsMixin", "OP_KIND", "RULINGS", "RULING_TABLE", "ruling_state"]

#: The table (schema version 7).
RULING_TABLE = "cover_ruling"
#: The op one change logs (``tda.ui.commands.KINDS`` lists it too).
OP_KIND = "cover_ruling"
#: What ``ruling`` may say (the table's CHECK).
RULINGS = ("blocks", "not_blocks")

_COLUMNS = ("ruling", "by", "ts", "view", "step", "note")
_SELECT = ('SELECT desktop, above, below, ruling, "by", ts, view, step, note '
           "FROM cover_ruling")


def _row(r) -> dict:
    return {"desktop": int(r["desktop"]), "above": str(r["above"]), "below": str(r["below"]),
            "ruling": str(r["ruling"]), "by": r["by"], "ts": r["ts"], "view": r["view"],
            "step": None if r["step"] is None else int(r["step"]), "note": r["note"]}


def ruling_state(above: str, below: str, row: Optional[dict]) -> dict:
    """One row (or none) of the pair as an op's state."""
    state = {"above": str(above), "below": str(below), "exists": row is not None}
    for name in _COLUMNS:
        state[name] = None if row is None else row.get(name)
    return state


class CoverRulingsMixin:
    """``cover_ruling`` for :class:`tda.core.db.Db`."""

    def cover_ruling(self, desktop: int, above: str, below: str) -> Optional[dict]:
        """The ruling on one ordered pair, or ``None``."""
        r = self.conn.execute(f"{_SELECT} WHERE desktop=? AND above=? AND below=?",
                              (int(desktop), str(above), str(below))).fetchone()
        return None if r is None else _row(r)

    def cover_rulings(self, desktop: int) -> dict[tuple[str, str], dict]:
        """Every ruling of one desktop, ``{(above, below): row}``, in key order."""
        rows = self.conn.execute(f"{_SELECT} WHERE desktop=? ORDER BY above, below",
                                 (int(desktop),)).fetchall()
        return {(str(r["above"]), str(r["below"])): _row(r) for r in rows}

    def _write_ruling_state(self, desktop: int, state: dict) -> None:
        """Make the row of ``state``'s pair look like ``state``; ``exists: False`` removes it."""
        keys = (int(desktop), str(state["above"]), str(state["below"]))
        if state.get("exists"):
            self.conn.execute(
                'INSERT INTO cover_ruling(desktop, above, below, ruling, "by", ts, view, step, '
                "note) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(desktop, above, below) "
                'DO UPDATE SET ruling=excluded.ruling, "by"=excluded."by", ts=excluded.ts, '
                "view=excluded.view, step=excluded.step, note=excluded.note",
                keys + tuple(state.get(name) for name in _COLUMNS))
        else:
            self.conn.execute("DELETE FROM cover_ruling WHERE desktop=? AND above=? AND below=?",
                              keys)

    def _log_rulings(self, desktop: int, view: Optional[str], after: list[dict],
                     before: list[dict], by: Optional[str]) -> tuple[dict, dict]:
        payload = {"desktop": int(desktop), "rows": after}
        inverse = {"desktop": int(desktop), "rows": before}
        with self.transaction():
            self._ensure_desktop(int(desktop))
            for state in after:
                self._write_ruling_state(desktop, state)
            self.log_op(int(desktop), str(view or ""), OP_KIND, payload, inverse, str(by or ""))
        return payload, inverse

    def set_cover_rulings(self, desktop: int, pairs: Iterable[tuple[str, str]], ruling: str, *,
                          view: Optional[str], step: Optional[int], by: Optional[str],
                          note: Optional[str] = None) -> Optional[tuple[dict, dict]]:
        """Rule every pair of ``pairs`` the same way, as **one** op; ``(payload, inverse)``.

        Upserts; a pair that already carries exactly this ruling (and note)
        is left alone, and ``None`` -- nothing written, nothing logged -- when
        that is every pair.  One transaction: the rows and the ``op_log`` row.
        """
        if ruling not in RULINGS:
            raise ValueError(f"a ruling is one of {RULINGS}, not {ruling!r}")
        stamp = R.now_iso()
        after, before, seen = [], [], set()
        for above, below in pairs:
            pair = (str(above), str(below))
            if pair in seen or pair[0] == pair[1]:
                continue
            seen.add(pair)
            old = self.cover_ruling(desktop, *pair)
            if old is not None and old["ruling"] == ruling and old.get("note") == note:
                continue
            new = {"ruling": ruling, "by": by, "ts": stamp, "view": view,
                   "step": None if step is None else int(step), "note": note}
            after.append(ruling_state(*pair, new))
            before.append(ruling_state(*pair, old))
        if not after:
            return None
        return self._log_rulings(desktop, view, after, before, by)

    def set_cover_ruling(self, desktop: int, above: str, below: str, ruling: str, *,
                         view: Optional[str], step: Optional[int], by: Optional[str],
                         note: Optional[str] = None) -> Optional[tuple[dict, dict]]:
        """Rule one ordered pair (upsert, one ``cover_ruling`` op); ``(payload, inverse)``.

        The inverse is the previous row, or its deletion; ``None`` when the
        pair already says exactly that.
        """
        return self.set_cover_rulings(desktop, [(above, below)], ruling, view=view, step=step,
                                      by=by, note=note)

    def clear_cover_ruling(self, desktop: int, above: str, below: str, *,
                           view: Optional[str] = None,
                           by: Optional[str] = None) -> Optional[tuple[dict, dict]]:
        """Take one ruling away (one op whose inverse is the row); ``None`` when there is none.

        ``view`` / ``by`` are the ``op_log`` row's (default: the ruling's own).
        """
        old = self.cover_ruling(desktop, above, below)
        if old is None:
            return None
        return self._log_rulings(desktop, view if view is not None else old.get("view"),
                                 [ruling_state(above, below, None)],
                                 [ruling_state(above, below, old)],
                                 by if by is not None else old.get("by"))

    def apply_cover_ruling_op(self, payload: dict) -> int:
        """Both directions of a ``cover_ruling`` op (the undo stack's); returns its desktop."""
        desktop = int(payload["desktop"])
        with self.transaction():
            self._ensure_desktop(desktop)
            for state in payload.get("rows") or []:
                self._write_ruling_state(desktop, state)
        return desktop

    def ruling_counts(self) -> dict[int, dict[str, int]]:
        """``{desktop: {ruling: n}}`` of every ruling the database holds."""
        out: dict[int, dict[str, int]] = {}
        for r in self.conn.execute(
                "SELECT desktop, ruling, COUNT(*) AS n FROM cover_ruling GROUP BY desktop, ruling"):
            out.setdefault(int(r["desktop"]), {})[str(r["ruling"])] = int(r["n"])
        return out
