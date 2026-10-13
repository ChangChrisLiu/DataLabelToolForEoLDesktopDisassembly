"""The queue of frozen frames still waiting to be compared against their inputs.

Spec 3.4 says a ``verified`` row is never silently overwritten: when the inputs
change, the frame is recompiled and any disagreement is queued as a conflict.
Doing that for every frame an edit reaches is what made a commit cost seconds,
so the work moved off the GUI thread -- and the moment it is asynchronous, the
*request* has to outlive the process.  A crash between "the shape changed" and
"the frozen frame was checked" would otherwise lose a conflict for good, and a
conflict that is never raised is worse than a slow tool.

The table is therefore the queue itself, not a cache of it: one row per frame
waiting, added with ``INSERT OR IGNORE`` and removed when the check is done, so
two connections (the GUI's and the sweeper's) can work on it without ever
reading a value in order to write it back.
"""
from __future__ import annotations

from typing import Iterable, Optional

from tda.core import dbrows as R

__all__ = ["RecheckMixin"]


class RecheckMixin:
    """Pending truth re-checks, for a repository holding ``self.conn``."""

    def add_rechecks(self, desktop: int, view: str, steps: Iterable[int]) -> list[int]:
        """Queue these frames for a re-check; returns the steps queued.

        Asking again for a frame that is already queued does not add a second
        row -- it bumps that row's ``gen``. That is what makes a request landing
        while the sweeper is working on the very same frame safe: the sweeper
        clears the row only for the generation it read, so the newer request
        outlives the older one's clear (and outlives a crash, since it is a row
        rather than something held in memory).
        """
        wanted = sorted({int(s) for s in steps})
        if not wanted:
            return []
        with self._tx():
            self.conn.execute("INSERT OR IGNORE INTO desktop(id) VALUES(?)", (int(desktop),))
            self.conn.executemany(
                "INSERT INTO recheck_queue(desktop, step, view, requested_at, gen) "
                "VALUES(?, ?, ?, ?, 0) "
                "ON CONFLICT(desktop, step, view) DO UPDATE SET "
                "gen = gen + 1, requested_at = excluded.requested_at",
                [(int(desktop), step, str(view), R.now_iso()) for step in wanted],
            )
        return wanted

    def frozen_steps(self, desktop: int, view: str) -> list[int]:
        """Steps of one view a human has frozen: the frame, or any row in it.

        "Frozen" is the same notion :meth:`tda.core.truth.TruthService` works
        with, and it is one query so that a caller who needs to queue only
        *part* of a view (a pose re-cut reaches two segments, not a hundred
        frames) cannot end up asking a different question than
        :meth:`queue_rechecks_for_view` does.
        """
        rows = self.conn.execute(
            "SELECT step FROM frame WHERE desktop=? AND view=? AND review_status='verified' "
            "UNION "
            "SELECT step FROM compiled_mask WHERE desktop=? AND view=? AND status='verified'",
            (int(desktop), str(view), int(desktop), str(view)),
        ).fetchall()
        return sorted({int(r["step"]) for r in rows})

    def frozen_steps_in(self, desktop: int, view: str, steps: Iterable[int]) -> list[int]:
        """The steps among ``steps`` a human has frozen, ascending (task U7).

        :meth:`frozen_steps` narrowed to the frames one edit reaches, in **one**
        query however many there are: the frame's own ``review_status``, *or*
        any compiled row a human signed.  The second half is the one that
        matters: a confirmed frame that gained a part is demoted to
        ``needs_review`` but keeps every row the annotator signed, and those
        rows disagree with a re-trace exactly as before.  Asking only the flag
        -- what the edit path did until U7 -- left 102 such disagreements on
        D13/scan unqueued, to surface five at a time on each visit.
        """
        wanted = sorted({int(s) for s in steps})
        if not wanted:
            return []
        marks = ",".join("?" * len(wanted))
        rows = self.conn.execute(
            "SELECT step FROM frame WHERE desktop=? AND view=? AND review_status='verified' "
            f"AND step IN ({marks}) "
            "UNION "
            "SELECT step FROM compiled_mask WHERE desktop=? AND view=? AND status='verified' "
            f"AND step IN ({marks})",
            (int(desktop), str(view), *wanted, int(desktop), str(view), *wanted),
        ).fetchall()
        return sorted({int(r["step"]) for r in rows})

    def frozen_steps_to_recheck(self, desktop: int, view: str) -> list[int]:
        """Frozen steps of one view with no open conflict: what opening it owes (U7 round 2).

        :meth:`frozen_steps` minus every step that has an open conflict, in
        **one** query.  A frame with an open conflict has been compared
        already -- that comparison is what queued the conflict -- and the
        annotator's answer is what it is waiting for, not another look.  Every
        other frozen frame may hold a disagreement nobody looked for: before
        U7 the edit path re-checked only frames whose flag said ``verified``,
        so a confirmed frame demoted by 添加零件 kept its signed rows and a
        re-trace's conflicts there were never raised.  Queueing them all is
        cheap -- a frame whose digest is current costs the sweeper
        milliseconds -- and idempotent.
        """
        rows = self.conn.execute(
            "SELECT step FROM frame WHERE desktop=? AND view=? AND review_status='verified' "
            "UNION "
            "SELECT step FROM compiled_mask WHERE desktop=? AND view=? AND status='verified' "
            "EXCEPT "
            "SELECT step FROM conflict WHERE desktop=? AND view=? AND status='open'",
            (int(desktop), str(view)) * 3,
        ).fetchall()
        return sorted({int(r["step"]) for r in rows})

    def queue_rechecks_for_view(self, desktop: int, view: str) -> list[int]:
        """Queue **every frozen frame** of one view; returns the steps queued.

        What a write that changes the compiler's inputs wholesale has to call:
        deleting a pose segment, dropping its corners or ROI, purging a batch of
        imported draft shapes. Those do not go through the session's edit path,
        so nothing else would ever compare the frozen frames they reach against
        what the inputs now say -- and a conflict that is never raised is the
        one failure mode the queue exists to prevent (spec 3.4).

        "Frozen" is the same notion :meth:`tda.core.truth.TruthService` works
        with: the frame's own ``review_status``, or any compiled row a human
        signed. Queueing is idempotent, so a caller may be generous.
        """
        return self.add_rechecks(desktop, view, self.frozen_steps(desktop, view))

    def rechecks(self, desktop: int, view: Optional[str] = None) -> list[int]:
        """Logical steps still waiting for a re-check, ascending."""
        return [step for step, _gen in self.recheck_items(desktop, view)]

    def recheck_items(self, desktop: int, view: Optional[str] = None
                      ) -> list[tuple[int, int]]:
        """``(step, gen)`` for every frame still waiting, ascending by step.

        The generation is what a worker has to carry back to
        :meth:`clear_recheck`; reading the steps without it is only safe for
        display.
        """
        sql = "SELECT step, gen FROM recheck_queue WHERE desktop=?"
        args: list = [int(desktop)]
        if view is not None:
            sql += " AND view=?"
            args.append(str(view))
        return [(int(r["step"]), int(r["gen"]))
                for r in self.conn.execute(sql + " ORDER BY step", args)]

    def recheck_generation(self, desktop: int, view: str, step: int) -> Optional[int]:
        """The generation stamp of one queued frame, or ``None`` when it is not."""
        row = self.conn.execute(
            "SELECT gen FROM recheck_queue WHERE desktop=? AND step=? AND view=?",
            (int(desktop), int(step), str(view)),
        ).fetchone()
        return None if row is None else int(row["gen"])

    def clear_recheck(self, desktop: int, view: str, step: int,
                      gen: Optional[int] = None) -> bool:
        """Retire one re-check request; ``True`` when a row was actually removed.

        With ``gen`` the row goes only if it still carries that stamp, so a
        request that arrived while the frame was being checked survives. Without
        it the row goes whatever it says, which is only right for a caller that
        holds the write lock for the whole drain.
        """
        sql = "DELETE FROM recheck_queue WHERE desktop=? AND step=? AND view=?"
        args: list = [int(desktop), int(step), str(view)]
        if gen is not None:
            sql += " AND gen=?"
            args.append(int(gen))
        with self._tx():
            return self.conn.execute(sql, args).rowcount > 0
