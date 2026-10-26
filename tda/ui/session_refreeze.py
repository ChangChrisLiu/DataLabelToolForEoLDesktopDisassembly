"""「重签这一帧」 / 「重签全部」 in the session (task U32).

The two Review buttons ask the same thing of the truth service as ``cli
refreeze`` (:meth:`tda.core.truth.TruthService.refreeze`): write today's compile
into the confirmed rows of this view that differ from it only within the
tolerance.  The window plans first (:meth:`ResignMixin.refreeze_plan`, which
writes nothing), names the frame count in its question, and hands the plan
back (:meth:`ResignMixin.refreeze`): a frame whose inputs or rows did not move
since is written as planned, the rest are planned again -- one compilation
per frame, not two.

Nothing here touches the inputs, so the compilations the session keeps stay
good; only the review queues are re-read.  The background re-check is stopped
for the write, as :meth:`~tda.ui.session_review.ReviewMixin.refresh_all` stops
it, so one writer at a time touches the confirmed rows.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

__all__ = ["ResignMixin"]

log = logging.getLogger("tda.session")


def _empty() -> dict:
    return {"frames": [], "considered": 0, "frames_rewritten": 0, "rewritten": 0, "px": 0,
            "disputed": 0, "open_conflicts": 0, "missing": 0, "unsigned": 0, "moved": 0,
            "overlap_before": None, "overlap_after": None, "ops": [], "run": "",
            "dry_run": True}


class ResignMixin:
    """「重签」 for :class:`~tda.ui.session.AnnotationSession`."""

    def refreeze_plan(self, steps: Optional[Iterable[int]] = None) -> dict:
        """Which confirmed frames of this view 「重签」 would rewrite, and how; writes nothing.

        ``steps`` narrows it (``[current step]`` for 「重签这一帧」); the
        answer is :meth:`~tda.core.truth.TruthService.refreeze`'s, with the
        overlapping pixels measured.
        """
        if not self.is_open:
            return _empty()
        return self.truth.refreeze(self.desktop, self.view, steps=steps, dry_run=True,
                                   measure=True, annotator=self.annotator)

    def refreeze(self, plan: Optional[dict] = None,
                 steps: Optional[Iterable[int]] = None) -> dict:
        """Re-sign: the frames of ``plan`` (a :meth:`refreeze_plan` answer), or of ``steps``.

        Signed by this session's annotator; one ``refreeze_frame`` op per frame.
        Returns the truth service's totals.  Raises ``RuntimeError`` (writing
        nothing) when the background re-check will not stop.
        """
        if not self.is_open:
            return _empty()
        planned = None if plan is None else [f for f in plan.get("frames") or () if f.changes]
        running = self.sweeper.is_running
        if not self.sweeper.stop():
            log.error("refreeze aborted: the truth sweeper would not stop")
            raise RuntimeError("the truth sweeper is still running; cannot re-sign now")
        try:
            result = self.truth.refreeze(self.desktop, self.view, steps=steps,
                                         annotator=self.annotator, measure=True,
                                         plans=planned)
        finally:
            if running and self.sweeper_enabled:
                self.sweeper.open(self.desktop, self.view)
                self.sweeper.enqueue(self.db.rechecks(self.desktop, self.view))
        log.info("refreeze %s/%s: %s", self.desktop, self.view,
                 {k: v for k, v in result.items() if k != "frames"})
        # the inputs did not move: the kept compilations stay good, only the
        # queues (a re-signed label, a vis-review row) are read again
        self.review.invalidate()
        self._announce()
        return result
