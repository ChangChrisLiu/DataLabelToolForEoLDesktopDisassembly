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

**Task W3h: 「上下翻转」** (:meth:`RulingsMixin.flip_pair_layers`) is the one
gesture of the section that *is* an input of the compiler.  A row of the
dock's 「日志里下面那件先动了」 fold (W3f's 日志已证明不挡) may be a layer order
painted the wrong way round rather than a 不挡 -- the user decides with the
picture; the button writes the pair override ``below`` above ``above`` for the view's pose
segment through :func:`tda.ui.session_edit.commit_pair_override` -- one
``set_pair_override`` op, carrying a ``note`` naming the pair and the frames --
then runs the view's queued re-checks synchronously, closes the conflicts
**of the two parts only** with ``accept_new`` (the annotator asked for exactly
that), re-signs a frame the pass demoted, and runs U32's refreeze over the
frozen frames it reached, so the part that *gained* pixels within the 2 %
tolerance -- no conflict, no rewrite otherwise -- is re-signed too (spec 3.4
v1.31's hole).  It refuses, writing nothing, when the segment already holds
that override, when the picture's own order is a hand-set override, when the
reached frames hold an open conflict of any part (a conflict the flip did not
raise must not be accepted by it; the view's queued re-checks run first so an
older one surfaces here), or when the pair shares no frame.  The op on the
undo stack is :data:`FLIP_OP`: ``Ctrl+Z`` removes the override **and** settles
the two parts' rows the same way, so an undo leaves no conflict behind.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

from tda.core.db_rulings import OP_KIND
from tda.core.model import FrameKey
from tda.core.truth import StaleConflictError
from tda.core.truth_inputs import InputCache, pose_segment_of
from tda.ui import session_api as api
from tda.ui import session_edit as edit
from tda.ui.commands import Op
from tda.ui.cover_rulings import FLIP_NOTE
from tda.ui.session_api import SessionRefusal
from tda.ui.session_ops import frozen_among

__all__ = ["FLIP_ALREADY", "FLIP_BY_HAND", "FLIP_CONFLICT", "FLIP_NO_FRAMES", "FLIP_NO_VIEW",
           "FLIP_OP", "FLIP_SWEEPER", "OP_KIND", "RulingsMixin"]

log = logging.getLogger(__name__)

#: The undo-stack kind of 「上下翻转」 (task W3h): the pair override plus the rows it
#: settles, in both directions (the op-log row stays ``set_pair_override``).
FLIP_OP = "flip_pair_layers"
#: Why a flip was refused (task W3h); each is the status line's sentence.
FLIP_NO_VIEW = "没有翻：没有打开的视图"
FLIP_ALREADY = "没有翻：这一段已经把 {below} 翻到 {above} 之上了"
FLIP_BY_HAND = "没有翻：这一段是「改层级」手动把 {above} 放在 {below} 之上的，先撤掉那一条"
FLIP_NO_FRAMES = "没有翻：这一段里没有一帧两件都画了形状"
FLIP_CONFLICT = "没有翻：第 {step} 帧的 {instance} 还有未裁决的冲突（共 {n} 条），先在复查队列里处理"
FLIP_SWEEPER = "没有翻：后台复检还在跑，停不下来；稍后再试"
#: How many passes the accept loop makes over conflicts a stale accept re-queued.
FLIP_ROUNDS = 3
VERIFIED = "verified"


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

    # ------------------------------------------------------- 上下翻转 (W3h)
    def flip_pair_layers(self, above: str, below: str) -> dict:
        """Put ``below`` above ``above`` for this view's pose segment, and settle what it moves (W3h).

        See the module docstring.  Returns :func:`tda.ui.session_edit.commit_pair_override`'s
        result plus :meth:`_settle_flip`'s: ``frames`` (the steps the override
        reaches), ``frozen`` (the confirmed ones among them), ``accepted``
        (conflicts of the two parts closed with ``accept_new``), ``left_open``
        (``(step, instance)`` of conflicts on those frames still open afterwards
        -- another part's, which this never touches), ``resigned`` (frames the
        pass demoted and re-signed whole), ``verify_refused`` (``(step, reason)``
        of frames that refused), ``rewritten`` / ``rewritten_frames`` (rows the
        tolerance left behind, re-signed by U32's refreeze).  Raises
        :class:`~tda.ui.session_api.SessionRefusal` with the reason when nothing
        was written.  The op (:data:`FLIP_OP`) goes onto the undo stack:
        ``Ctrl+Z`` takes the override away and settles the two parts' rows the
        same way (:meth:`_apply_flip_op`), so no conflict is left behind.
        """
        if not self.is_open:
            raise SessionRefusal(FLIP_NO_VIEW)
        above, below = str(above), str(below)
        key = self.current()
        desktop, view = int(self.desktop), str(self.view)
        cache = InputCache()
        seg = pose_segment_of(self.db, key, cache)
        held = {(p.above, p.below) for p in self.db.pair_overrides(desktop, view, seg)}
        if (below, above) in held:
            raise SessionRefusal(FLIP_ALREADY.format(above=above, below=below))
        if (above, below) in held:
            raise SessionRefusal(FLIP_BY_HAND.format(above=above, below=below))
        steps = edit.pair_steps(self.db, self.tax, key, above, below, seg, cache)
        if not steps:
            raise SessionRefusal(FLIP_NO_FRAMES)
        reach = set(steps)
        note = FLIP_NOTE.format(below=below, above=above,
                                frames=" ".join(str(s) for s in steps))
        # one drain, not two racing over the same rows (refresh_all's rule)
        running = self.sweeper.is_running
        if not self.sweeper.stop():
            raise SessionRefusal(FLIP_SWEEPER)
        try:
            # re-checks queued earlier run first: a conflict they raise is not this
            # flip's to accept, so it refuses like any other standing one
            self.truth.run_pending_rechecks(desktop, view)
            standing = [(int(c["step"]), str(c["instance"] or ""))
                        for c in self.db.conflicts(desktop, view, open_only=True)
                        if int(c["step"]) in reach]
            if standing:
                step, instance = standing[0]
                raise SessionRefusal(FLIP_CONFLICT.format(step=step, instance=instance,
                                                          n=len(standing)))
            frozen = frozen_among(self.db, desktop, view, steps)
            status = self._frame_status(desktop, view, frozen)
            result = edit.commit_pair_override(
                self.db, self.truth, key, below, above, self.annotator,
                known=self._known_instances(), extra={"note": note, "flip": True})
            settled = self._settle_flip(desktop, view, reach, {above, below}, frozen, status)
        finally:
            if running and self.sweeper_enabled:
                self.sweeper.open(desktop, view)
                self.sweeper.enqueue(self.db.rechecks(desktop, view))
        result.update(settled)
        # the same payload / inverse as the pair override's op, under the kind
        # whose handler settles the rows both ways
        result["op"] = Op(FLIP_OP, result["op"].payload, result["op"].inverse)
        log.info("flip D%s %s seg %s: %s above %s on %d frame(s), %d frozen: %s", desktop, view,
                 seg, below, above, len(steps), len(frozen),
                 {k: settled[k] for k in ("accepted", "left_open", "resigned", "verify_refused",
                                          "rewritten")})
        self._after_edit(result)
        # the resolutions cleared those frames' digests: the sweeper stamps them again
        self._hand_to_sweeper(frozen)
        return result

    def _apply_flip_op(self, payload: dict) -> None:
        """Both directions of a :data:`FLIP_OP` on the undo stack (W3h).

        The override is written or removed as the payload says
        (:func:`tda.ui.session_edit.apply_pair_override`, which settles the
        frame in hand and queues the frozen ones), then the frozen frames are
        settled exactly as the press settled them (:meth:`_settle_flip`): the
        two parts' conflicts accepted, demoted frames re-signed, the rows the
        tolerance left behind re-signed.  An undo therefore puts the rows back
        where they were, signed by whoever undid, and leaves no conflict.
        """
        desktop, view = int(payload["desktop"]), str(payload["view"])
        parts = {str(payload["above"]), str(payload["below"])}
        steps = [int(s) for s in payload.get("steps") or []]
        frozen = frozen_among(self.db, desktop, view, steps)
        status = self._frame_status(desktop, view, frozen)
        running = self.sweeper.is_running
        if not self.sweeper.stop():
            raise RuntimeError(FLIP_SWEEPER)
        try:
            stats = edit.apply_pair_override(self.db, self.truth, payload, self._step)
            settled = self._settle_flip(desktop, view, set(steps), parts, frozen, status)
        finally:
            if running and self.sweeper_enabled:
                self.sweeper.open(desktop, view)
                self.sweeper.enqueue(self.db.rechecks(desktop, view))
        log.info("flip %s D%s %s: %s", "applied" if payload.get("exists") else "undone",
                 desktop, view, {k: settled[k] for k in ("accepted", "left_open", "resigned",
                                                        "verify_refused", "rewritten")})
        self.review.problems.update(stats.get("problems") or {})
        self._settled(stats.get("frame"))
        self._hand_to_sweeper(frozen)
        self._announce()

    def _frame_status(self, desktop: int, view: str, steps: Iterable[int]) -> dict[int, str]:
        """Each frame's ``review_status`` now (``""`` without a row)."""
        return {int(s): str((self.db.get_frame(FrameKey(desktop, int(s), view)) or {})
                            .get("review_status") or "") for s in steps}

    def _settle_flip(self, desktop: int, view: str, reach: set, parts: set,
                     frozen: list, status_before: dict) -> dict:
        """After the override moved: re-check, accept the two parts, re-sign (both directions).

        The view's queued re-checks run now; every open conflict of ``parts``
        on the ``reach`` frames is closed with ``accept_new``; a frame that was
        ``verified`` before and is not now is re-signed whole
        (:meth:`_resign_frames`); then U32's refreeze
        (:meth:`~tda.core.truth.TruthService.refreeze`) rewrites the frozen rows
        the tolerance left behind -- the part that *gained* pixels within 2 %
        raises no conflict and would otherwise keep its old mask, a hole
        between the two (spec 3.4 v1.31).  The sweeper is stopped by the caller.
        """
        self.truth.run_pending_rechecks(desktop, view)
        accepted, left = self._accept_flip_conflicts(desktop, view, reach, parts)
        resigned, refused = self._resign_frames(desktop, view, frozen, status_before)
        refrozen = (self.truth.refreeze(desktop, view, steps=frozen, annotator=self.annotator)
                    if frozen else {})
        return {"frames": sorted(reach), "frozen": list(frozen), "accepted": accepted,
                "left_open": left, "resigned": resigned, "verify_refused": refused,
                "rewritten": int(refrozen.get("rewritten") or 0),
                "rewritten_frames": int(refrozen.get("frames_rewritten") or 0)}

    def _accept_flip_conflicts(self, desktop: int, view: str, reach: set,
                               parts: set) -> tuple[int, list[tuple[int, str]]]:
        """``accept_new`` on every open conflict of ``parts`` on the ``reach`` frames.

        One compilation per frame is shared by its conflicts (``prepared``); a
        conflict a stale accept re-queued is taken in the next round
        (:data:`FLIP_ROUNDS`).  Returns ``(accepted, still open on those frames)``.
        """
        accepted = 0
        cache = InputCache()  # after the write: the segment's overrides changed
        for _round in range(FLIP_ROUNDS):
            mine = [c for c in self.db.conflicts(desktop, view, open_only=True)
                    if int(c["step"]) in reach and str(c["instance"] or "") in parts]
            if not mine:
                break
            by_step: dict[int, list] = {}
            for conflict in mine:
                by_step.setdefault(int(conflict["step"]), []).append(conflict)
            for step, items in sorted(by_step.items()):
                fkey = FrameKey(desktop, step, view)
                prepared = self.truth.compile(fkey, cache)
                for conflict in items:
                    try:
                        self.truth.resolve_conflict(int(conflict["id"]), api.RESOLVE_ACCEPT_NEW,
                                                    self.annotator, prepared=prepared,
                                                    cache=cache)
                        accepted += 1
                    except StaleConflictError as stale:   # re-queued: the next round
                        log.info("flip: conflict %s re-queued: %s", conflict["id"], stale)
                    except (KeyError, ValueError) as refused:
                        log.info("flip: conflict %s refused: %s", conflict["id"], refused)
        left = [(int(c["step"]), str(c["instance"] or ""))
                for c in self.db.conflicts(desktop, view, open_only=True)
                if int(c["step"]) in reach]
        return accepted, left

    def _resign_frames(self, desktop: int, view: str, frozen: Iterable[int],
                       status_before: dict) -> tuple[int, list[tuple[int, str]]]:
        """``verify_frame`` on every frame that was ``verified`` before and is not now.

        A frame already demoted before the flip (a part added, waiting for the
        annotator) stays demoted: the flip is not its confirmation.  Returns
        ``(re-signed, (step, reason) of the refused)``.
        """
        resigned, refused = 0, []
        for step in frozen:
            if status_before.get(int(step)) != VERIFIED:
                continue
            fkey = FrameKey(desktop, int(step), view)
            flags = self.db.get_frame(fkey) or {}
            if flags.get("review_status") == VERIFIED:
                continue
            try:
                self.truth.verify_frame(fkey, self.annotator)
                resigned += 1
            except ValueError as why:
                refused.append((int(step), str(why)))
                log.info("flip: frame %s not re-signed: %s", step, why)
        return resigned, refused
