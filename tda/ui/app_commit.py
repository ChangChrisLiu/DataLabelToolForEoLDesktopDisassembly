"""Committing an edit, the scope bar, undo and the visibility keys.

Mixed into :class:`tda.ui.app.MainWindow` next to
:class:`tda.ui.app_edit.EditMixin`, which owns the layer these write back: what
is here is the moment the annotator says "yes" -- which scope the edit means,
what that scope is going to write, what ``Esc`` takes back, and the one history
every one of them lands in.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from tda.ui import app_actions as A
from tda.ui import app_compat as compat
from tda.ui import app_priors
from tda.ui import app_support as S
from tda.ui import session_api as api
from tda.ui.app_state import REMOVED_ROW, cut_note
from tda.ui.class_names import instance_label, visibility_zh
# the frame-range wording lives with the guide's other sentences now (U7: the
# add-parts dialog and the bulk accept say ranges too); importable from here
from tda.ui.guide import steps_text  # noqa: F401

__all__ = ["CommitMixin"]

#: The two halves of the layering vocabulary the bar has to reason about; the
#: session owns their meaning (:mod:`tda.ui.session_scope`), the window only
#: needs to know which suggestion is on the bar.
A_ZORDER_ABOVE = "zorder:above:"
A_ZORDER_BELOW = "zorder:below:"
SPLIT_PREFIX = "split+"

#: ``Ctrl+K`` on the frame where the shape in force already starts: the session
#: re-traces that version instead (``_cannot_split``), and said nothing -- the
#: D13 annotator believed they had split (task U6).
NO_SPLIT = "这一帧就是这版形状的起点：Ctrl+K 等于重画这一版，没有拆分"
#: A shape committed on a components-gallery part marked 「看不见」 (task U10).
ABSENT_CLEARED = ("画好了；它在这张里，「看不见」已取消（Ctrl+Z 一起撤回）/ drawn: it is in "
                  "this shot, so 'not in this shot' was taken back")
#: ``Ctrl+K`` on an open ✂ row with the layer untouched (U8 round 2b): the
#: version boundary is cut here and the shape on this side stays as it is.
SPLIT_UNCHANGED = ("已在这帧切出新版本（形状不变）：另一侧的帧现在单独一版，到那边用 Enter "
                   "改它，只影响那一段")
#: ``L`` with no layering suggestion on the bar.
NO_LAYERING = ("L 只用来回答画布下方的层级建议；现在没有层级建议 / L answers a "
               "layering suggestion, and none is open")
#: ``Enter`` while the layering question is up: it is not the answer.
LAYERING_ASKED = ("层级还没改：再按 L（或点「L 确定改层级」）确定，Esc 取消 / press L "
                  "again to change the layering, Esc to cancel")
LAYERING_CANCELLED = "层级没改 / the layering was not changed"
#: The scope bar's layering button, before and while the question is asked.
LAYER_BUTTON = "L 改层级"
LAYER_BUTTON_CONFIRM = "L 确定改层级"


def is_layering(scope: Optional[str]) -> bool:
    """Is ``scope`` one of the session's layering answers (with or without a split)?"""
    body = str(scope or "")
    if body.startswith(SPLIT_PREFIX):
        body = body[len(SPLIT_PREFIX):]
    return body.startswith((A_ZORDER_ABOVE, A_ZORDER_BELOW))



class CommitMixin:
    """``Enter`` / ``Esc`` / ``Space``, the scope bar, undo-redo, visibility."""

    # --------------------------------------------------------------- commits
    @S.guard
    def act_commit(self) -> None:
        """``Enter``: take the ghost, store the ROI, accept the scope, or commit.

        One chain, in the order of *what is on screen*.  The ROI is in it
        because while its bar is up it is what the bar is talking about: "Enter
        确认" next to a rectangle that ``Enter`` did not store -- it committed
        the instance loaded behind it instead -- is the kind of thing nobody
        notices until the rectangle is missing.  The draft ghost joins the same
        chain at the top: it can only be up when none of the others is (see
        :meth:`~tda.ui.app_adopt.AdoptMixin._adopt_refusal`), so the order
        between them never has to be guessed at.

        Above even the ghost: a ``P`` polygon half drawn (task U5a).  It is the
        gesture in the annotator's hand, so ``Enter`` closes it -- and with
        fewer than three vertices does nothing but say so -- before it can mean
        anything to the layer underneath.
        """
        if self.finish_shape():
            return
        if self.accept_draft_ghost():
            return
        if self.roi_editing:
            self.accept_roi()
            return
        if self._pending_scope is not None:
            if self._layering_asked:
                # The question on the bar is answered by L; Enter is the key
                # pressed a hundred times a day, and it changes no layering.
                self.report(LAYERING_ASKED)
                return
            # Task U6: Enter on a layering suggestion commits the shape only.
            # The layering itself is L's, asked and confirmed.
            self._commit(api.SCOPE_KEYFRAME if is_layering(self._pending_scope)
                         else self._pending_scope)
            return
        if getattr(self.session, "editing_instance", None) is not None \
                and self.session.editing_mask() is not None:
            if not self.has_uncommitted_edit():
                # Enter on a layer nobody has touched used to report
                # "committed (keyframe): False", write nothing, and clear the
                # layer -- so the annotator lost the instance they had just
                # loaded and had to double-click it again.
                self.report("没有可提交的修改 / nothing to commit on this layer")
                return
            scope = self.session.suggest_scope()
            if scope == api.SCOPE_KEYFRAME:
                self._commit(scope)
            else:
                self._offer_scope(scope)

    @S.guard
    def commit_with_suggested_scope(self) -> None:
        """Commit straight away, no bar: the close dialog's "Save".

        There is no non-modal conversation left to have, so the edit is
        written -- and a layering suggestion is written as what ``Enter`` would
        write, the shape only (task U6): a layering change needs its own key,
        and a dialog about closing the window is not where anybody gives it.
        """
        if getattr(self.session, "editing_instance", None) is None:
            return
        self._drop_scope_bar()
        scope = self.session.suggest_scope()
        self._commit(api.SCOPE_KEYFRAME if is_layering(scope) else scope)

    @S.guard
    def act_commit_override(self) -> None:
        """``Alt+Enter``: this frame only."""
        if self.refuse_under_shape() or self.refuse_under_ghost():
            return
        self._commit(api.SCOPE_FRAME_OVERRIDE)

    @S.guard
    def act_commit_split(self) -> None:
        """``Ctrl+K``: a new shape version from this step on -- the shape only.

        On an open layering suggestion too (task U6): the layering is ``L``'s,
        asked and confirmed, and no other key changes it.  The bar says that
        the pixels painted into the other part stay under it.  On the frame
        where the version in force already starts there is nothing to split
        off: the session re-traces that version, and :meth:`_commit` says so.
        """
        if self.refuse_under_shape() or self.refuse_under_ghost():
            return
        self._commit(api.SCOPE_SPLIT)

    @S.guard
    def act_commit_layering(self) -> None:
        """``L``: change the layering the bar suggests -- asked first, then done.

        Task U6.  The layering answer used to be what plain ``Enter`` did, and
        on D13/scan one ``Enter`` put the chassis over a screw for frames 1-40
        and turned two confirmed frames red.  So it has a key of its own, and
        the first press only *shows what it reaches*: the pose segment's frames
        and the confirmed ones among them, which will be checked again.  The
        second press (or the bar's button) writes it; ``Esc`` takes the
        question back and leaves the suggestion up; ``Enter`` says which key
        answers.
        """
        if self.refuse_under_shape() or self.refuse_under_ghost():
            return
        scope = self._pending_scope
        if not is_layering(scope):
            self.report(NO_LAYERING)
            return
        if not self._layering_asked:
            self._layering_asked = True
            self.scope_layer_button.setText(LAYER_BUTTON_CONFIRM)
            self.scope_bar.show_text(self._reach_question(scope))
            self.logger.info("layering asked: %s", scope)
            self.report("层级改动先看范围：画布下方提示条写着影响哪些帧，再按 L 确定，Esc 取消 / "
                        "check the reach on the bar, then L again")
            self.refresh_guidance()
            return
        self._commit(scope)

    def cancel_layering_question(self) -> bool:
        """``Esc`` on the layering question: back to the suggestion, nothing written."""
        if not self._layering_asked or self._pending_scope is None:
            return False
        self._layering_asked = False
        self.scope_layer_button.setText(LAYER_BUTTON)
        self.scope_bar.show_text(self._suggestion_text(self._pending_scope))
        self.report(LAYERING_CANCELLED)
        self.refresh_guidance()
        return True

    def layering_asked(self) -> bool:
        """Is the bar asking "确定？" about a layering change right now?"""
        return bool(self._layering_asked and self._pending_scope is not None)

    def scope_bar_text(self) -> str:
        """What the non-modal scope bar is currently saying."""
        return self.scope_bar.label.text()

    def _drop_scope_bar(self) -> None:
        """No suggestion, no question, no bar."""
        self._pending_scope = None
        self._layering_asked = False
        self.scope_layer_button.setText(LAYER_BUTTON)
        self.scope_bar.hide()

    def _offer_scope(self, scope: str) -> None:
        """Show the suggestion and its alternatives without blocking the canvas."""
        self._pending_scope = scope
        self._layering_asked = False
        self.scope_layer_button.setText(LAYER_BUTTON)
        self.scope_bar.show_text(self._suggestion_text(scope))
        self.logger.info("suggested scope: %s", scope)
        self.report("看画布下方提示条：Enter 只改形状，L 改层级 / read the bar: Enter "
                    "writes the shape, L changes the layering"
                    if is_layering(scope) else f"suggested scope: {scope}")

    def _pair_labels(self, scope: str) -> tuple[str, str, bool]:
        """``(edited, other, edited goes above)`` of a layering scope, as names to read."""
        body = scope[len(SPLIT_PREFIX):] if scope.startswith(SPLIT_PREFIX) else scope
        above = body.startswith(A_ZORDER_ABOVE)
        other = body[len(A_ZORDER_ABOVE if above else A_ZORDER_BELOW):]
        me = str(getattr(self.session, "editing_instance", None) or "")
        return (self._label_of(me)[0] or me, self._label_of(other)[0] or other, above)

    def _suggestion_text(self, scope: str) -> str:
        """The bar's sentence: what happened, and what each key would write (U6).

        In plain words, because the bar used to say ``建议范围 zorder:above:B``
        with "（Enter：仅层级 …）" behind it, and the quickstart said "一般 Enter
        接受": the D13 annotator pressed Enter and the chassis went over a
        screw on forty frames.  ``Enter`` now writes the shape only and says
        what that leaves on screen; ``L`` is the layering, and says what it
        hides.
        """
        if not is_layering(scope):
            return f"建议范围 {scope}"
        me, other, above = self._pair_labels(scope)
        tail = "。Alt+Enter 只改这一帧；Ctrl+K 从这帧起新版本（只改形状）；Esc 回去改"
        if above:
            return (f"你画到了「{other}」占着的像素（它盖在「{me}」上面，画进去的这块现在"
                    f"看不见）。Enter = 只改形状，「{other}」照样盖着；L = 改层级：把「{me}」"
                    f"放到「{other}」上面（两者重叠处「{other}」会被「{me}」盖住，整个位姿段"
                    f"都这样）{tail}")
        return (f"你擦掉的是「{me}」和「{other}」重叠的地方。Enter = 只改形状（擦掉的不再"
                f"算「{me}」的）；L = 不改形状、改层级：把「{other}」放到「{me}」上面（重叠处"
                f"「{me}」会被「{other}」盖住，整个位姿段都这样）{tail}")

    def _reach_question(self, scope: str) -> str:
        """The question ``L`` asks before it writes anything (U6 addendum, item 1).

        A pair override is per ``(view, pose segment)``: on D13/scan the one
        made on frame 40 reached frames 1-40, and the confirmed 39 and 40 came
        back as conflicts while the annotator believed they had changed one
        frame.  The reach is :meth:`~tda.ui.session_commits.CommitsMixin.preview`'s
        -- the pair's frames, and the shape's when the gesture carries pixels.
        """
        me, other, above = self._pair_labels(scope)
        upper, lower = (me, other) if above else (other, me)
        counts = compat.preview(self.session, scope) or {}
        steps = counts.get("steps") or []
        verified = counts.get("verified_steps") or []
        where = f"第 {steps_text(steps)} 帧" if steps else "（算不出影响哪些帧）"
        frozen = (f"其中已确认的第 {steps_text(verified)} 帧会被重新检查（可能产生冲突）"
                  if verified else "其中没有已确认的帧")
        return (f"层级改动作用于整个位姿段：{where}，{frozen}。把「{upper}」放到「{lower}」"
                f"上面，确定？再按 L（或点「{LAYER_BUTTON_CONFIRM}」）确定，Esc 取消")

    def _commit(self, scope: str) -> None:
        """Write the edit, or show why the session will not take it.

        A refusal -- "this part is on the bench, use the bench box" -- is an
        ordinary answer, not a failure: it keeps the editing layer so the
        annotator can press ``R`` and draw the box instead, and it does not go
        through the exception path, which would log a traceback for something
        the annotator simply has to read.
        """
        instance = getattr(self.session, "editing_instance", None)
        if instance is None:
            self.report("nothing is being edited")
            return
        if not self._area_is_plausible(instance, scope):
            return
        key = self.session.current()
        mask = self.session.editing_mask()
        pixels = int(mask.sum()) if mask is not None else 0
        started = time.perf_counter()
        extra = self._commit_extra(key, instance, mask)
        try:
            result = self.session.commit_edit(scope, extra=extra) or {}
        except ValueError as refused:
            self._drop_scope_bar()
            self.logger.info("refused commit %s %s step %s: %s",
                             instance, scope, key.step, refused)
            self.report_error(f"refused: {refused}")
            return
        self.logger.info("commit %s scope=%s step=%s px=%d ms=%.0f affected=%d",
                         instance, scope, key.step, pixels,
                         (time.perf_counter() - started) * 1000.0,
                         len(result.get("affected") or []))
        self._drop_scope_bar()
        self.session.clear_edit()
        # One of the three: what any copy of this instance held is in the
        # database now, so even one from an earlier run is stale.
        self.drop_sidecar(key, instance, foreign_ok=True)
        self.set_sam_instance(None)
        self._sync_editing_layer()
        self.refresh_overlay()
        # An edit reaches other frames: those rows change colour now, not when
        # the annotator next happens to stand on one of them.
        self.timeline.refresh_statuses()
        self.forget_card_hints()          # the hover outlines were read from shapes
        # No re_explain() here: committing re-renders the frame, which clears
        # assist_result, so a re-split would run against nothing.  The real one
        # happens at confirm time, where the answer is actually used.
        cut = cut_note(result, steps_text)
        if cut:
            # the commit stopped at a state boundary (U8 round 3): say where,
            # and which frames the edit reached
            self.logger.info("commit of %s at step %s cut at %s / %s", instance, key.step,
                             result.get("cut_below"), result.get("cut_top"))
            self.report(cut, hold_ms=6000)
            return
        if result.get("no_split"):
            # Ctrl+K where the version in force starts: it re-traced that one
            # (U6 addendum, item 2) -- and saying nothing let the annotator
            # believe the frames before this one had their own version now.
            self.logger.info("split of %s at step %s re-traced the version that "
                             "starts there", instance, key.step)
            self.report(NO_SPLIT, hold_ms=6000)
            return
        if result.get("unchanged_split"):
            # Ctrl+K on an open ✂ row, layer untouched: only the boundary (U8 2b)
            self.logger.info("split of %s at step %s with the shape unchanged",
                             instance, key.step)
            self.report(SPLIT_UNCHANGED, hold_ms=6000)
            return
        if result.get("cleared_absent"):
            # a shape drawn on a part marked 「看不见」 on a gallery shot (U10)
            self.report(ABSENT_CLEARED, hold_ms=6000)
            return
        self.report(f"committed ({scope}): {result.get('changed', '')}".strip())

    def _commit_extra(self, key, instance: str,
                      mask: Optional[np.ndarray]) -> Optional[dict]:
        """The op-log note this commit carries: what the *window* knows about it.

        Two kinds of fact, and they can both be true of one commit -- a draft
        adopted into the layer and then committed at a size the priors call
        implausible.  They are found in completely different ways, which is the
        point of keeping them apart:

        * the **area override** is a flag the annotator set by answering a
          warning.  It belongs to the one commit it was given for, taken or
          refused, or the next ``Enter`` would inherit an answer to a warning
          nobody read.
        * the **adoptions** are *derived*: the applied-and-not-undone strokes
          of this layer that came from a draft, each checked against the pixels
          actually being written.  Nothing has to remember to clear them --
          undoing the stroke removes it from the history, and erasing the
          pixels removes it at the overlap check.

        ``None`` rather than ``{}`` when there is nothing to say: the session
        checks a dict for JSON-serialisability and merges it, and an empty one
        would only add work to every commit.
        """
        facts: dict = {}
        override, self._override_facts = self._override_facts, None
        if override:
            facts |= dict(override)
        adopted = self.adoptions_for_commit(key, instance, mask)
        if adopted:
            facts["adopted"] = adopted
        return facts or None

    @S.guard
    def act_clear_edit(self) -> None:
        """``Esc``: dismiss the ghost, the ROI rectangle, the bench arm or the layer.

        Whichever gesture owns the canvas right now is the one ``Esc`` answers:
        while the ROI bar is up that is the rectangle, not the instance that
        happens to be loaded behind it, and while a draft ghost is up it is the
        ghost -- which is why it dismisses *only* the ghost and the layer under
        it survives.

        First of all, a ``P`` polygon or a ``Y`` circle half drawn (task U5a):
        ``Esc`` drops that shape and nothing else -- the layer it would have
        been filled into is the next ``Esc``'s.
        """
        if self.cancel_shapes():
            return
        if self.clear_draft_ghost():
            self.report("草稿已取消 / the draft was dismissed")
            return
        if self.cancel_layering_question():
            return          # the question goes; the suggestion and the layer stay (U6)
        if self._pending_warning is not None:
            # The warning is the thing on screen: Esc answers it by taking the
            # annotator back to the layer, not by throwing the layer away.
            self._pending_warning = None
            self.warn_bar.hide()
            self.report("回到编辑 / back to the mask")
            return
        if self.roi_editing:
            self.cancel_roi_edit()
            self.report("ROI unchanged")
            return
        if self.bench_instance is not None:
            self.disarm_bench()
            self.report("bench box cancelled")
            self.refresh_guidance()
            return
        instance = getattr(self.session, "editing_instance", None)
        if instance is not None:
            key = self.session.current()
            self.session.clear_edit()
            # Esc discards *the layer on screen*.  A crash copy from an earlier
            # run is a question nobody has answered, so it -- and its offer --
            # stay; only this window's own copy goes.
            self.drop_sidecar(key, instance)
            self._offer_restore(key, instance)
            self._drop_scope_bar()
            self.set_sam_instance(None)
            self._sync_editing_layer()
            self.report("edit discarded")

    @S.guard
    def act_confirm(self) -> bool:
        """``Space``: verify the frame; on refusal the problems are shown, not a dialog.

        The unexplained differences are re-derived *here*, against what the frame
        holds now, and only what is still unexplained goes to the review queue.
        If the comparison has not come back yet it is finished synchronously
        under a short cap; if even that fails the step is recorded as "not
        analysed" rather than as "nothing unexplained", which would quietly
        claim the frame had been checked.
        """
        if self.refuse_under_shape():
            return False      # the half-drawn shape would go with the frame (U5a)
        if self.refuse_under_ghost():
            return False      # the preview owns every commit key while it is up
        if not self.can_leave_edit():
            return False      # confirming steps the frame back: same gate
        if not self._open_the_selected_entry():
            return False
        step = self.session.current().step
        blobs = self.unexplained_at_confirm()
        ok = self.task_card.confirm()
        self.logger.info("confirm step %s: %s", step, "ok" if ok else "refused")
        if ok:
            self.hand_over_unexplained(step, blobs)
            # One more frame is done: both places that count say so.
            self.timeline.refresh_statuses()
            self.refresh_desktop_counts()
            self.report(f"{self.frame_word(step)} confirmed")
        else:
            # One line, whatever the frame is missing.  A real start frame has
            # 60+ missing shapes, which put 2,550 characters into the one-line
            # status bar and asked the window to be 30,612 px wide.
            count = self.task_card.problem_count()
            self.report(f"{self.frame_word(step)} is not complete: {count} problem(s) — "
                        f"见任务卡 / see the task card")
            self.refresh_guidance()      # the pane now says what blocks Space
        return bool(ok)

    # ------------------------------------------------------- the size warning
    def warn_bar_text(self) -> str:
        """What the area warning bar is currently saying."""
        return self.warn_bar.label.text()

    def _area_is_plausible(self, instance: str, scope: str) -> bool:
        """``False`` when a warning was raised and is waiting for a second Enter.

        Never blocks: the second press writes the mask exactly as it is and the
        override is logged, because the annotator is the authority on what a
        part looks like.  The rehearsal committed 1,502,386 px as a ``screw``
        and 40 masks under 50 px with nothing said either way, which is the
        only outcome this rules out.
        """
        pending = self._pending_warning
        if pending is not None and pending[0] == scope:
            self._pending_warning = None
            self.warn_bar.hide()
            # Both places: the log line is what somebody greps, the payload is
            # what the commit itself carries.  Nothing joins a log line back to
            # one row of the op log, and that row is the mask's audit trail.
            self._override_facts = dict(pending[2])
            self.logger.info("area_warning_overridden instance=%s scope=%s: %s",
                             instance, scope, pending[1])
            return True
        mask = self.session.editing_mask()
        if mask is None:
            return True
        cls = self._class_of(instance)
        warning = app_priors.area_warning(mask, cls, self._roi_area(), self.priors)
        if warning is None:
            self._pending_warning = None
            self.warn_bar.hide()
            return True
        self._pending_warning = (scope, warning,
                                 self._warning_facts(mask, cls, warning))
        self.warn_bar.show_text(f"{warning}  —— Enter 仍然提交 / Esc 回去改")
        self.report(warning)
        self.logger.info("area_warning instance=%s scope=%s: %s",
                         instance, scope, warning)
        return False

    def _warning_facts(self, mask, cls: str, warning: str) -> dict:
        """What an override of this warning should say in the op log.

        Plain JSON, because the op log is JSON: the size that was questioned,
        the band it was judged against when the class has one -- the "too
        small" rule applies to every class, including one the priors have never
        seen -- and the sentence the annotator read before going ahead.
        """
        band = self.priors.band(cls) if cls else None
        facts: dict = {"area_warning_overridden": True,
                       "area_px": int(np.count_nonzero(mask)),
                       "area_warning": str(warning)}
        if band is not None:
            facts["area_bounds"] = [float(band[0]), float(band[1])]
        return facts

    def _class_of(self, instance: str) -> str:
        """The taxonomy class of an instance, for the per-class prior."""
        for row in self.session.instance_rows():
            if str(row.get("key")) == str(instance):
                return str(row.get("cls") or "")
        return str(instance).split(".", 1)[0] if instance else ""

    def _roi_area(self) -> float:
        """Area of the stored ROI, or of the whole frame when there is none."""
        roi = self.roi()
        if roi is not None:
            return max(1.0, float(roi[2] - roi[0]) * float(roi[3] - roi[1]))
        hw = None if self.overlay is None else self.overlay.hw
        return max(1.0, float(hw[0]) * float(hw[1])) if hw else 1.0

    def _open_the_selected_entry(self) -> bool:
        """In Review mode, make ``Enter`` mean what its label says.

        The binding reads "accept the frame the queue points at" and it
        confirmed whatever was on the canvas: the annotator clicked an entry,
        pressed ``Enter``, and a different frame was marked verified.  The
        selected entry's frame is opened first -- through the same gate as any
        other move -- and only then confirmed.  ``False`` means the move was
        refused, so nothing is confirmed either.
        """
        if self.mode != A.MODE_REVIEW:
            return True
        step = self.review.selected_step()
        if step is None or int(step) == int(self.session.current().step):
            return True
        return self.leave_frame(
            lambda: self.session.goto(int(step), force=True)
        )

    # ------------------------------------------------------------------ undo
    @S.guard
    def act_undo(self) -> None:
        self._history(self.session.undo(), "undo")

    @S.guard
    def act_redo(self) -> None:
        self._history(self.session.redo(), "redo")

    def _history(self, moved: bool, what: str) -> None:
        if not moved:
            self.report(f"nothing to {what}")
            return
        self.refresh_overlay()
        self._sync_editing_layer()
        # An undo can take the layer back to what begin_edit loaded, and then
        # the sidecar it wrote (or is about to) describes work that no longer
        # exists: re-queue it, which drops both when there is nothing left.
        instance = getattr(self.session, "editing_instance", None)
        mask = self.session.editing_mask()
        if instance is not None and mask is not None:
            self.queue_sidecar(self.session.current(), instance, mask)
        self.timeline.refresh_statuses()   # it takes back other frames too
        self.forget_card_hints()
        self.update_status()
        self.report(what)

    @S.guard
    def _on_editing_changed(self, mask: object) -> None:
        """The session moved the editing layer behind our back (undo/redo)."""
        self._sync_editing_layer()

    # ------------------------------------------------------------ visibility
    @S.guard
    def act_set_visibility(self, value: str) -> None:
        if not self._editable_row() or not self._visibility_target():
            return
        self.instances.set_visibility(value)
        self._after_visibility()

    @S.guard
    def act_cycle_visibility(self) -> None:
        if not self._editable_row() or not self._visibility_target():
            return
        self.instances.cycle_visibility()
        self._after_visibility()

    def _visibility_target(self) -> bool:
        """Is an instance selected for ``1``-``7`` / ``V``?  Says so when not.

        The keys acted on the instance table's selection and on nothing else,
        silently (round 3): with no row selected a digit did nothing and said
        nothing.
        """
        if self.instances.selected_instance():
            return True
        self.report("可见性键 1–7 / V 作用于实例表里选中的零件：先在右下实例表点一行 / "
                    "select a part in the instance table first")
        return False

    def _after_visibility(self) -> None:
        """Repaint, and say what changed and how to take it back (round 3).

        A digit key changed a frame's visibility label without a word, which
        is how the trial DB's chassis came to be "out of view" on step 42: the
        status line and a short-lived canvas banner now name the part, the new
        value and ``Ctrl+Z``.
        """
        self.refresh_overlay()
        instance = self.instances.selected_instance()
        row = next((r for r in self.instances.rows() if r.get("key") == instance), None)
        if row is not None:
            label = self._label_of(instance)[0] or instance_label(str(instance),
                                                                  row.get("cls"))
            note = (f"{label}：这一帧的可见性 → {visibility_zh(row.get('visibility', ''))}"
                    f"（Ctrl+Z 撤销）")
            self.report(note)
            self.flash_visibility_note(note)
        self.task_card.refresh()         # a ✔ row may now say "已设为不可见"
        self.timeline.refresh_statuses()
        self.update_status()

    @S.guard
    def act_toggle_hidden(self) -> None:
        if not self._editable_row():
            return
        self.instances.toggle_hidden()
        self.refresh_overlay()

    def _editable_row(self) -> bool:
        """Is the instance table pointing at something these keys can change?

        The ``Removed`` toggle lists parts the frame no longer has; they are
        read-only, and ``H`` / ``V`` / ``1``-``7`` on one of them did nothing at
        all -- which reads exactly like the key not working.
        """
        if self.instances.selected_is_removed():
            self.report(REMOVED_ROW)
            return False
        return True

