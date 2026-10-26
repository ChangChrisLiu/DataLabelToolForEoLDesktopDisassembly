"""Committing an edit, the scope bar and undo.

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
from tda.ui.session_cable import CABLE_NOW_PARTIAL
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

#: ``Enter`` on a layer nobody has touched.
NOTHING_TO_COMMIT = "没有可提交的修改 / nothing to commit on this layer"
#: ``Ctrl+Enter`` (改全部) on the frame where the version in force starts: it is
#: what ``Enter`` writes there, and it says so (task U16).
ALL_IS_ENTER = "这一帧就是这版形状的起点：改全部和 Q 一样，重画的就是这一版"
#: A shape committed on a components-gallery part marked 「看不见」 (task U10).
ABSENT_CLEARED = ("画好了；它在这张里，「看不见」已取消（Ctrl+Z 一起撤回）/ drawn: it is in "
                  "this shot, so 'not in this shot' was taken back")
#: ``Enter`` on an open ✂ row with the layer untouched (U8 round 2b): the
#: version boundary is cut here and the shape on this side stays as it is.
SPLIT_UNCHANGED = ("已在这帧切出新版本（形状不变）：另一侧的帧现在单独一版，到那边用 Q "
                   "改它，只影响那一段")
#: The commit (``Q``) or ``L`` while 改全部 asks: neither is the answer (task U16).
ALL_ASKED = ("还没改：再按 Ctrl+Q（或点「Ctrl+Q 确定改全部」）确定，Esc 取消 / "
             "press Ctrl+Q again to write it everywhere, Esc to cancel")
ALL_CANCELLED = "没有改全部 / nothing was written"
#: The scope bar's 改全部 button, before and while the question is asked.
ALL_BUTTON = "Ctrl+Q 改全部"
ALL_BUTTON_CONFIRM = "Ctrl+Q 确定改全部"
#: After the commit wrote a split because the version was carried (task U16):
#: the frames behind that keep the old shape, and the key that reaches them.
BEHIND_KEPT = "已从这帧起新版本；第 {steps} 帧{confirmed}仍是旧样子，要一起改用 Ctrl+Q"
#: 改全部 on a version that starts here, the frames behind on the
#: previous one (U16 review): asked first when confirmed frames are among them,
#: and said once written.
INTO_QUESTION = ("把这版形状也写到第 {steps} 帧{confirmed}？再按 Ctrl+Q（或点「"
                 + ALL_BUTTON_CONFIRM + "」）确定，Esc 取消")
INTO_DONE = "已把这版形状也写到第 {steps} 帧{confirmed}"
#: The key of each commit action, as the area warning names it (task U16):
#: read off the key map, so it is the left-hand key since task U22.
WARNING_KEYS = {name: A.action_named(name).keys[0]
                for name in ("commit", "commit_all", "commit_override")}
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

        **What the commit means** (task U16): *from this frame on* in the walk
        -- :meth:`_enter_scope`, the session's judgement.  The annotator walks
        backwards and confirms as they go; a shape changed on frame k is the
        shape from k on, and the frames behind them stay as they are.  Where
        the version in force starts it is re-traced, as before.  Editing the
        version everywhere it applies, confirmed frames behind included, is
        ``Ctrl+Enter`` (:meth:`act_commit_all`).  ``Ctrl+K`` is this slot too.
        """
        if self.finish_shape():
            return
        if self.accept_draft_ghost():
            return
        if self.roi_editing:
            self.accept_roi()
            return
        if self._all_asked:
            # the 改全部 question is answered by Ctrl+Enter (task U16)
            self.report(ALL_ASKED)
            return
        if self._pending_scope is not None:
            if self._layering_asked:
                # The question on the bar is answered by L; Enter is the key
                # pressed a hundred times a day, and it changes no layering.
                self.report(LAYERING_ASKED)
                return
            # Task U6: Enter on a layering suggestion commits the shape only,
            # from this frame on (U16). The layering itself is L's, asked and
            # confirmed.
            if is_layering(self._pending_scope):
                enter = self._enter_scope()
                self._commit(enter, behind=enter == api.SCOPE_SPLIT)
            else:
                self._commit(self._pending_scope)
            return
        instance = getattr(self.session, "editing_instance", None)
        if instance is not None and self.session.editing_mask() is not None:
            if not self.has_uncommitted_edit():
                if self._open_split_here(instance):
                    # An open ✂ row whose shape is already right on this side:
                    # the version boundary is the edit (U8 round 2b, Ctrl+K's
                    # until U16). A cut copy here is refused by the session.
                    self._commit(api.SCOPE_SPLIT)
                    return
                # Enter on a layer nobody has touched used to report
                # "committed (keyframe): False", write nothing, and clear the
                # layer -- so the annotator lost the instance they had just
                # loaded and had to double-click it again.
                self.report(NOTHING_TO_COMMIT)
                return
            scope = self.session.suggest_scope()
            if scope == api.SCOPE_KEYFRAME:
                enter = self._enter_scope()
                self._commit(enter, behind=enter == api.SCOPE_SPLIT)
            else:
                self._offer_scope(scope)

    def _enter_scope(self) -> str:
        """The pixel scope ``Enter`` writes here: from this frame on (task U16).

        ``split`` where the version in force is carried from frames the walk
        has passed, ``keyframe`` where it starts -- the session's answer
        (:func:`tda.ui.app_compat.enter_scope`).
        """
        return compat.enter_scope(self.session)

    def _open_split_here(self, instance: str) -> bool:
        """Is ``instance``'s row on this frame's card an open ✂? The session's answer.

        Asked only for an untouched layer, where it decides whether ``Enter``
        writes at all; the panel's copy of the card is the fallback.
        """
        ask = getattr(self.session, "open_split_here", None)
        if callable(ask):
            return bool(ask(instance))
        return self._open_split_editing()

    @S.guard
    def commit_with_suggested_scope(self) -> None:
        """Commit straight away, no bar: the close dialog's "Save".

        There is no non-modal conversation left to have, so the edit is
        written -- and a layering suggestion is written as what ``Enter`` would
        write, the shape only (task U6): a layering change needs its own key,
        and a dialog about closing the window is not where anybody gives it.
        With ``Enter``'s scope, too: from this frame on (task U16).
        """
        if getattr(self.session, "editing_instance", None) is None:
            return
        self._drop_scope_bar()
        scope = self.session.suggest_scope()
        self._commit(self._enter_scope()
                     if scope == api.SCOPE_KEYFRAME or is_layering(scope) else scope)

    @S.guard
    def act_commit_override(self) -> None:
        """``Alt+Enter``: this frame only."""
        if self.refuse_under_shape() or self.refuse_under_ghost():
            return
        self._commit(api.SCOPE_FRAME_OVERRIDE, action="commit_override")

    @S.guard
    def act_commit_all(self) -> None:
        """``Ctrl+Enter``: 改全部 -- the version in force, everywhere it applies.

        Task U16. ``Enter`` is "from this frame on" since the user's ruling;
        this is the rare other answer, the re-trace that reaches every frame
        of the version -- the confirmed ones the walk has passed included, and
        those come back to be checked and may conflict. So, modelled on ``L``
        (:meth:`act_commit_layering`): when that reach holds a confirmed frame
        the first press only *asks* on the bar -- how many frames, which
        confirmed ones -- and a second ``Ctrl+Enter`` (or the bar's button)
        writes; ``Esc`` takes the question back; ``Enter`` and ``L`` say which
        key answers. With no confirmed frame in reach it writes at once.

        Where the version starts here there are two cases (U16 review, M1).
        The frames behind still on the *previous* version -- the split ``Enter``
        just wrote, which the status line named with "要一起改用 Ctrl+Enter" --
        get this shape too: it is written into that version and this frame's
        own is dropped (:data:`~tda.ui.session_api.SCOPE_INTO_PREVIOUS`), asked
        first when confirmed frames are among them; the layer when it holds
        pixels, the stored shape when not -- and right after that ``Enter``,
        with nothing loaded, the part it named. Nothing behind on another
        version: it is what ``Enter`` writes, written, and said. On an open ✂
        row the session refuses it with its own sentence (it would rewrite the
        other state's frames), so it is not asked first. The shape only on a
        layering suggestion, like every key but ``L`` (task U6).
        """
        if self.refuse_under_shape() or self.refuse_under_ghost():
            return
        if self._layering_asked:
            self.report(LAYERING_ASKED)
            return
        if self._all_asked:
            self._commit(self._all_scope, action="commit_all",
                         note=self._all_note)
            return
        if not self._load_hinted_part():
            self.report(NOTHING_TO_COMMIT)
            return
        behind = compat.previous_behind(self.session)
        if behind is not None:
            self._commit_into_previous(behind)
            return
        if not self.has_uncommitted_edit():
            self.report(NOTHING_TO_COMMIT)
            return
        if self._enter_scope() == api.SCOPE_KEYFRAME:
            self._commit(api.SCOPE_KEYFRAME, action="commit_all", note=ALL_IS_ENTER)
            return
        counts = compat.preview(self.session, api.SCOPE_KEYFRAME) or {}
        if not counts.get("verified_steps") or self._open_split_editing():
            self._commit(api.SCOPE_KEYFRAME, action="commit_all")
            return
        self._ask_all(api.SCOPE_KEYFRAME, self._all_question(counts), "")
        self.logger.info("commit_all asked: %s", counts.get("verified_steps"))

    def _ask_all(self, scope: str, question: str, note: str) -> None:
        """Put 改全部's question on the bar; the second press writes ``scope``."""
        self._all_asked = True
        self._all_scope, self._all_note = scope, note
        self._show_scope_buttons(question=True)
        self.scope_bar.show_text(question)
        self.report("改全部先看范围：画布下方提示条写着影响哪些帧，再按 Ctrl+Q 确定，"
                    "Esc 取消 / check the reach on the bar, then Ctrl+Q again")
        self.refresh_guidance()

    def _commit_into_previous(self, behind: dict) -> None:
        """改全部 on a version that starts here: write it into the previous one too.

        A state boundary in the way is the session's refusal (the commit path
        says it); confirmed frames behind are asked about first.
        """
        steps = behind.get("steps") or []
        frozen = behind.get("frozen") or []
        confirmed = f"（其中已确认 {len(frozen)} 帧，会重新检查）" if frozen else ""
        done = INTO_DONE.format(steps=steps_text(steps), confirmed=confirmed)
        if behind.get("refuse") or not frozen:
            self._commit(api.SCOPE_INTO_PREVIOUS, action="commit_all", note=done)
            return
        self._ask_all(api.SCOPE_INTO_PREVIOUS,
                      INTO_QUESTION.format(steps=steps_text(steps), confirmed=confirmed),
                      done)
        self.logger.info("commit_all into the previous version asked: %s", frozen)

    def _load_hinted_part(self) -> bool:
        """Is a part being drawn -- or loaded now, the one ``Enter`` just named?

        Right after ``Enter``'s split the layer is gone and the status line
        says "要一起改用 Ctrl+Enter" about the part it wrote: with nothing
        being drawn on that frame, ``Ctrl+Enter`` loads that part's stored
        shape and answers for it (U16 review, M1).
        """
        if getattr(self.session, "editing_instance", None) is not None \
                and self.session.editing_mask() is not None:
            return True
        hint = self._behind_hint
        if hint is None or not self._hint_here():
            return False
        if compat.previous_behind(self.session, hint[1]) is None:
            self._hint_live = False      # undone, or changed since: nothing to fold
            return False
        self.on_request_edit(hint[1])
        return getattr(self.session, "editing_instance", None) == hint[1]

    def _all_question(self, counts: dict) -> str:
        """What ``Ctrl+Enter`` asks before it writes (task U16).

        :meth:`~tda.ui.session_commits.CommitMixin.preview` of the keyframe
        scope -- the frames the re-trace writes, after the cuts at state
        boundaries -- and the confirmed ones among them.
        """
        steps = counts.get("steps") or []
        verified = counts.get("verified_steps") or []
        where = (f"第 {steps_text(steps)} 帧（共 {len(steps)} 帧）" if steps
                 else "（算不出影响哪些帧）")
        return (f"改全部会改这一版形状用到的每一帧：{where}，其中已确认的第 "
                f"{steps_text(verified)} 帧会被重新检查（可能产生冲突）。确定？再按 "
                f"Ctrl+Q（或点「{ALL_BUTTON_CONFIRM}」）确定，Esc 取消")

    def cancel_all_question(self) -> bool:
        """``Esc`` on the 改全部 question: nothing written, the layer stays (U16)."""
        if not self._all_asked:
            return False
        self._all_asked = False
        self._show_scope_buttons(question=False)
        if self._pending_scope is not None:
            self.scope_bar.show_text(self._suggestion_text(self._pending_scope))
        else:
            self.scope_bar.hide()
        self.report(ALL_CANCELLED)
        self.refresh_guidance()
        return True

    def all_asked(self) -> bool:
        """Is the bar asking "确定？" about 改全部 right now?"""
        return bool(self._all_asked)

    def _show_scope_buttons(self, question: bool) -> None:
        """The bar's buttons for a suggestion, or only the 改全部 confirmation.

        While 改全部 asks, the one button is the answer; the suggestion's
        Enter / L / Alt+Enter are not, and on a frame with no suggestion they
        would offer things nobody was asked about.
        """
        self.scope_all_button.setText(ALL_BUTTON_CONFIRM if question else ALL_BUTTON)
        for button in (self.scope_enter_button, self.scope_layer_button,
                       self.scope_override_button):
            button.setVisible(not question)

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
        if self._all_asked:
            self.report(ALL_ASKED)      # the bar is asking about 改全部 (U16)
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
        self._all_asked = False
        self.scope_layer_button.setText(LAYER_BUTTON)
        self._show_scope_buttons(question=False)
        self.scope_bar.hide()

    def _offer_scope(self, scope: str) -> None:
        """Show the suggestion and its alternatives without blocking the canvas."""
        self._pending_scope = scope
        self._layering_asked = False
        self._all_asked = False
        self.scope_layer_button.setText(LAYER_BUTTON)
        self._show_scope_buttons(question=False)
        self.scope_bar.show_text(self._suggestion_text(scope))
        self.logger.info("suggested scope: %s", scope)
        self.report("看画布下方提示条：Q 只改形状，L 改层级 / read the bar: Q "
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
        hides.  ``Enter``'s shape is from this frame on; ``Ctrl+Enter`` is the
        other shape-only answer, 改全部 (task U16).
        """
        if not is_layering(scope):
            return f"建议范围 {scope}"
        me, other, above = self._pair_labels(scope)
        tail = "。Alt+Q 只改这一帧；Ctrl+Q 改全部（只改形状）；Esc 回去改"
        if above:
            return (f"你画到了「{other}」占着的像素（它盖在「{me}」上面，画进去的这块现在"
                    f"看不见）。Q = 只改形状，「{other}」照样盖着；L = 改层级：把「{me}」"
                    f"放到「{other}」上面（两者重叠处「{other}」会被「{me}」盖住，整个位姿段"
                    f"都这样）{tail}")
        return (f"你擦掉的是「{me}」和「{other}」重叠的地方。Q = 只改形状（擦掉的不再"
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

    def _commit(self, scope: str, *, action: str = "commit", note: str = "",
                behind: bool = False) -> None:
        """Write the edit, or show why the session will not take it.

        A refusal -- "this part is on the bench, use the bench box" -- is an
        ordinary answer, not a failure: it keeps the editing layer so the
        annotator can press ``R`` and draw the box instead, and it does not go
        through the exception path, which would log a traceback for something
        the annotator simply has to read.

        ``action`` is the key's action that asked (an area warning names the
        key to press again); ``note`` is said in place of the plain
        "committed" line when nothing more particular happened (task U16).
        ``behind``: ``Enter`` wrote this split because the version was carried
        from frames the walk has passed -- which keep the old shape, and are
        then named (:meth:`_kept_behind`).
        """
        instance = getattr(self.session, "editing_instance", None)
        if instance is None:
            self.report("nothing is being edited")
            return
        if not self._area_is_plausible(instance, scope, action):
            return
        kept = self._kept_behind(instance) if behind and scope == api.SCOPE_SPLIT else None
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
        old = self._behind_note(kept, result)
        # what a bare Ctrl+Enter on this frame now acts on (U16 review); the
        # merge-in keeps it, idle, for its own undo to bring back (R3)
        if old:
            self._behind_hint, self._hint_live = (key, instance), True
        elif scope == api.SCOPE_INTO_PREVIOUS:
            self._hint_live = False
        else:
            self._behind_hint, self._hint_live = None, False
        if old:
            # U16: the frames behind kept the old shape -- said where the
            # annotator looks, with the key that reaches them
            self.flash_visibility_note(old)
        if cut:
            # the commit stopped at a state boundary (U8 round 3): say where,
            # and which frames the edit reached
            self.logger.info("commit of %s at step %s cut at %s / %s", instance, key.step,
                             result.get("cut_below"), result.get("cut_top"))
            self.report(f"{cut}。{old}" if old else cut, hold_ms=6000)
            return
        if old:
            self.report(old, hold_ms=6000)
            return
        if result.get("unchanged_split"):
            # Enter on an open ✂ row, layer untouched: only the boundary (U8 2b)
            self.logger.info("split of %s at step %s with the shape unchanged",
                             instance, key.step)
            self.report(SPLIT_UNCHANGED, hold_ms=6000)
            return
        if result.get("cleared_absent"):
            # a shape drawn on a part marked 「看不见」 on a gallery shot (U10)
            self.report(ABSENT_CLEARED, hold_ms=6000)
            return
        if result.get("cable_partial"):
            # pixels drawn on a part 被线挡住（完全）: part of it is seen (U15)
            self.report(CABLE_NOW_PARTIAL.format(label=self._label_of(instance)[0] or instance),
                        hold_ms=6000)
            return
        if note:
            self.report(note, hold_ms=6000)
            return
        self.report(f"committed ({scope}): {result.get('changed', '')}".strip())

    def _kept_behind(self, instance: str) -> Optional[dict]:
        """What 改全部 would reach here, asked *before* ``Enter``'s split is written.

        Task U16, the review's discoverability gap: an annotator who drew a
        rough first draft two frames back and fixes it here expects the fix to
        reach those frames, and the split leaves them as they were.  This is
        ``retrace_reach(scope=keyframe)`` -- the frames ``Ctrl+Enter`` would
        write, after the cuts at state boundaries -- read while the version in
        force is still the carried one; :meth:`_behind_note` takes off the
        frames the split reached.  ``None`` on an open ✂ row (the frames
        behind are the other state's and keep their shape on purpose; 改全部
        is refused there) and when the session cannot answer.
        """
        reach = getattr(self.session, "retrace_reach", None)
        if not callable(reach) or self._open_split_here(instance):
            return None
        try:
            return reach(instance, api.SCOPE_KEYFRAME)
        except Exception:  # noqa: BLE001 - a hint is never worth a failed commit
            return None

    def _behind_note(self, kept: Optional[dict], result: dict) -> str:
        """「已从这帧起新版本；第 … 帧（其中已确认 n 帧）仍是旧样子…」, or ``""``."""
        if not kept:
            return ""
        reached = set(result.get("edit_reach") or result.get("affected") or ())
        steps = [int(s) for s in kept.get("steps") or () if int(s) not in reached]
        if not steps:
            return ""
        frozen = [s for s in kept.get("frozen") or () if int(s) in set(steps)]
        confirmed = f"（其中已确认 {len(frozen)} 帧）" if frozen else ""
        return BEHIND_KEPT.format(steps=steps_text(steps), confirmed=confirmed)

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
        if self.cancel_all_question():
            return          # ... and so does 改全部's (U16)
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
        # confirming steps the frame on: a 这一帧的说明 typed and not saved is
        # saved first (task U35), and said with the confirmation
        flush = getattr(self, "flush_frame_note", None)
        noted = flush() if callable(flush) else None
        step = self.session.current().step
        blobs = self.unexplained_at_confirm()
        ok = self.task_card.confirm()
        self.logger.info("confirm step %s: %s", step, "ok" if ok else "refused")
        # left only when the confirmation stepped the frame on
        said = self.note_saved_text(noted, left=bool(ok)) if noted is not None else ""
        tail = f"；{said}" if said else ""
        if ok:
            self.hand_over_unexplained(step, blobs)
            # One more frame is done: both places that count say so.
            self.timeline.refresh_statuses()
            self.refresh_desktop_counts()
            self.report(f"{self.frame_word(step)} confirmed{tail}")
        else:
            # One line, whatever the frame is missing.  A real start frame has
            # 60+ missing shapes, which put 2,550 characters into the one-line
            # status bar and asked the window to be 30,612 px wide.
            count = self.task_card.problem_count()
            self.report(f"{self.frame_word(step)} is not complete: {count} problem(s) — "
                        f"见任务卡 / see the task card{tail}")
            self.refresh_guidance()      # the pane now says what blocks Space
        return bool(ok)

    # ------------------------------------------------------- the size warning
    def warn_bar_text(self) -> str:
        """What the area warning bar is currently saying."""
        return self.warn_bar.label.text()

    def _area_is_plausible(self, instance: str, scope: str,
                           action: str = "commit") -> bool:
        """``False`` when a warning was raised and is waiting for a second Enter.

        Never blocks: the second press writes the mask exactly as it is and the
        override is logged, because the annotator is the authority on what a
        part looks like.  The rehearsal committed 1,502,386 px as a ``screw``
        and 40 masks under 50 px with nothing said either way, which is the
        only outcome this rules out.

        ``action`` -- the key that asked -- is kept with the warning, for the
        guide to name the key to press again: since task U16 one scope can be
        ``Enter``'s or ``Ctrl+Enter``'s.
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
                                 self._warning_facts(mask, cls, warning), action)
        again = WARNING_KEYS.get(action, WARNING_KEYS["commit"])
        self.warn_commit_button.setText(f"{again} 仍然提交")
        self.warn_bar.show_text(f"{warning}  —— {again} 仍然提交 / Esc 回去改")
        self.report(warning)
        self.logger.info("area_warning instance=%s scope=%s: %s",
                         instance, scope, warning)
        return False

    def _answer_warning(self) -> None:
        """The warning bar's 「仍然提交」: the key that raised the warning, again.

        Since task U16 one scope can be ``Enter``'s or ``Ctrl+Enter``'s, and
        ``Enter`` after a ``Ctrl+Enter`` warning means a different scope -- a
        new question, not the answer -- so the button presses the key that asked.
        """
        pending = self._pending_warning
        action = str(pending[3]) if pending is not None and len(pending) > 3 else "commit"
        slot = {"commit_all": self.act_commit_all,
                "commit_override": self.act_commit_override}.get(action, self.act_commit)
        slot()

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
        # a frame's note (task U35) is named: it may be another frame's than this
        said = self._note_history(undo=True)
        self._history(self.session.undo(), said or "undo")

    @S.guard
    def act_redo(self) -> None:
        said = self._note_history(undo=False)
        self._history(self.session.redo(), said or "redo")

    def _note_history(self, undo: bool) -> str:
        texts = getattr(self, "note_op_text", None)
        return texts(undo) if callable(texts) else ""

    def _history(self, moved: bool, what: str) -> None:
        if not moved:
            self.report(f"nothing to {what}")
            return
        # the split the hint named may be gone -- or back, after undoing the
        # merge-in (U16 review, R3): asked again, once per undo/redo
        hint = self._behind_hint
        self._hint_live = bool(
            hint is not None and hint[0] == self.session.current()
            and compat.previous_behind(self.session, hint[1]) is not None)
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
    # The label keys (1-7, Ctrl+V) are Review's 可见性审查 since task U24:
    # :mod:`tda.ui.app_visreview` owns their slots.

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

