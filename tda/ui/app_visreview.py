"""「可见性审查」 in the main window (task U24): the fifth Review tab and its keys.

Mixed into :class:`tda.ui.app.MainWindow`.  While annotating there is no label
key at all (the compiler derives the labels; `` ` `` still says 被线挡住 while
drawing).  In Review, the tab lists the confirmed labels still to audit, one
line per part and run of frames (按段) -- or one per frame (按帧, the panel's
toggle):

* **selecting** a line opens its frame -- through the same gate as any move --
  and selects the part in the instance table (on OAK / RealSense the 主视角
  dock, task U23, follows the selection);
* every key answers the **whole selected line**, as one op of the session
  (one ``Ctrl+Z``), and stamps its rows reviewed: ``Space`` (``Shift+Space``)
  as it is, a digit (``Ctrl+V`` the next value) a label by hand, `` ` `` /
  ``Shift+``` 被线挡住 完全 / 部分, ``Z`` / ``Shift+Z`` 角度看不到 完全 / 部分.
  A 40-frame run is one ``Z``. One frame of a run: the 按帧 listing;
* after an answer the selection moves to the next line and its frame.

On the other Review tabs ``Space`` still accepts the frame, as it did.

Task U36 adds two buttons under the list, no keys: 「从这帧起另算」 cuts the
selected run at the frame on screen (the keys then answer the part from that
frame on), and 「同步到其他视角」 copies the run just answered onto the same
steps of the other views where the part is hidden the same way and nobody has
answered -- offered after an answer when there is something to copy (the
status line says how many frames), one op, one ``Ctrl+Z``. The other views'
suggested reason is the palette's default (its border) and the guide's line.
"""
from __future__ import annotations

from typing import Optional

from tda.ui import app_actions as A
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui import session_api as api
from tda.ui import visreview_text as VT
from tda.ui.class_names import instance_label
from tda.ui.session_api import SessionRefusal

__all__ = ["NOT_IN_REVIEW", "OTHER_TAB", "VisReviewMixin"]

#: A label key outside Review (a stray call; the keys are not bound there).
NOT_IN_REVIEW = ("可见性标签只在复查模式的「可见性审查」里设：点顶部 Review，右侧选「可见性审查」/ "
                 "visibility labels are set in Review's 可见性审查 tab")
#: A verdict key while another Review tab is on top.
OTHER_TAB = "先切到右侧 Review 的「可见性审查」，选一条 / open the 可见性审查 tab first"
#: The frame already disagreed with what was confirmed before the answer (a
#: re-trace elsewhere): the label is written, the conflict is Review's to settle.
CONFLICT_LEFT = "；有的帧原来就和确认时不一样：冲突留在 Conflicts 里"
#: Pixels a label revealed on frames other than the one on screen (U15 review
#: round 4): written, not signed -- their conflicts are Review's.
WITHHELD = "；第 {steps} 帧露出了确认时没有的像素：没代签，冲突留在 Conflicts 里"


class VisReviewMixin:
    """The 「可见性审查」 tab's selection and keys, for the main window."""

    def _init_visreview(self) -> None:
        self.review.sigVisibilityPicked.connect(self.on_visibility_picked)
        self.review.sigSplitRun.connect(self.act_split_run)
        self.review.sigSyncViews.connect(self.act_sync_views)
        #: ``(desktop, view, part, steps)`` of the run last answered, while a
        #: sync of it to the other views is on offer (task U36).
        self._vis_sync: Optional[tuple] = None
        self.follow_visibility_mode()

    def follow_visibility_mode(self) -> None:
        """The session lists the queue only while Review shows it (it reads every row)."""
        live = getattr(self.session, "set_visibility_live", None)
        if callable(live):
            live(self.mode == A.MODE_REVIEW)

    def on_visibility_tab(self) -> bool:
        """Is Review on screen with the 「可见性审查」 tab on top?"""
        return (self.mode == A.MODE_REVIEW
                and self.review.current_queue() == api.QUEUE_VISIBILITY)

    # -- selecting a line ------------------------------------------------------
    @S.guard
    def on_visibility_picked(self, step: int, instance: str) -> None:
        """A line was selected: its frame (through the gate), and its part."""
        if self.mode != A.MODE_REVIEW or not compat.is_open(self.session):
            return
        self._go_to_row(int(step), str(instance))

    def _go_to_row(self, step: int, instance: str) -> bool:
        if int(step) != int(self.session.current().step):
            if not self.leave_frame(lambda: self.session.goto(int(step), force=True)):
                return False
        self.instances.select_instance(str(instance))
        self.refresh_guidance()
        return True

    def _verdict_target(self) -> Optional[tuple[dict, int]]:
        """``(the selected line, the frame it is answered on)``; says why not."""
        if self.mode != A.MODE_REVIEW:
            self.report(NOT_IN_REVIEW)
            return None
        if self.review.current_queue() != api.QUEUE_VISIBILITY:
            self.report(OTHER_TAB)
            return None
        entry = self.review.selected_visibility()
        if entry is None:
            self.report(VT.PICK_A_ROW)
            return None
        steps = [int(s) for s in entry.get("steps") or ()]
        here = int(self.session.current().step)
        step = here if here in steps else steps[0]
        if not self._go_to_row(step, str(entry["instance"])):
            return None
        return entry, step

    # -- the keys ----------------------------------------------------------------
    @S.guard
    def act_vis_review(self, verdict: str, label: Optional[str] = None) -> None:
        """``Space`` / `` ` `` / ``Z`` (and their Shift halves): the whole selected line."""
        if verdict == "computed" and self.mode == A.MODE_REVIEW and not self.on_visibility_tab():
            self.act_confirm()          # the other queues: Space accepts the frame, as before
            return
        target = self._verdict_target()
        if target is None:
            return
        entry, _step = target
        self._answer(entry, verdict, label, [int(s) for s in entry["steps"]])

    @S.guard
    def act_set_visibility(self, value: str) -> None:
        """A digit (``Ctrl`` + digit): this label, by hand, over the selected line."""
        self.act_vis_review("label", value)

    @S.guard
    def act_cycle_visibility(self) -> None:
        """``Ctrl+V``: the value after the line's label, by hand."""
        entry = self.review.selected_visibility() if self.on_visibility_tab() else None
        current = str((entry or {}).get("visibility") or api.VISIBILITY_VALUES[0])
        try:
            index = api.VISIBILITY_VALUES.index(current)
        except ValueError:
            index = -1
        self.act_vis_review("label", api.VISIBILITY_VALUES[(index + 1) % len(
            api.VISIBILITY_VALUES)])

    # -- answering ---------------------------------------------------------------
    def _answer(self, entry: dict, verdict: str, label: Optional[str], steps: list[int]) -> None:
        instance = str(entry["instance"])
        row = self.review.visibility_row()
        try:
            result = self.session.review_visibility(instance, verdict, label, steps=steps)
        except SessionRefusal as refused:
            self.report(f"可见性审查：{refused}")
            return
        name = self._label_of(instance)[0] or instance_label(instance)
        said = label if verdict != "computed" else entry.get("visibility")
        note = VT.verdict_note(name, verdict, said, steps)
        withheld = result.get("withheld") or []
        if withheld:
            note += WITHHELD.format(steps=VT.steps_text(withheld))
        elif result.get("conflicts"):
            note += CONFLICT_LEFT
        self.logger.info("visibility review %s %s %s %s: %s", instance, steps, verdict, label,
                         {k: result.get(k) for k in ("moved", "accepted", "withheld",
                                                     "conflicts")})
        note += self._offer_sync(instance, steps)
        self.review.refresh()
        if not self._walk_on(row):
            note += f"；{VT.DONE}"
        self.refresh_overlay()
        self.timeline.refresh_statuses()
        self.update_status()
        self.report(note)
        self.refresh_guidance()

    # -- 同步到其他视角 / 从这帧起另算 (task U36) ----------------------------------------
    def _offer_sync(self, instance: str, steps: list[int]) -> str:
        """After an answer: offer its sync when another view has frames for it.

        Returns what the status line adds (``""`` when there is nothing to copy).
        """
        plan_of = getattr(self.session, "sync_visibility_plan", None)
        plan = plan_of(instance, steps) if callable(plan_of) else {"count": 0}
        if not plan.get("count"):
            self._vis_sync = None
            self.review.set_sync_offer(None)
            return ""
        key = self.session.current()
        self._vis_sync = (key.desktop, key.view, str(instance), [int(s) for s in steps])
        where = "、".join(f"{view} {len(found)} 帧"
                         for view, found in sorted(plan["views"].items()))
        name = self._label_of(instance)[0] or instance_label(instance)
        self.review.set_sync_offer(f"{name} 第 {VT.steps_text(steps)} 帧 → {where}")
        return VT.SYNC_OFFER.format(n=plan["count"])

    @S.guard
    def act_sync_views(self) -> None:
        """「同步到其他视角」: the run last answered, onto the other views (one op)."""
        offer = getattr(self, "_vis_sync", None)
        if (offer is None or self.mode != A.MODE_REVIEW or not compat.is_open(self.session)
                or (offer[0], offer[1]) != (self.session.current().desktop,
                                            self.session.current().view)):
            self._vis_sync = None
            self.review.set_sync_offer(None)
            self.report(VT.SYNC_IDLE)
            return
        _desktop, _view, instance, steps = offer
        try:
            result = self.session.sync_visibility(instance, steps)
        except SessionRefusal as refused:
            self.report(f"可见性审查：{refused}")
            return
        finally:
            self._vis_sync = None
            self.review.set_sync_offer(None)
        name = self._label_of(instance)[0] or instance_label(instance)
        self.logger.info("visibility sync %s %s: %s", instance, steps,
                         {k: result.get(k) for k in ("written", "skipped", "accepted",
                                                     "withheld", "conflicts")})
        self.review.refresh()
        self.update_status()
        self.report(VT.sync_note(name, result))
        self.refresh_guidance()

    @S.guard
    def act_split_run(self) -> None:
        """「从这帧起另算」: the selected run, cut at the frame on screen."""
        if self.mode != A.MODE_REVIEW or not compat.is_open(self.session):
            self.report(NOT_IN_REVIEW)
            return
        entry = self.review.selected_visibility()
        if entry is None:
            self.report(VT.PICK_A_ROW)
            return
        step = int(self.session.current().step)
        instance = str(entry["instance"])
        try:
            result = self.session.split_visibility_run(instance, step, entry.get("steps") or ())
        except SessionRefusal as refused:
            self.report(f"可见性审查：{refused}")
            return
        # the refresh keeps the part selected at the frame on screen: the line
        # that now starts there (or, merged, the whole run again)
        self.review.refresh()
        picked = self.review.selected_visibility() or {}
        name = self._label_of(instance)[0] or instance_label(instance)
        if result.get("merged"):
            self.report(VT.MERGED.format(step=step))
        else:
            self.report(VT.SPLIT_DONE.format(step=step, label=name,
                                             steps=VT.steps_text(picked.get("steps") or [step])))
        self.refresh_guidance()

    def _walk_on(self, row: int) -> bool:
        """After an answer: the next line -- the one now where the answered one was.

        Its frame (unless the frame on screen is in it) and its part. ``False``
        when the list is empty: the view's review is done.
        """
        if not self.review.select_visibility_row(row):
            return False
        picked = self.review.selected_visibility()
        if picked is None:
            return False
        steps = [int(s) for s in picked.get("steps") or ()]
        here = int(self.session.current().step)
        self._go_to_row(here if here in steps else steps[0], str(picked["instance"]))
        return True
