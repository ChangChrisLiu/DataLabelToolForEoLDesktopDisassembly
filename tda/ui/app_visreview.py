"""「可见性审查」 in the main window (task U24): the fifth Review tab and its keys.

Mixed into :class:`tda.ui.app.MainWindow`.  While annotating there is no label
key at all (the compiler derives the labels; `` ` `` still says 被线挡住 while
drawing).  In Review, the tab lists the confirmed labels still to audit, one
line per part and run of frames:

* **selecting** a line opens its frame -- through the same gate as any move --
  and selects the part in the instance table (on OAK / RealSense the 主视角
  dock, task U23, follows the selection);
* ``Space`` stamps the selected part on the frame on screen as it is,
  ``Shift+Space`` the whole run; a digit (``Ctrl+V`` the next value) sets a
  label by hand, `` ` `` / ``Shift+``` is 被线挡住 完全 / 部分, ``Z`` /
  ``Shift+Z`` is 角度看不到 完全 / 部分.  Every answer is one op of the session
  (``Ctrl+Z``) and stamps the row reviewed;
* after an answer about one frame the walk goes on to the next frame of the
  same run, then to the next line, so a run is a few presses of one key.

On the other Review tabs ``Space`` still accepts the frame, as it did.
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
CONFLICT_LEFT = "；这一帧原来就和确认时不一样：冲突留在 Conflicts 里"


class VisReviewMixin:
    """The 「可见性审查」 tab's selection and keys, for the main window."""

    def _init_visreview(self) -> None:
        self.review.sigVisibilityPicked.connect(self.on_visibility_picked)
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
        """``Space`` / `` ` `` / ``Z`` (and their Shift halves) on the selected line."""
        if verdict == "computed" and self.mode == A.MODE_REVIEW and not self.on_visibility_tab():
            self.act_confirm()          # the other queues: Space accepts the frame, as before
            return
        target = self._verdict_target()
        if target is None:
            return
        entry, step = target
        self._answer(entry, verdict, label, [step])

    @S.guard
    def act_vis_review_run(self) -> None:
        """``Shift+Space``: every frame of the selected line, as it is."""
        target = self._verdict_target()
        if target is None:
            return
        entry, _step = target
        self._answer(entry, "computed", None, [int(s) for s in entry["steps"]])

    @S.guard
    def act_set_visibility(self, value: str) -> None:
        """A digit (``Ctrl`` + digit): this label, by hand, on the selected line's frame."""
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
        try:
            result = self.session.review_visibility(
                instance, verdict, label, steps=steps if verdict == "computed" else None)
        except SessionRefusal as refused:
            self.report(f"可见性审查：{refused}")
            return
        name = self._label_of(instance)[0] or instance_label(instance)
        said = label if verdict != "computed" else entry.get("visibility")
        note = VT.verdict_note(name, verdict, said, steps)
        if result.get("conflicts"):
            note += CONFLICT_LEFT
        self.logger.info("visibility review %s %s %s %s: %s", instance, steps, verdict, label,
                         {k: result.get(k) for k in ("moved", "accepted", "withheld",
                                                     "conflicts")})
        self.review.refresh()
        if not self._walk_on(entry, steps):
            note += f"；{VT.DONE}"
        self.refresh_overlay()
        self.timeline.refresh_statuses()
        self.update_status()
        self.report(note)
        self.refresh_guidance()

    def _walk_on(self, entry: dict, done: list[int]) -> bool:
        """After an answer: the next frame of the same run, else the next line.

        ``False`` when the list is empty: the view's review is done.
        """
        instance = str(entry["instance"])
        run = [int(s) for s in entry.get("steps") or ()]
        later = [s for s in run if s > max(done)]
        earlier = [s for s in run if s < min(done)]
        wanted = later[:1] or earlier[-1:]
        if not self.review.select_visibility(instance, wanted or run):
            return False
        picked = self.review.selected_visibility()
        if picked is None:
            return False
        steps = [int(s) for s in picked.get("steps") or ()]
        here = int(self.session.current().step)
        if picked.get("instance") == instance and wanted and wanted[0] in steps:
            step = wanted[0]
        else:
            step = here if here in steps else steps[0]
        self._go_to_row(step, str(picked["instance"]))
        return True
