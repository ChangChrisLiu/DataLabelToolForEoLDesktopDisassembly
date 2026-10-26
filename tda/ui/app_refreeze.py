"""「重签这一帧」 / 「重签全部」 in the main window (task U32).

Mixed into :class:`tda.ui.app.MainWindow`.  The Review panel's two buttons
(:attr:`tda.ui.panels.review.ReviewPanel.sigRefreeze`; no key) end here:

1. the session **plans** -- which confirmed frames of the view (or the frame on
   screen) hold rows that differ from today's compile only within the
   tolerance; nothing is written (busy cursor: the frames are compiled);
2. nothing to re-sign: the status line says so, and why (not confirmed; every
   row agrees; the differences are conflicts);
3. otherwise a question that **names the frame count** -- 「重签 n 帧（m 行，
   k px）？」 -- with the overlapping pixels before and after and what is left
   alone;
4. yes: the session writes that plan (one op per frame, signed by this
   annotator) and the status line says what it did.
"""
from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui.panels.review import REFREEZE_FRAME

__all__ = ["RefreezeWindowMixin", "refreeze_question", "refreeze_status", "steps_ranges"]

TITLE = "重签 / re-sign"
NO_VIEW = "没有打开的视图 / no view is open"
NOT_CONFIRMED = "这一帧还没确认：重签只改已确认的帧 / this frame is not confirmed"
NO_CONFIRMED = "这个视图还没有已确认的帧 / no confirmed frame in this view"
NOTHING = "没有要重签的行：已确认的行都和今天的编译一致 / nothing to re-sign"
CANCELLED = "重签已取消 / re-sign cancelled"


def _n(value) -> str:
    return f"{int(value or 0):,}".replace(",", " ")


def steps_ranges(steps: Iterable[int]) -> str:
    """``[17, 18, 19, 41]`` -> ``17–19、41``."""
    values = sorted({int(s) for s in steps})
    runs: list[list[int]] = []
    for step in values:
        if runs and step == runs[-1][1] + 1:
            runs[-1][1] = step
        else:
            runs.append([step, step])
    return "、".join(str(a) if a == b else f"{a}–{b}" for a, b in runs)


def _left_alone(plan: dict) -> list[str]:
    lines = []
    stood = int(plan.get("disputed") or 0) + int(plan.get("open_conflicts") or 0)
    if stood:
        lines.append(f"超出容差（或有开着的冲突）的 {stood} 行不动：它们是冲突，在 Conflicts 里处理。")
    waiting = int(plan.get("deferred") or 0)
    if waiting:
        lines.append(f"{waiting} 行挨着这些冲突的零件，等冲突处理完再重签。")
    gained = int(plan.get("missing") or 0) + int(plan.get("unsigned") or 0)
    if gained:
        lines.append(f"{gained} 个零件在它的帧上没确认过：不代签（帧会回到需复核）。")
    return lines


def refreeze_question(plan: dict) -> list[str]:
    """The confirmation's lines; the first names the frame count."""
    frames = [f for f in plan.get("frames") or () if f.changes]
    lines = [f"重签 {len(frames)} 帧（{plan['rewritten']} 行，{_n(plan['px'])} px）？",
             f"帧：{steps_ranges(f.step for f in frames)}",
             "这些已确认帧里有的行和今天的编译只差容差以内（不算冲突，一直存着旧像素）。"
             "重签把它们改成今天的编译，帧仍是已确认，每帧一个可撤销的 op。"]
    if plan.get("overlap_before") is not None:
        lines.append(f"两个已确认零件共用的像素：{_n(plan['overlap_before'])} → "
                     f"{_n(plan['overlap_after'])} px。")
    return lines + _left_alone(plan)


def refreeze_status(result: dict) -> str:
    """The status line after a re-sign."""
    text = (f"已重签 {result['frames_rewritten']} 帧、{result['rewritten']} 行"
            f"（{_n(result['px'])} px）")
    if result.get("overlap_before") is not None:
        text += f"；重叠 {_n(result['overlap_before'])} → {_n(result['overlap_after'])} px"
    stood = int(result.get("disputed") or 0) + int(result.get("open_conflicts") or 0)
    if stood:
        text += f"；超出容差的 {stood} 行仍是冲突"
    if result.get("deferred"):
        text += f"；{result['deferred']} 行等冲突处理完再重签"
    if result.get("moved"):
        text += f"；{result['moved']} 帧写的时候变了，没改（再按一次）"
    return text + f" / re-signed {result['rewritten']} rows on {result['frames_rewritten']} frames"


class RefreezeWindowMixin:
    """The Review panel's 「重签」 buttons, for the main window."""

    def _init_refreeze(self) -> None:
        self.review.sigRefreeze.connect(self.on_refreeze)

    def ask_refreeze_confirm(self, lines: list[str]) -> bool:
        """The question (``lines``; the first names the frame count); ``True``: go ahead.

        A method so that a test answers it without a modal event loop.
        """
        answer = QMessageBox.question(self, TITLE, "\n".join(lines),
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes

    @S.guard
    def on_refreeze(self, scope: str) -> Optional[dict]:
        """「重签这一帧」 (``"frame"``) or 「重签全部」 (``"all"``): plan, ask, write."""
        from PySide6.QtWidgets import QApplication

        if not compat.is_open(self.session) or not callable(
                getattr(self.session, "refreeze_plan", None)):
            self.report(NO_VIEW)
            return None
        here = int(self.session.current().step)
        steps = [here] if scope == REFREEZE_FRAME else None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            plan = self.session.refreeze_plan(steps)
        finally:
            QApplication.restoreOverrideCursor()
        if not plan["considered"]:
            self.report(NOT_CONFIRMED if scope == REFREEZE_FRAME else NO_CONFIRMED)
            return None
        if not plan["rewritten"]:
            self.report("；".join([NOTHING, *_left_alone(plan)]))
            return None
        if not self.ask_refreeze_confirm(refreeze_question(plan)):
            self.report(CANCELLED)
            return None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = self.session.refreeze(plan)
        finally:
            QApplication.restoreOverrideCursor()
        self.logger.info("refreeze %s: %s", scope,
                         {k: v for k, v in result.items() if k != "frames"})
        self.review.refresh()
        self.review_refreshes += 1
        self.refresh_overlay()
        self.timeline.refresh_statuses()
        self.update_status()
        self.report(refreeze_status(result))
        self.refresh_guidance()
        return result
