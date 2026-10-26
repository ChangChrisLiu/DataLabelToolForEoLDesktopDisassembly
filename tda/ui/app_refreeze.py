"""「重签这一帧」 / 「重签全部」 in the main window (task U32).

Mixed into :class:`tda.ui.app.MainWindow`.  The Review panel's two buttons
(:attr:`tda.ui.panels.review.ReviewPanel.sigRefreeze`; no key) end here:

1. the session **plans** -- which confirmed frames of the view (or the frame on
   screen) hold rows that differ from today's compile only within the
   tolerance; nothing is written (busy cursor: the frames are compiled);
2. nothing to re-sign: the status line says so, and why (not confirmed; every
   row agrees; the differences are conflicts);
3. otherwise a question that **names the frame count** -- 「重签 n 帧（m 行，
   k px）？」 -- with the overlapping pixels before and after, the labels that
   cross 看得见 / 看不见 (frame and part), what is left alone, and how to undo
   it (the command line; ``Ctrl+Z`` does not);
4. yes: the database is backed up with the command line's own helper
   (:func:`tda.cli_common.safety_backup`) -- no copy, nothing written -- then
   the session writes that plan (one op per frame, signed by this annotator)
   and the status line says what it did.
"""
from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from tda.cli_common import safety_backup
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
BACKUP_FAILED = ("备份失败，什么都没写（paths.yaml 的 backup_dir 不可用：原始数据盘没接上？）/ "
                 "backup failed: nothing was re-signed")


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
    drifted = int(plan.get("drifted") or 0)
    if drifted:
        lines.append(f"{drifted} 行重签过、离人签的那一版已超出容差：不再重签，是冲突（刷新后在 Conflicts 里）。")
    waiting = int(plan.get("deferred") or 0)
    if waiting:
        lines.append(f"{waiting} 行挨着这些冲突的零件，等冲突处理完再重签。")
    gained = int(plan.get("missing") or 0) + int(plan.get("unsigned") or 0)
    if gained:
        lines.append(f"{gained} 个零件在它的帧上没确认过：不代签（帧会回到需复核）。")
    return lines


#: How many label flips the question names before it says "…".
FLIPS_NAMED = 8


def refreeze_question(plan: dict, where: Optional[tuple[int, str]] = None) -> list[str]:
    """The confirmation's lines; the first names the frame count.

    The labels that cross 看得见 / 看不见 are named with their frame and part
    (review N3), and the last line says how to take it back -- the command
    line, after the window is closed: ``Ctrl+Z`` does not (review N7).
    ``where`` is ``(desktop, view)`` for that line.
    """
    frames = [f for f in plan.get("frames") or () if f.changes]
    lines = [f"重签 {len(frames)} 帧（{plan['rewritten']} 行，{_n(plan['px'])} px）？",
             f"帧：{steps_ranges(f.step for f in frames)}",
             "这些已确认帧里有的行和今天的编译只差容差以内（不算冲突，一直存着旧像素）。"
             "重签把它们改成今天的编译，帧仍是已确认；先备份，每帧记一个 op。"]
    if plan.get("overlap_before") is not None:
        lines.append(f"两个已确认零件共用的像素：{_n(plan['overlap_before'])} → "
                     f"{_n(plan['overlap_after'])} px。")
    flips = plan.get("flips") or []
    if flips:
        named = "；".join(f"第 {step} 帧 {inst}：{old} → {new}"
                         for step, inst, old, new in flips[:FLIPS_NAMED])
        more = f"；…共 {len(flips)} 行" if len(flips) > FLIPS_NAMED else ""
        lines.append(f"标签在看得见 / 看不见之间翻转（之后在可见性审查里看一眼）：{named}{more}。")
    lines += _left_alone(plan)
    desktop, view = where if where is not None else ("N", "V")
    lines.append(f"撤销不能用 Ctrl+Z：关掉应用后 python -m tda.cli refreeze --desktop {desktop} "
                 f"--view {view} --undo last")
    return lines


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
    if result.get("flips"):
        text += f"；{len(result['flips'])} 个标签在看得见 / 看不见之间翻转（可见性审查里看一眼）"
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
        where = (int(self.session.desktop), str(self.session.view))
        if not self.ask_refreeze_confirm(refreeze_question(plan, where)):
            self.report(CANCELLED)
            return None
        # the same safety copy the command line takes before it rewrites
        # confirmed rows (review B2); no copy, no write
        if not safety_backup(self.paths, self.session.db, "refreeze",
                             "re-signing rewrites confirmed rows"):
            self.report(BACKUP_FAILED)
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
